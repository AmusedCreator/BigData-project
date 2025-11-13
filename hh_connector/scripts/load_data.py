"""
Загружает JSON array (массив JSON-объектов) в коллекцию MongoDB.

- Работает пакетами (batch_size) --- не держит весь файл в памяти.
- Создает уникальный индекс по полю 'id' и игнорирует дубликаты.
"""
from db.mongo_connection import get_db  
import json  
from pymongo.errors import BulkWriteError


# Определяем нужные поля для загрузки
needed_fields = {
    "id",
    "name",
    "area.name",
    "employer.name",
    "employer.industry",
    "professional_roles",
    "specializations",
    "employment.name",
    "schedule.name",
    "salary.from",
    "salary.to",
    "salary.currency",
    "snippet.requirement",
    "snippet.responsibility",
    "description",
    "key_skills",
    "published_at",
    "employment_form",
    "experience",
}

def filter_document(doc):
    """Фильтрует документ, оставляя только нужные поля"""
    filtered = {}
    
    for field in needed_fields:
        # Обрабатываем вложенные поля (с точками)
        if '.' in field:
            # Для вложенных полей типа "area.name", "salary.from"
            parts = field.split('.')
            current = doc
            try:
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        current = None
                        break
                if current is not None:
                    # Создаем вложенную структуру в целевом документе
                    target = filtered
                    for part in parts[:-1]:  # Все части кроме последней
                        if part not in target:
                            target[part] = {}
                        target = target[part]
                    target[parts[-1]] = current
            except (TypeError, KeyError):
                continue
        else:
            # Для простых полей
            if field in doc:
                filtered[field] = doc[field]
    
    return filtered

def load_ndjson_to_mongo(file_path: str, collection_name: str = 'vacancies', batch_size: int = 1000):
    db = get_db()
    coll = db[collection_name]
    
    # Создаем уникальный индекс по полю 'id' (если его еще нет)
    try:
        coll.create_index('id', unique=True)
        print("Создан уникальный индекс по полю 'id'")
    except Exception as e:
        print(f"Индекс по полю 'id' уже существует или ошибка: {e}")

    total_processed = 0
    inserted = 0
    duplicates_skipped = 0
    batch = []

    try:
        # Читаем и парсим весь файл как JSON массив
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Загружено {len(data)} документов из JSON массива")
        
        # Обрабатываем каждый документ в массиве
        for original_doc in data:
            # ФИЛЬТРУЕМ документ - оставляем только нужные поля
            filtered_doc = filter_document(original_doc)
            
            # Пропускаем пустые документы (если нет ни одного нужного поля)
            if not filtered_doc:
                continue
                
            batch.append(filtered_doc)
            total_processed += 1
            
            if len(batch) >= batch_size:
                inserted_in_batch, duplicates_in_batch = _insert_batch(coll, batch)
                inserted += inserted_in_batch
                duplicates_skipped += duplicates_in_batch
                batch = []
                print(f"Обработано: {total_processed}, Вставлено: {inserted}, Дубликатов: {duplicates_skipped}")

        # Вставляем остаток
        if batch:
            inserted_in_batch, duplicates_in_batch = _insert_batch(coll, batch)
            inserted += inserted_in_batch
            duplicates_skipped += duplicates_in_batch

    except json.JSONDecodeError as e:
        print(f'Ошибка при парсинге JSON файла: {e}')
        return
    except Exception as e:
        print(f'Общая ошибка при загрузке файла: {e}')
        return

    print(f"Готово. Обработано документов: {total_processed}")
    print(f"Вставлено документов: {inserted}, Пропущено дубликатов: {duplicates_skipped}")

    # Проверяем математику
    if total_processed != (inserted + duplicates_skipped):
        print(f"Расхождение: {total_processed} != {inserted} + {duplicates_skipped}")

    # Финальная проверка
    total_in_db = coll.count_documents({})
    print(f"Всего документов в коллекции: {total_in_db}")

def _insert_batch(collection, batch):
    """Вставляет пачку документов, обрабатывает дубликаты"""
    try:
        result = collection.insert_many(batch, ordered=False)
        return len(result.inserted_ids), 0
        
    except BulkWriteError as bwe:
        write_errors = bwe.details['writeErrors']
        successful_inserts = bwe.details['nInserted']
        duplicates_count = len([err for err in write_errors if err['code'] == 11000])
        
        print(f"Пакет: вставлено {successful_inserts}, дубликатов {duplicates_count} (всего {len(batch)} документов)")
        return successful_inserts, duplicates_count
        
    except Exception as e:
        print(f'Ошибка при вставке пакета: {e}')
        return 0, 0

if __name__ == '__main__':
    import sys
    import os
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Правильный путь к файлу данных
        path = os.path.join('..', 'data', 'vacancies_it.json')
    load_ndjson_to_mongo(path)