import json
import os
import re

def merge_specific_hh_files(output_file="general_vac.json"):
    """
    Объединяет конкретные файлы hh_vacancies_all*.json
    """
    # Создаем список файлов в правильном порядке
    files_to_merge = []
    
    # Основной файл
    if os.path.exists("hh_vacancies_all.json"):
        files_to_merge.append("hh_vacancies_all.json")
    
    # Файлы с номерами в скобках
    for i in range(1, 20):  # проверяем до (19)
        filename = f"hh_vacancies_all ({i}).json"
        if os.path.exists(filename):
            files_to_merge.append(filename)
    
    if not files_to_merge:
        print("Файлы hh_vacancies_all*.json не найдены!")
        return
    
    print("Объединяю файлы:")
    for file in files_to_merge:
        print(f"  - {file}")
    
    # Объединяем данные
    all_vacancies = []
    
    for file_path in files_to_merge:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                all_vacancies.extend(data)
                print(f"    Добавлено {len(data)} вакансий из {file_path}")
            else:
                print(f"    Ошибка: {file_path} не является массивом")
                
        except Exception as e:
            print(f"    Ошибка при чтении {file_path}: {e}")
    
    # Сохраняем результат
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_vacancies, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Готово! Создан файл: {output_file}")
    print(f"✓ Всего вакансий: {len(all_vacancies)}")
    print(f"✓ Объединено файлов: {len(files_to_merge)}")

if __name__ == "__main__":
    merge_specific_hh_files("general_vac.json")