from db.mongo_connection import get_db
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def industry_distribution_plot(limit: int = 10, save_path: str = 'ib_industries.png'):
    """
    Анализ отраслей работодателей, которые ищут специалистов ИБ (professional_roles.id = "116").
    Работает с MongoDB, даже если версия не поддерживает $isObject/$isArray.
    """
    db = get_db()
    vac_coll = db['vacancies']

    pipeline = [
        {"$match": {"professional_roles.id": "116"}},
        # подгружаем полную запись работодателя
        {"$lookup": {
            "from": "employers",
            "localField": "employer.id",
            "foreignField": "id",
            "as": "employer_full"
        }},
        {"$addFields": {"employer_full": {"$arrayElemAt": ["$employer_full", 0]}}},
        # сформируем поле industries_raw — предпочитаем employer_full.industries, иначе employer.industry, иначе null
        {"$addFields": {
            "industries_raw": {
                "$cond": [
                    {"$ne": ["$employer_full.industries", None]},
                    "$employer_full.industries",
                    {"$cond": [
                        {"$ne": ["$employer.industry", None]},
                        "$employer.industry",
                        None
                    ]}
                ]
            }
        }},
        # Приведём industries_raw к массиву industries_arr:
        # - если это массив -> оставляем как есть
        # - если это объект или строка -> оборачиваем в массив [industries_raw]
        # - иначе -> пустой массив
        {"$addFields": {
            "industries_arr": {
                "$cond": [
                    {"$eq": [{"$type": "$industries_raw"}, "array"]},
                    "$industries_raw",
                    {"$cond": [
                        {"$in": [{"$type": "$industries_raw"}, ["object", "string", "int", "double"]]},
                        ["$industries_raw"],
                        []
                    ]}
                ]
            }
        }},
        # разворачиваем массив (если пустой — документ будет отброшен)
        {"$unwind": {"path": "$industries_arr", "preserveNullAndEmptyArrays": False}},
        # теперь industries_arr может быть объект (с полем name) или строкой; извлекаем имя
        {"$addFields": {
            "industry_name": {
                "$cond": [
                    {"$eq": [{"$type": "$industries_arr"}, "object"]},
                    {"$ifNull": ["$industries_arr.name", "$industries_arr"]},
                    {"$ifNull": ["$industries_arr", ""]}  # если строка — берём строку
                ]
            }
        }},
        # фильтрация пустых/нулевых названий
        {"$match": {"industry_name": {"$ne": "", "$ne": None}}},
        # группируем по названию отрасли
        {"$group": {"_id": "$industry_name", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ]

    try:
        results = list(vac_coll.aggregate(pipeline))
    except Exception as e:
        print("Ошибка при выполнении aggregation pipeline:", type(e), e)
        print("Возможная причина — старая версия MongoDB или неподдерживаемая стадия/оператор.")
        return

    df = pd.DataFrame(results)
    if df.empty:
        print("Нет данных для выбранного фильтра (professional_roles.id = '116' или отсутствуют industries).")
        return

    df = df.rename(columns={'_id': 'industry'}).set_index('industry')
    # исключаем возможные пустые индексы
    df = df[~df.index.isnull() & (df.index != '')]
    if df.empty:
        print("После фильтрации данных по названиям отраслей ничего не осталось.")
        return

    df['percent'] = df['count'] / df['count'].sum() * 100
    df = df.sort_values('count', ascending=True)

    # --- Рисуем график доли в процентах ---
    plt.figure(figsize=(10, max(6, 0.4 * len(df))))
    ax = df['percent'].plot.barh(edgecolor='black', color='skyblue')
    ax.invert_yaxis()
    ax.set_xlabel('Доля вакансий, %')
    ax.set_ylabel('')
    ax.set_title('Доля вакансий по отраслям (ИБ), %')

    # Отображаем подписи оси Y слева полностью (по возможности)
    ax.set_yticklabels([label.get_text()[:40] + ('...' if len(label.get_text()) > 40 else '') for label in ax.get_yticklabels()])

    # Подписи процентов
    for i, pct in enumerate(df['percent']):
        ax.text(pct if pct > 0 else 0.1, i, f' {pct:.1f}%', va='center', fontsize=9)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    industry_distribution_plot(limit=12, save_path='ib_industries_top12.png')
