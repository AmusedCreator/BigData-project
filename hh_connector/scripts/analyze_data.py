from db.mongo_connection import get_db
import pandas as pd
import matplotlib.pyplot as plt

def top_cities_plot(limit: int = 4, save_path: str = 'top_cities.png'):
    """
    Строит два графика:
    1. Количество вакансий (логарифмическая шкала)
    2. Доля вакансий в процентах
    Только для специалистов по информационной безопасности (professional_roles.id = 116)
    """
    db = get_db()
    coll = db['vacancies']

    pipeline = [
        {"$match": {"professional_roles.id": "116"}},
        {"$group": {"_id": "$area.name", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ]

    results = list(coll.aggregate(pipeline))
    df = pd.DataFrame(results)
    if df.empty:
        print('Нет данных для построения top_cities_plot')
        return

    df = df.rename(columns={'_id': 'city'})
    df = df.set_index('city')
    df['percent'] = df['count'] / df['count'].sum() * 100

    # --- Создаём два графика рядом ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1️ Абсолютное количество (логарифмическая шкала)
    df['count'].plot.barh(ax=axes[0], color='skyblue', edgecolor='black', logx=True)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Количество вакансий (логарифмическая шкала)')
    axes[0].set_ylabel('Город')
    axes[0].set_title('Количество вакансий ИБ по городам')

    # 2️ Проценты
    df['percent'].plot.barh(ax=axes[1], color='lightgreen', edgecolor='black')
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Доля вакансий, %')
    axes[1].set_ylabel('Город')
    axes[1].set_yticklabels([])  # убираем повторные названия городов
    axes[1].set_title('Доля вакансий ИБ по городам (%)')

    plt.tight_layout()
    plt.savefig(save_path)
    print('Сохранён график:', save_path)
    plt.close()

def employment_forms_plot(save_path: str = 'employment_forms_percent.png'):
    """
    Анализ доли форм занятости (в %) для специалистов по информационной безопасности (professional_roles.id = 116)
    """
    db = get_db()
    coll = db['vacancies']

    pipeline = [
        {"$match": {"professional_roles.id": "116"}},
        {"$group": {"_id": "$employment.name", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]

    results = list(coll.aggregate(pipeline))
    df = pd.DataFrame(results)

    if df.empty:
        print('Нет данных для построения employment_forms_plot')
        return

    df = df.rename(columns={'_id': 'employment_form'})
    df = df.set_index('employment_form')
    df['percent'] = df['count'] / df['count'].sum() * 100

    # --- Один график: доля вакансий (%)
    plt.figure(figsize=(7, 5))
    df['percent'].plot.barh(color='mediumseagreen', edgecolor='black')
    plt.gca().invert_yaxis()
    plt.xlabel('Доля вакансий, %')
    plt.ylabel('Форма занятости')
    plt.title('Доля вакансий по формам занятости (ИБ)')
    plt.tight_layout()
    plt.savefig(save_path)
    print('Сохранён график:', save_path)
    plt.close()
    
def analyze_all():
    top_cities_plot()
    employment_forms_plot()

if __name__ == '__main__':
    analyze_all()

