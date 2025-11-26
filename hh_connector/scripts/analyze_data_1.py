from db.mongo_connection import get_db
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def profession_distribution_plot(role_ids=None, table_path="professions_table.csv", plot_path="professions_plot.png"):
    """
    Строит таблицу и график распределения профессий по процентам.
    
    Параметры:
    - role_ids: список id профессий, чтобы фильтровать (если None — все вакансии)
    - table_path: путь для сохранения CSV таблицы
    - plot_path: путь для сохранения графика PNG
    
    Возвращает pandas.DataFrame с колонками ['Профессия', 'Количество', 'Процент']
    """
    db = get_db()
    coll = db['vacancies']

    # --- агрегируем профессии ---
    match_stage = {"$match": {}} if not role_ids else {"$match": {"professional_roles.id": {"$in": role_ids}}}
    
    pipeline = [
        match_stage,
        {"$unwind": "$professional_roles"},
        {"$group": {"_id": "$professional_roles.name", "count": {"$sum": 1}}},  # считаем вакансии по профессиям
        {"$sort": {"count": -1}}
    ]

    results = list(coll.aggregate(pipeline))
    if not results:
        print("Нет данных для построения графика распределения профессий.")
        return pd.DataFrame(columns=["Профессия", "Количество", "Процент"])

    df = pd.DataFrame(results)
    df = df.rename(columns={"_id": "Профессия"})
    df['Процент'] = df['count'] / df['count'].sum() * 100
    df = df.sort_values("count", ascending=False)

    # --- сохраняем таблицу ---
    df.to_csv(table_path, index=False, encoding="utf-8-sig")
    print(f"Таблица сохранена: {table_path}")

    # --- строим график ---
    plt.figure(figsize=(10, 6))
    plt.barh(df["Профессия"], df["Процент"], color="skyblue", edgecolor="black")
    plt.gca().invert_yaxis()
    plt.xlabel("Процент (%)")
    plt.ylabel("Профессия")
    plt.title("Распределение профессий по базе вакансий")
    plt.grid(axis="x", alpha=0.3)

    # подписи с процентами
    for i, v in enumerate(df["Процент"]):
        plt.text(v + 0.5, i, f"{v:.1f}%", va='center')

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"График сохранён: {plot_path}")

    return df

def analyze_professions(role_ids=None, table_path="professions_table.csv", plot_path="professions_plot.png"):
    """
    Удобная обёртка для вызова функции.
    """
    return profession_distribution_plot(role_ids=role_ids, table_path=table_path, plot_path=plot_path)
