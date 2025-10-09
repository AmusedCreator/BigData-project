from db.mongo_connection import get_db
import pandas as pd
import matplotlib.pyplot as plt

def top_cities_plot(limit: int = 10, save_path: str = 'top_cities.png'):
    db = get_db()
    coll = db['vacancies']
    pipeline = [
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
    ax = df['count'].plot.barh()
    ax.invert_yaxis()
    ax.set_xlabel('Количество вакансий')
    ax.set_ylabel('Город')
    ax.set_title('Топ городов по количеству вакансий')
    plt.tight_layout()
    plt.savefig(save_path)
    print('Сохранён график:', save_path)
    plt.close()

def analyze_all():
    top_cities_plot()

if __name__ == '__main__':
    analyze_all()

