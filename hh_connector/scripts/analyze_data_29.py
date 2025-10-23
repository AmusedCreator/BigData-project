from db.mongo_connection import get_db
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from collections import Counter
import re
from collections import defaultdict
#warnings.filterwarnings('ignore')

def experience_levels_plot(save_path: str = 'experience_levels.png'):
    """
    Анализирует распределение вакансий по уровням позиций
    и определяет наиболее частый уровень запроса
    """
    # Извлечение данных из MongoDB
    db = get_db()
    coll = db['vacancies']

    # data = list(coll.find({}, {'vacancy_name': 1, 'description': 1, 'experience': 1}))

    """"# Проверим структуру одной записи
    sample_doc = coll.find_one()
    print("Пример документа:")
    print(sample_doc)

    # Посмотрим на все поля в коллекции
    all_fields = set()
    for doc in coll.find().limit(5):
        all_fields.update(doc.keys())
    print("\nВсе поля в коллекции:")
    print(all_fields)"""
    data = list(coll.find({}, {
        'name': 1,  # название вакансии
        'snippet.requirement': 1,  # требования (часть описания)
        'snippet.responsibility': 1,  # обязанности (часть описания)
        'experience': 1,  # опыт (если есть в данных)
        '_id': 0
    }))

    print(f"Извлечено {len(data)} записей")
    if data:
        print("Первая запись:")
        print(data[0])

    df = pd.DataFrame(data)
    print(f"\nDataFrame shape: {df.shape}")
    print(df.head())

    print(f"Извлечено {len(data)} записей")
    if data:
        print("Первая запись:")
        print(data[0])

    df = pd.DataFrame(data)
    print(f"\nDataFrame shape: {df.shape}")
    print(df.head())

    # Функция для определения уровня позиции
    def detect_level(text):
        """
        Определяет уровень позиции на основе текста (название + требования + обязанности)
        """
        if not text or pd.isna(text):
            return 'Не определен'

        text = str(text).lower()

        # Паттерны для определения уровней (от старшего к младшему для приоритета)
        patterns = {
            'lead': r'\blead\b|\bруководитель\b|\bhead\b|\bуправление\b|\bteam lead\b',
            'senior': r'\bsenior\b|\bсеньор\b|\bстарший\b|\bsr\.\b|\bопыт\s*[5-9]|\bопыт\s*[1-9][0-9]',
            'middle': r'\bmiddle\b|\bмиддл\b|\bмладший\b|\bmid\b|\bопыт\s*[1-4]',
            'junior': r'\bjunior\b|\bджуниор\b|\bначинающий\b|\bстажер\b|\bintern\b|\bбез\s*опыта\b|\bjun\b'
        }

        for level, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return level.capitalize()

        return 'Не определен'

    # Создаем объединенный текст для анализа
    def combine_text(row):
        """Объединяет все текстовые поля для анализа"""
        text_parts = []

        if 'name' in row and row['name']:
            text_parts.append(str(row['name']))

        if 'snippet' in row and row['snippet']:
            snippet = row['snippet']
            if 'requirement' in snippet and snippet['requirement']:
                text_parts.append(str(snippet['requirement']))
            if 'responsibility' in snippet and snippet['responsibility']:
                text_parts.append(str(snippet['responsibility']))

        if 'experience' in row and row['experience']:
            text_parts.append(str(row['experience']))

        return ' '.join(text_parts)

    # Применяем анализ к данным
    print("\n" + "=" * 50)
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ УРОВНЕЙ ПОЗИЦИЙ")
    print("=" * 50)

    # Создаем столбец с объединенным текстом
    df['combined_text'] = df.apply(combine_text, axis=1)

    # Определяем уровень для каждой вакансии
    df['level'] = df['combined_text'].apply(detect_level)

    # Анализируем распределение
    level_distribution = df['level'].value_counts()
    level_percentage = df['level'].value_counts(normalize=True) * 100

    """print("\nРаспределение вакансий по уровням:")
    print("-" * 40)
    for level in ['Junior', 'Middle', 'Senior', 'Lead', 'Не определен']:
        if level in level_distribution:
            count = level_distribution[level]
            percent = level_percentage[level]
            print(f"{level:<12}: {count:>4} вакансий ({percent:.1f}%)")"""
    # Подготовка данных для визуализации
    plot_data = level_distribution.reset_index()
    plot_data.columns = ['Уровень', 'Количество']

    # Создаем график
    plt.figure(figsize=(10, 6))
    bars = plt.bar(plot_data['Уровень'], plot_data['Количество'],
                   color=['#2E8B57', '#4682B4', '#DAA520', '#CD5C5C', '#808080'])

    plt.title('Распределение вакансий в ИБ по уровням позиций', fontsize=14, fontweight='bold')
    plt.xlabel('Уровень позиции', fontsize=12)
    plt.ylabel('Количество вакансий', fontsize=12)

    # Добавляем подписи значений на столбцах
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}', ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    print('Сохранён график:', save_path)
    plt.close()

def forecast_cybersecurity_demand():
    def get_vacancies_time_series():
        """
        Извлекает временной ряд количества вакансий по ИБ
        """
        db = get_db()
        coll = db['vacancies']

        # Агрегация для получения месячной статистики вакансий
        # Получаем все документы с нужными полями
        cursor = coll.find(
            {},
            {'published_at': 1}
        ).sort('published_at', 1)

        # Собираем все даты
        dates = []
        for doc in cursor:
            if 'published_at' in doc and doc['published_at']:
                dates.append(doc['published_at'])

        # Создаем DataFrame
        df = pd.DataFrame({'date': dates})
        df['date'] = pd.to_datetime(df['date'])

        # Агрегируем по месяцам
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_counts = df.groupby('year_month').size()

        # Преобразуем обратно в datetime
        monthly_ts = monthly_counts.to_timestamp()



        return pd.DataFrame({'count': monthly_ts})

    def prepare_time_series(df):
        """
        Подготовка временного ряда для анализа
        """
        # Ресемплирование до месячных данных
        monthly_ts = df['count'].resample('M').sum()

        # Проверка на стационарность
        def check_stationarity(timeseries):
            result = adfuller(timeseries.dropna())
            print(f'ADF Statistic: {result[0]:.3f}')
            print(f'p-value: {result[1]:.3f}')
            if result[1] <= 0.05:
                print("Ряд стационарен")
                return True
            else:
                print("Ряд нестационарен")
                return False

        print("Проверка стационарности исходного ряда:")
        is_stationary = check_stationarity(monthly_ts)

        # Если ряд нестационарен, применяем дифференцирование
        if not is_stationary:
            monthly_ts_diff = monthly_ts.diff().dropna()
            print("\nПосле дифференцирования:")
            check_stationarity(monthly_ts_diff)
        else:
            monthly_ts_diff = monthly_ts

        return monthly_ts, monthly_ts_diff

    def analyze_seasonality(monthly_ts):
        """
        Анализ сезонности временного ряда
        """
        print("\nАнализ сезонности:")
        decomposition = seasonal_decompose(monthly_ts, model='additive', period=12)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        decomposition.observed.plot(ax=axes[0], title='Исходный ряд')
        decomposition.trend.plot(ax=axes[1], title='Тренд')
        decomposition.seasonal.plot(ax=axes[2], title='Сезонность')
        decomposition.resid.plot(ax=axes[3], title='Остатки')
        plt.tight_layout()
        plt.show()

        return decomposition

    def find_best_sarima_params(monthly_ts):
        """
        Поиск оптимальных параметров для SARIMA с использованием AIC
        """
        # Упрощенный поиск параметров (для полного поиска нужно больше времени)
        best_aic = np.inf
        best_order = None
        best_seasonal_order = None

        # Ограниченный набор параметров для демонстрации
        p_values = [0, 1]
        d_values = [1]
        q_values = [0, 1]
        P_values = [0, 1]
        D_values = [1]
        Q_values = [0, 1]

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                try:
                                    model = SARIMAX(monthly_ts,
                                                    order=(p, d, q),
                                                    seasonal_order=(P, D, Q, 12),
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                                    results = model.fit(disp=False)

                                    if results.aic < best_aic:
                                        best_aic = results.aic
                                        best_order = (p, d, q)
                                        best_seasonal_order = (P, D, Q, 12)
                                        print(
                                            f"Новые лучшие параметры: SARIMA{best_order}x{best_seasonal_order} AIC: {best_aic:.2f}")
                                except:
                                    continue

        print(f"\nОптимальные параметры: SARIMA{best_order}x{best_seasonal_order}")
        print(f"Лучший AIC: {best_aic:.2f}")

        return best_order, best_seasonal_order

    def build_sarima_model(monthly_ts, order, seasonal_order):
        """
        Построение и обучение модели SARIMA
        """
        model = SARIMAX(monthly_ts,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)

        fitted_model = model.fit(disp=False)

        print("\nСтатистика модели:")
        print(fitted_model.summary())

        return fitted_model

    def forecast_demand(model, monthly_ts, periods=12):
        """
        Прогнозирование спроса на следующий год
        """
        # Прогноз на указанное количество периодов
        forecast = model.get_forecast(steps=periods)
        forecast_ci = forecast.conf_int()

        # Создание DataFrame с прогнозом
        last_date = monthly_ts.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                       periods=periods, freq='M')

        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'predicted_count': forecast.predicted_mean.values,
            'lower_ci': forecast_ci.iloc[:, 0].values,
            'upper_ci': forecast_ci.iloc[:, 1].values
        }).set_index('date')

        return forecast_df, forecast

    def plot_results(monthly_ts, forecast_df, model):
        """
        Визуализация результатов
        """
        plt.figure(figsize=(15, 8))

        # Исторические данные
        plt.plot(monthly_ts.index, monthly_ts.values,
                 label='Исторические данные', color='blue', linewidth=2)

        # Прогноз
        plt.plot(forecast_df.index, forecast_df['predicted_count'],
                 label='Прогноз', color='red', linewidth=2, linestyle='--')

        # Доверительный интервал
        plt.fill_between(forecast_df.index,
                         forecast_df['lower_ci'],
                         forecast_df['upper_ci'],
                         color='red', alpha=0.1, label='95% доверительный интервал')

        plt.title('Прогноз спроса на специалистов по ИБ на следующий год', fontsize=14)
        plt.xlabel('Дата')
        plt.ylabel('Количество вакансий')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    print("=== Прогноз спроса на специалистов по ИБ ===\n")

    # 1. Загрузка данных
    print("1. Загрузка данных из MongoDB...")
    df = get_vacancies_time_series()
    print(f"Загружено {len(df)} записей")
    print("Временной ряд количества вакансий по месяцам:")
    print(df)
    print("\n" + "=" * 50)

    # 2. Подготовка временного ряда
    print("\n2. Подготовка временного ряда...")
    monthly_ts, monthly_ts_diff = prepare_time_series(df)
    print("Стационарный и нестационарный временные ряды:")
    print(monthly_ts)
    print("\n" + "=" * 50)
    print(monthly_ts_diff)

    """"# 3. Анализ сезонности
    print("\n3. Анализ сезонности...")
    decomposition = analyze_seasonality(monthly_ts)

    # 4. Поиск оптимальных параметров SARIMA
    print("\n4. Поиск оптимальных параметров SARIMA...")
    best_order, best_seasonal_order = find_best_sarima_params(monthly_ts)

    # 5. Построение модели
    print("\n5. Построение модели SARIMA...")
    model = build_sarima_model(monthly_ts, best_order, best_seasonal_order)

    # 6. Прогнозирование
    print("\n6. Прогнозирование спроса на следующий год...")
    forecast_df, forecast = forecast_demand(model, monthly_ts, periods=12)

    # 7. Визуализация результатов
    print("\n7. Визуализация результатов...")
    plot_results(monthly_ts, forecast_df, model)

    # 8. Вывод прогноза
    print("\n8. Прогноз на следующий год:")
    print("=" * 50)
    for date, row in forecast_df.iterrows():
        print(f"{date.strftime('%Y-%m')}: {row['predicted_count']:.0f} вакансий "
              f"(95% ДИ: {row['lower_ci']:.0f} - {row['upper_ci']:.0f})")

    # 9. Анализ тренда
    print("\n9. Анализ тренда:")
    total_forecast = forecast_df['predicted_count'].sum()
    average_monthly = forecast_df['predicted_count'].mean()
    growth_rate = ((forecast_df['predicted_count'].iloc[-1] - monthly_ts.iloc[-1]) / monthly_ts.iloc[-1]) * 100

    print(f"Общий прогнозируемый спрос за год: {total_forecast:.0f} вакансий")
    print(f"Среднемесячный спрос: {average_monthly:.0f} вакансий")
    print(f"Темп роста к последнему месяцу: {growth_rate:.1f}%")"""


def analyze_all():
    # top_cities_plot()
    experience_levels_plot()
    #show_fields()
    #ib_vacancies_filter()
    #forecast_cybersecurity_demand()



if __name__ == '__main__':
    analyze_all()
