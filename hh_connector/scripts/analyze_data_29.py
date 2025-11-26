from db.mongo_connection import get_db
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import numpy as np
#from pmdarima import auto_arima
from collections import Counter
import re
from collections import defaultdict
from datetime import datetime, timedelta
import logging
#warnings.filterwarnings('ignore')

def experience_levels_plot(save_path: str = 'experience_levels.png'):
    """
    Анализирует распределение вакансий по уровням позиций
    и определяет наиболее частый уровень запроса
    """
    # Извлечение данных из MongoDB
    db = get_db()
    coll = db['vacancies']

    # Отбор вакансий с профессиональной ролью 116
    data = list(coll.find({
        'professional_roles.id': '116'  # Фильтр по коду профессии 116
    }, {
        'name': 1,  # название вакансии
        'snippet.requirement': 1,  # требования (часть описания)
        'snippet.responsibility': 1,  # обязанности (часть описания)
        'experience': 1,  # опыт (если есть в данных)
        'professional_roles': 1,  # профессиональные роли (для проверки)
        '_id': 0
    }))

    print(f"Извлечено {len(data)} записей с professional_roles.id = '116'")
    """if data:
        print("Первая запись:")
        print(data[0])"""

    df = pd.DataFrame(data)
    """print(f"\nDataFrame shape: {df.shape}")
    print(df.head())"""

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

    # Создаем столбец с объединенным текстом
    df['combined_text'] = df.apply(combine_text, axis=1)

    # Определяем уровень для каждой вакансии
    df['level'] = df['combined_text'].apply(detect_level)

    # Анализируем распределение
    level_distribution = df['level'].value_counts()
    level_percentage = df['level'].value_counts(normalize=True) * 100

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


def forecast_cybersecurity_demand_0(save_path: str = 'forecast_cybersecurity.png'):
    db = get_db()
    coll = db['vacancies']

    """cursor = coll.find(
        {'professional_roles.id': '116'},
        {'published_at': 1}
    ).sort('published_at', 1)"""
    cursor = coll.find({
        'professional_roles.id': '116',
        'published_at': {
            '$gte': '2022-01-01T00:00:00',
            '$lte': '2025-12-31T23:59:59'
        }
    }, {'published_at': 1}).sort('published_at', 1)

    dates = []
    for doc in cursor:
        if 'published_at' in doc and doc['published_at']:
            dates.append(doc['published_at'])

    def make_data(dates):
        # Создаем DataFrame
        df = pd.DataFrame({'date': dates})
        df['date'] = pd.to_datetime(df['date'])

        # АГРЕГАЦИЯ: Группировка по месяцам без использования .dt аксессора
        # Преобразуем даты в строки формата 'YYYY-MM-01' для группировки по месяцам
        df['month_key'] = df['date'].apply(lambda x: x.strftime('%Y-%m-01'))

        # Группируем по месячному ключу и считаем количество
        monthly_counts = df.groupby('month_key').size().reset_index(name='vacancy_count')

        # Преобразуем обратно в datetime
        monthly_counts['date'] = pd.to_datetime(monthly_counts['month_key'])
        monthly_counts = monthly_counts[['date', 'vacancy_count']]

        # ОЧИСТКА: Проверка на выбросы
        if len(monthly_counts) > 0:
            Q1 = monthly_counts['vacancy_count'].quantile(0.25)
            Q3 = monthly_counts['vacancy_count'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_mask = (monthly_counts['vacancy_count'] < lower_bound) | (
                        monthly_counts['vacancy_count'] > upper_bound)
            outliers_count = outliers_mask.sum()

            print(f"Найдено выбросов: {outliers_count}")

            # Сглаживание выбросов
            if outliers_count > 0:
                # Простое сглаживание - заменяем выбросы медианным значением
                median_val = monthly_counts['vacancy_count'].median()
                monthly_counts.loc[outliers_mask, 'vacancy_count'] = median_val
                print("Выбросы сглажены")

        # СОЗДАНИЕ ВРЕМЕННОГО РЯДА
        time_series = monthly_counts.sort_values('date').reset_index(drop=True)

        """print(f"Создан временной ряд с {len(time_series)} месячными наблюдениями")
        print("Первые 5 записей:")
        print(time_series.head())"""
        return time_series

    time_series = make_data(dates)

    def exploratory_data_analysis_ts(data):
        """
        Проводит разведочный анализ временного ряда с автоматической обработкой нестационарности.

        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame с временным рядом
        date_col : str
            Название колонки с датами (по умолчанию 'date')
        value_col : str
            Название колонки со значениями (по умолчанию 'vacancy_count')
        period : int
            Период сезонности (по умолчанию 12 для месячных данных)
        max_diffs : int
            Максимальное количество дифференцирований для стабилизации ряда

        Returns:
        --------
        dict: Словарь с результатами анализа
        """

        # Создаем копию данных и устанавливаем дату как индекс
        date_col = 'date'
        value_col = 'vacancy_count'
        period = 12
        max_diffs = 3
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        ts = df[value_col]

        results = {}

        """print("=" * 60)
        print("РАЗВЕДОЧНЫЙ АНАЛИЗ ВРЕМЕННОГО РЯДА")
        print("=" * 60)

        # 1. Базовая статистика
        print("\nБАЗОВАЯ СТАТИСТИКА:")
        print("-" * 30)
        print(f"Период данных: {ts.index.min().strftime('%Y-%m-%d')} - {ts.index.max().strftime('%Y-%m-%d')}")
        print(f"Количество наблюдений: {len(ts)}")
        print(f"Среднее значение: {ts.mean():.2f}")
        print(f"Стандартное отклонение: {ts.std():.2f}")
        print(f"Минимальное значение: {ts.min()}")
        print(f"Максимальное значение: {ts.max()}")
        print(f"Медиана: {ts.median():.2f}")

        # 2. Декомпозиция ряда
        print("\nДЕКОМПОЗИЦИЯ РЯДА:")
        print("-" * 30)"""
        try:
            decomposition = seasonal_decompose(ts, model='additive', period=period, extrapolate_trend='freq')

            trend_strength = 1 - (decomposition.resid.var() / (decomposition.trend + decomposition.resid).var())
            seasonal_strength = 1 - (decomposition.resid.var() / (decomposition.seasonal + decomposition.resid).var())

            """print(f"Сила тренда: {trend_strength:.4f}")
            print(f"Сила сезонности: {seasonal_strength:.4f}")"""

            # Сохраняем компоненты декомпозиции
            results['decomposition'] = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'trend_strength': trend_strength,
                'seasonal_strength': seasonal_strength
            }

        except Exception as e:
            print(f"Ошибка при декомпозиции: {e}")
            results['decomposition'] = None

        # 3. Проверка на стационарность и преобразование при необходимости
        """print("\nАНАЛИЗ СТАЦИОНАРНОСТИ:")
        print("-" * 30)"""

        def check_stationarity(series):
            """Проверяет стационарность ряда с помощью теста Дики-Фуллера"""
            adf_test = adfuller(series.dropna())
            is_stationary = adf_test[1] <= 0.05

            """print(f"{name}:")
            print(f"  ADF Statistic: {adf_test[0]:.4f}")
            print(f"  p-value: {adf_test[1]:.4f}")
            print(f"  Стационарен: {'ДА' if is_stationary else 'НЕТ'}")"""

            if not is_stationary:
                print(
                    f"  Критические значения: 1%={adf_test[4]['1%']:.3f}, 5%={adf_test[4]['5%']:.3f}, 10%={adf_test[4]['10%']:.3f}")

            return adf_test, is_stationary

        # Проверяем исходный ряд
        adf_original, is_original_stationary = check_stationarity(ts)

        results['original_series'] = {
            'series': ts,
            'adf_test': adf_original,
            'is_stationary': is_original_stationary
        }

        # Если ряд нестационарен, применяем дифференцирование
        stationary_series = ts.copy()
        diff_order = 0
        adf_stationary = adf_original

        if not is_original_stationary:
            """print(f"\nПРЕОБРАЗОВАНИЕ РЯДА:")
            print("-" * 20)"""

            for i in range(1, max_diffs + 1):
                diff_series = ts.diff(i).dropna()
                #adf_test, is_stationary = check_stationarity(diff_series, f"Ряд после {i}-го дифференцирования")
                adf_test, is_stationary = check_stationarity(diff_series)

                if is_stationary:
                    stationary_series = diff_series
                    diff_order = i
                    adf_stationary = adf_test
                    print(f"✓ Успешно стабилизирован после {i}-го дифференцирования")
                    break
                else:
                    print(f"✗ После {i}-го дифференцирования ряд все еще нестационарен")

            if diff_order == 0:
                print("⚠ Не удалось достичь стационарности после максимального количества дифференцирований")
                # Используем ряд после первого дифференцирования как лучший вариант
                stationary_series = ts.diff(1).dropna()
                diff_order = 1
        else:
            print("✓ Ряд стационарен, преобразование не требуется")

        # 4. Анализ сезонности
        """print("\nАНАЛИЗ СЕЗОННОСТИ:")
        print("-" * 25)"""
        if hasattr(ts.index, 'month') and len(ts) >= 12:
            monthly_stats = ts.groupby(ts.index.month).agg(['mean', 'std', 'min', 'max'])
            month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                           'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']

            # Находим месяцы с максимальной и минимальной активностью
            max_month_idx = monthly_stats['mean'].idxmax()
            min_month_idx = monthly_stats['mean'].idxmin()

            """print(
                f"Месяц с максимальной активностью: {month_names[max_month_idx - 1]} (среднее: {monthly_stats.loc[max_month_idx, 'mean']:.1f})")
            print(
                f"Месяц с минимальной активностью: {month_names[min_month_idx - 1]} (среднее: {monthly_stats.loc[min_month_idx, 'mean']:.1f})")
            print(f"Размах сезонности: {monthly_stats['mean'].max() - monthly_stats['mean'].min():.1f}")"""

            results['seasonality_analysis'] = {
                'monthly_stats': monthly_stats,
                'peak_month': max_month_idx,
                'low_month': min_month_idx,
                'seasonal_amplitude': monthly_stats['mean'].max() - monthly_stats['mean'].min()
            }
        else:
            print("Недостаточно данных для анализа сезонности")
            results['seasonality_analysis'] = None

        # 5. Сохраняем финальные результаты
        results['stationary_series'] = {
            'series': stationary_series,
            'diff_order': diff_order,
            'adf_test': adf_stationary,
            'is_stationary': adf_stationary[1] <= 0.05
        }

        results['analysis_summary'] = {
            'original_observations': len(ts),
            'stationary_observations': len(stationary_series),
            'required_differencing': diff_order,
            'is_final_series_stationary': adf_stationary[1] <= 0.05,
            'final_p_value': adf_stationary[1]
        }

        # 6. Финальный отчет
        """print("\n" + "=" * 60)
        print("ИТОГОВЫЙ ОТЧЕТ")
        print("=" * 60)
        print(f"Исходный ряд: {len(ts)} наблюдений")
        print(f"Стационарный ряд: {len(stationary_series)} наблюдений")
        print(f"Применено дифференцирований: {diff_order}")
        print(f"Финальный p-value: {adf_stationary[1]:.6f}")
        print(f"Ряд готов для прогнозирования: {'ДА' if adf_stationary[1] <= 0.05 else 'НЕТ'}")

        if results.get('decomposition'):
            print(f"Сила тренда: {results['decomposition']['trend_strength']:.4f}")
            print(f"Сила сезонности: {results['decomposition']['seasonal_strength']:.4f}")

        if results.get('seasonality_analysis'):
            month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                           'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
            peak_month = results['seasonality_analysis']['peak_month']
            low_month = results['seasonality_analysis']['low_month']
            print(f"Пиковый месяц: {month_names[peak_month - 1]}")
            print(f"Месяц спада: {month_names[low_month - 1]}")"""

        return results

    results = exploratory_data_analysis_ts(time_series)

    def calculate_metrics(actual, predicted):
        """Расчет метрик качества без использования sklearn"""
        actual = np.array(actual)
        predicted = np.array(predicted)

        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(actual - predicted))

        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        # MAPE (Mean Absolute Percentage Error) - избегаем деления на 0
        mask = actual != 0
        if np.any(mask):
            mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        else:
            mape = np.nan

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

    def analyze_forecast_seasonality(forecast_series):
        """Анализ сезонности в прогнозируемых данных"""

        # Группируем по месяцам
        monthly_forecast = forecast_series.groupby(forecast_series.index.month).mean()

        # Находим пиковые и минимальные месяцы
        peak_month = monthly_forecast.idxmax()
        low_month = monthly_forecast.idxmin()

        month_names = {
            1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
            5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
            9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
        }

        return {
            'monthly_means': monthly_forecast,
            'peak_month': peak_month,
            'peak_month_name': month_names.get(peak_month, 'Неизвестно'),
            'low_month': low_month,
            'low_month_name': month_names.get(low_month, 'Неизвестно'),
            'amplitude': monthly_forecast.max() - monthly_forecast.min(),
            'total_annual_demand': forecast_series.sum()
        }

    def build_ets_forecast_0(step_2_output, forecast_months=12, plot_results=True):
        """
        Построение модели ETS для прогнозирования спроса на специалистов по ИБ

        Parameters:
        -----------
        step_2_output : dict
            Словарь с результатами из step_2_output.txt
        forecast_months : int
            Количество месяцев для прогноза (по умолчанию 12)
        plot_results : bool
            Визуализировать ли результаты (по умолчанию True)

        Returns:
        --------
        dict
            Словарь с результатами прогнозирования
        """

        # Извлекаем исходные данные
        original_series = step_2_output['original_series']['series']

        # Преобразуем в pandas Series с правильным индексом datetime
        if not isinstance(original_series.index, pd.DatetimeIndex):
            dates = pd.to_datetime(original_series.index)
            series_data = pd.Series(original_series.values, index=dates, name='vacancy_count')
        else:
            series_data = original_series.copy()

        # Сортируем по дате на всякий случай
        series_data = series_data.sort_index()

        print(f"Исходные данные: {len(series_data)} наблюдений")
        print(f"Период данных: {series_data.index.min()} - {series_data.index.max()}")
        print(f"Среднее количество вакансий: {series_data.mean():.2f}")

        # Разделяем на обучающую и тестовую выборки (последние 12 месяцев для валидации)
        if len(series_data) > 24:
            train_data = series_data[:-12]
            test_data = series_data[-12:]
        else:
            train_data = series_data
            test_data = None

        print(f"\nОбучающая выборка: {len(train_data)} наблюдений")
        if test_data is not None:
            print(f"Тестовая выборка: {len(test_data)} наблюдений.", test_data)

        try:
            # Построение ETS модели
            print("\nПостроение ETS модели...")

            # Пробуем разные конфигурации модели
            best_model = None
            best_aic = np.inf
            best_config = None

            # Тестируем разные конфигурации
            configs = [
                {'error': 'add', 'trend': 'add', 'seasonal': 'add', 'damped_trend': True},
                {'error': 'add', 'trend': 'add', 'seasonal': 'add', 'damped_trend': False},
                {'error': 'add', 'trend': None, 'seasonal': 'add', 'damped_trend': False},
                {'error': 'mul', 'trend': 'add', 'seasonal': 'mul', 'damped_trend': True},
            ]

            for config in configs:
                #print(config)
                try:
                    model = ETSModel(
                        train_data,
                        error=config['error'],
                        trend=config['trend'],
                        seasonal=config['seasonal'],
                        seasonal_periods=12,
                        damped_trend=config.get('damped_trend', False)
                    )

                    fitted_model = model.fit(disp=False, maxiter=1000)

                    #print(fitted_model)

                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_model = fitted_model
                        best_config = config

                except Exception as e:
                    print(f"Конфигурация {config} не сработала: {e}")
                    continue

            if best_model is None:
                # Если ни одна конфигурация не сработала, пробуем простую модель
                print("Используем простую модель...")
                model = ETSModel(
                    train_data,
                    error='add',
                    trend='add',
                    seasonal=None,
                    damped_trend=True
                )
                best_model = model.fit(disp=False)
                best_config = {'error': 'add', 'trend': 'add', 'seasonal': None, 'damped_trend': True}

            print(f"Лучшая конфигурация: {best_config}")
            print(f"AIC лучшей модели: {best_aic:.2f}")

            # Прогноз
            forecast_values = best_model.forecast(steps=forecast_months)
            print(forecast_values)

            # Для доверительных интервалов используем симуляцию
            n_simulations = 1000
            simulations = best_model.simulate(
                nsimulations=forecast_months,
                repetitions=n_simulations,
                anchor='end'
            )

            # Рассчитываем доверительные интервалы
            lower_bound = np.percentile(simulations, 2.5, axis=1)
            upper_bound = np.percentile(simulations, 97.5, axis=1)

            # Создаем индекс для прогноза
            last_date = train_data.index[-1]
            if pd.infer_freq(train_data.index) == 'MS':  # Monthly Start
                forecast_index = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=forecast_months,
                    freq='MS'
                )
            else:
                forecast_index = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=forecast_months,
                    freq='M'
                )

            forecast_series = pd.Series(forecast_values, index=forecast_index)
            confidence_df = pd.DataFrame({
                'lower': lower_bound,
                'upper': upper_bound
            }, index=forecast_index)

            # Оценка модели на тестовых данных (если есть)
            test_metrics = {}
            test_pred = None

            if test_data is not None and len(test_data) > 0:
                # Прогноз на длину тестовой выборки
                test_pred_values = best_model.forecast(steps=len(test_data))
                test_pred = pd.Series(test_pred_values, index=test_data.index)
                print("test_pred_values:", "\n", test_pred_values)
                print("--------")
                print("test_pred:", "\n", test_pred)
                print("--------")

                # Метрики качества без sklearn
                test_metrics = calculate_metrics(test_data.values, test_pred.values)
                print(test_metrics)

                print(f"\nМетрики качества на тестовых данных:")
                print(f"MAE: {test_metrics['MAE']:.2f}")
                print(f"RMSE: {test_metrics['RMSE']:.2f}")
                print(f"MAPE: {test_metrics['MAPE']:.2f}%")

            # Визуализация результатов
            if plot_results:
                plt.figure(figsize=(14, 10))

                # Основной график
                plt.subplot(2, 1, 1)
                plt.plot(train_data.index, train_data.values, label='Исторические данные', color='blue', linewidth=2)

                if test_data is not None and test_pred is not None:
                    plt.plot(test_data.index, test_data.values, label='Тестовые данные', color='green', linewidth=2)
                    plt.plot(test_data.index, test_pred.values, label='Прогноз на тест', color='orange', linestyle='--',
                             linewidth=2)

                plt.plot(forecast_series.index, forecast_series.values, label='Прогноз', color='red', linewidth=2)
                plt.fill_between(
                    forecast_series.index,
                    confidence_df['lower'],
                    confidence_df['upper'],
                    color='red', alpha=0.2, label='95% доверительный интервал'
                )

                plt.title('Прогноз спроса на специалистов по ИБ (ETS модель)', fontsize=14, fontweight='bold')
                plt.xlabel('Дата')
                plt.ylabel('Количество вакансий')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # График компонентов
                plt.subplot(2, 1, 2)
                if hasattr(best_model, 'fittedvalues'):
                    fitted_values = best_model.fittedvalues
                    if len(fitted_values) == len(train_data):
                        plt.plot(train_data.index, fitted_values,
                                 label='Подгонка модели', color='orange', linewidth=2)
                    plt.plot(train_data.index, train_data.values, label='Фактические данные',
                             color='blue', alpha=0.7, linewidth=1)

                    plt.title('Подгонка модели к историческим данным')
                    plt.xlabel('Дата')
                    plt.ylabel('Количество вакансий')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                plt.tight_layout()
                #plt.show()
                plt.savefig(save_path)
                print('Сохранён график:', save_path)
                plt.close()

            # Анализ сезонности в прогнозе
            seasonal_analysis = analyze_forecast_seasonality(forecast_series)

            # Формируем результаты
            results = {
                'model': best_model,
                'forecast': forecast_series,
                'confidence_intervals': confidence_df,
                'model_summary': {
                    'model_type': f"ETS({best_config['error'][0].upper()},{best_config['trend'][0].upper() if best_config['trend'] else 'N'},{best_config['seasonal'][0].upper() if best_config['seasonal'] else 'N'})",
                    'aic': best_model.aic,
                    'bic': best_model.bic,
                    'config': best_config,
                },
                'test_metrics': test_metrics,
                'seasonal_analysis': seasonal_analysis
                #'recommendations': generate_recommendations(forecast_series, seasonal_analysis)
            }

            # Вывод ключевых insights
            print("\n" + "=" * 60)
            print("КЛЮЧЕВЫЕ ВЫВОДЫ ДЛЯ СПЕЦИАЛИСТОВ ПО ИБ")
            print("=" * 60)
            print(f"Средний прогнозируемый спрос: {forecast_series.mean():.1f} вакансий/мес")
            print(f"Пиковый месяц: {seasonal_analysis['peak_month_name']} ({forecast_series.max():.1f} вакансий)")
            print(f"Самый низкий спрос: {seasonal_analysis['low_month_name']} ({forecast_series.min():.1f} вакансий)")
            print(f"Сезонная амплитуда: {seasonal_analysis['amplitude']:.1f} вакансий")


            return results

        except Exception as e:
            print(f"Ошибка при построении модели: {e}")
            import traceback
            traceback.print_exc()
            return None

    def build_ets_forecast(step_2_output, forecast_months=12, plot_results=True):
        """
        Построение модели ETS для прогнозирования спроса на специалистов по ИБ
        """
        # Извлекаем исходные данные
        original_series = step_2_output['original_series']['series']

        # Преобразуем в pandas Series с правильным индексом datetime
        if not isinstance(original_series.index, pd.DatetimeIndex):
            dates = pd.to_datetime(original_series.index)
            series_data = pd.Series(original_series.values, index=dates, name='vacancy_count')
        else:
            series_data = original_series.copy()

        # Сортируем по дате на всякий случай
        series_data = series_data.sort_index()

        print(f"Исходные данные: {len(series_data)} наблюдений")
        print(f"Период данных: {series_data.index.min()} - {series_data.index.max()}")
        print(f"Среднее количество вакансий: {series_data.mean():.2f}")

        # Разделяем на обучающую и тестовую выборки (последние 12 месяцев для валидации)
        if len(series_data) > 24:
            train_data = series_data[:-12]
            test_data = series_data[-12:]
        else:
            train_data = series_data
            test_data = None

        print(f"\nОбучающая выборка: {len(train_data)} наблюдений")
        if test_data is not None:
            print(f"Тестовая выборка: {len(test_data)} наблюдений")

        try:
            # Построение ETS модели
            print("\nПостроение ETS модели...")

            # Пробуем разные конфигурации модели
            best_model = None
            best_aic = np.inf
            best_config = None

            # Тестируем разные конфигурации
            configs = [
                {'error': 'add', 'trend': 'add', 'seasonal': 'add', 'damped_trend': True},
                {'error': 'add', 'trend': 'add', 'seasonal': 'add', 'damped_trend': False},
                {'error': 'add', 'trend': None, 'seasonal': 'add', 'damped_trend': False},
                {'error': 'mul', 'trend': 'add', 'seasonal': 'mul', 'damped_trend': True},
            ]

            for config in configs:
                try:
                    model = ETSModel(
                        train_data,
                        error=config['error'],
                        trend=config['trend'],
                        seasonal=config['seasonal'],
                        seasonal_periods=12,
                        damped_trend=config.get('damped_trend', False)
                    )

                    fitted_model = model.fit(disp=False, maxiter=1000)

                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_model = fitted_model
                        best_config = config

                except Exception as e:
                    print(f"Конфигурация {config} не сработала: {e}")
                    continue

            if best_model is None:
                # Если ни одна конфигурация не сработала, пробуем простую модель
                print("Используем простую модель...")
                model = ETSModel(
                    train_data,
                    error='add',
                    trend='add',
                    seasonal=None,
                    damped_trend=True
                )
                best_model = model.fit(disp=False)
                best_config = {'error': 'add', 'trend': 'add', 'seasonal': None, 'damped_trend': True}

            print(f"Лучшая конфигурация: {best_config}")
            print(f"AIC лучшей модели: {best_aic:.2f}")

            # Прогноз
            forecast_values = best_model.forecast(steps=forecast_months)
            forecast_values_data = forecast_values.values

            # Для доверительных интервалов используем симуляцию
            n_simulations = 1000
            simulations = best_model.simulate(
                nsimulations=forecast_months,
                repetitions=n_simulations,
                anchor='end'
            )

            # Рассчитываем доверительные интервалы
            lower_bound = np.percentile(simulations, 2.5, axis=1)
            upper_bound = np.percentile(simulations, 97.5, axis=1)

            # Создаем индекс для прогноза
            last_date = train_data.index[-1]
            if pd.infer_freq(train_data.index) == 'MS':  # Monthly Start
                forecast_index = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=forecast_months,
                    freq='MS'
                )
            else:
                forecast_index = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=forecast_months,
                    freq='M'
                )
            print("forecast_values:\n", forecast_values)

            forecast_series = pd.Series(forecast_values_data, index=forecast_index)
            print("forecast_series:\n", forecast_series)
            confidence_df = pd.DataFrame({
                'lower': lower_bound,
                'upper': upper_bound
            }, index=forecast_index)

            # Оценка модели на тестовых данных (если есть)
            test_metrics = {}
            test_pred = None

            if test_data is not None and len(test_data) > 0:
                # Прогноз на длину тестовой выборки
                test_pred_values = best_model.forecast(steps=len(test_data))

                # ИСПРАВЛЕНИЕ: Извлекаем значения, а не весь Series
                test_pred_values_data = test_pred_values.values  # Получаем массив значений


                # ИСПРАВЛЕНИЕ: Создаем правильный индекс для тестового прогноза
                # Используем те же даты, что и в test_data
                test_pred = pd.Series(test_pred_values_data, index=test_data.index)

                """print("test_pred_values:", "\n", test_pred_values)
                print("--------")
                print("test_pred:", "\n", test_pred)
                print("--------")"""

                # Метрики качества без sklearn
                test_metrics = calculate_metrics(test_data.values, test_pred.values)
                """print(test_metrics)

                print(f"\nМетрики качества на тестовых данных:")
                print(f"MAE: {test_metrics['MAE']:.2f}")
                print(f"RMSE: {test_metrics['RMSE']:.2f}")
                print(f"MAPE: {test_metrics['MAPE']:.2f}%")"""

            if plot_results:
                plt.figure(figsize=(14, 10))

                # Основной график
                plt.subplot(2, 1, 1)
                plt.plot(train_data.index, train_data.values, label='Исторические данные', color='blue', linewidth=2)

                if test_data is not None and test_pred is not None:
                    plt.plot(test_data.index, test_data.values, label='Тестовые данные', color='green', linewidth=2)
                    plt.plot(test_data.index, test_pred.values, label='Прогноз на тест', color='orange', linestyle='--',
                             linewidth=2)

                plt.plot(forecast_series.index, forecast_series.values, label='Прогноз', color='red', linewidth=2)
                plt.fill_between(
                    forecast_series.index,
                    confidence_df['lower'],
                    confidence_df['upper'],
                    color='red', alpha=0.2, label='95% доверительный интервал'
                )

                plt.title('Прогноз спроса на специалистов по ИБ (ETS модель)', fontsize=14, fontweight='bold')
                plt.xlabel('Дата')
                plt.ylabel('Количество вакансий')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # График компонентов
                plt.subplot(2, 1, 2)
                if hasattr(best_model, 'fittedvalues'):
                    fitted_values = best_model.fittedvalues
                    if len(fitted_values) == len(train_data):
                        plt.plot(train_data.index, fitted_values,
                                 label='Подгонка модели', color='orange', linewidth=2)
                    plt.plot(train_data.index, train_data.values, label='Фактические данные',
                             color='blue', alpha=0.7, linewidth=1)

                    plt.title('Подгонка модели к историческим данным')
                    plt.xlabel('Дата')
                    plt.ylabel('Количество вакансий')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                plt.tight_layout()
                #plt.show()
                plt.savefig(save_path)
                print('Сохранён график:', save_path)
                plt.close()

            # Анализ сезонности в прогнозе
            seasonal_analysis = analyze_forecast_seasonality(forecast_series)

            # Формируем результаты
            results = {
                'model': best_model,
                'forecast': forecast_series,
                'confidence_intervals': confidence_df,
                'model_summary': {
                    'model_type': f"ETS({best_config['error'][0].upper()},{best_config['trend'][0].upper() if best_config['trend'] else 'N'},{best_config['seasonal'][0].upper() if best_config['seasonal'] else 'N'})",
                    'aic': best_model.aic,
                    'bic': best_model.bic,
                    'config': best_config,
                },
                'test_metrics': test_metrics,
                'seasonal_analysis': seasonal_analysis
                #'recommendations': generate_recommendations(forecast_series, seasonal_analysis)
            }

            # Вывод ключевых insights
            print("\n" + "=" * 60)
            print("КЛЮЧЕВЫЕ ВЫВОДЫ ДЛЯ СПЕЦИАЛИСТОВ ПО ИБ")
            print("=" * 60)
            print(f"Средний прогнозируемый спрос: {forecast_series.mean():.1f} вакансий/мес")
            print(f"Пиковый месяц: {seasonal_analysis['peak_month_name']} ({forecast_series.max():.1f} вакансий)")
            print(f"Самый низкий спрос: {seasonal_analysis['low_month_name']} ({forecast_series.min():.1f} вакансий)")
            print(f"Сезонная амплитуда: {seasonal_analysis['amplitude']:.1f} вакансий")

            return results

        except Exception as e:
            print(f"Ошибка при построении модели: {e}")
            import traceback
            traceback.print_exc()
            return None

    build_ets_forecast(results)


def forecast_cybersecurity_demand(save_path: str = 'forecast_cybersecurity.png'):
    db = get_db()
    coll = db['vacancies']

    cursor = coll.find({
        'professional_roles.id': '116',
        'published_at': {
            '$gte': '2022-01-01T00:00:00',
            '$lte': '2025-07-31T23:59:59'  # Данные до конца июля 2025
        }
    }, {'published_at': 1}).sort('published_at', 1)

    dates = []
    for doc in cursor:
        if 'published_at' in doc and doc['published_at']:
            dates.append(doc['published_at'])

    def make_data(dates):
        # Создаем DataFrame
        df = pd.DataFrame({'date': dates})
        df['date'] = pd.to_datetime(df['date'])

        # АГРЕГАЦИЯ: Группировка по месяцам без использования .dt аксессора
        # Преобразуем даты в строки формата 'YYYY-MM-01' для группировки по месяцам
        df['month_key'] = df['date'].apply(lambda x: x.strftime('%Y-%m-01'))

        # Группируем по месячному ключу и считаем количество
        monthly_counts = df.groupby('month_key').size().reset_index(name='vacancy_count')

        # Преобразуем обратно в datetime
        monthly_counts['date'] = pd.to_datetime(monthly_counts['month_key'])
        monthly_counts = monthly_counts[['date', 'vacancy_count']]

        # ОЧИСТКА: Проверка на выбросы
        if len(monthly_counts) > 0:
            Q1 = monthly_counts['vacancy_count'].quantile(0.25)
            Q3 = monthly_counts['vacancy_count'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_mask = (monthly_counts['vacancy_count'] < lower_bound) | (
                    monthly_counts['vacancy_count'] > upper_bound)
            outliers_count = outliers_mask.sum()

            print(f"Найдено выбросов: {outliers_count}")

            # Сглаживание выбросов
            if outliers_count > 0:
                # Простое сглаживание - заменяем выбросы медианным значением
                median_val = monthly_counts['vacancy_count'].median()
                monthly_counts.loc[outliers_mask, 'vacancy_count'] = median_val
                print("Выбросы сглажены")

        # СОЗДАНИЕ ВРЕМЕННОГО РЯДА
        time_series = monthly_counts.sort_values('date').reset_index(drop=True)

        """print(f"Создан временной ряд с {len(time_series)} месячными наблюдениями")
        print("Первые 5 записей:")
        print(time_series.head())"""
        return time_series

    time_series = make_data(dates)

    def exploratory_data_analysis_ts(data):
        """
        Проводит разведочный анализ временного ряда с автоматической обработкой нестационарности.
        """

        # Создаем копию данных и устанавливаем дату как индекс
        date_col = 'date'
        value_col = 'vacancy_count'
        period = 12
        max_diffs = 3
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        # ИСПРАВЛЕНИЕ: Устанавливаем частоту временного ряда
        ts = df[value_col].asfreq('MS')  # Monthly Start frequency

        # Заполняем пропущенные месяцы нулями
        ts = ts.fillna(0)

        results = {}

        try:
            decomposition = seasonal_decompose(ts, model='additive', period=period, extrapolate_trend='freq')

            trend_strength = 1 - (decomposition.resid.var() / (decomposition.trend + decomposition.resid).var())
            seasonal_strength = 1 - (decomposition.resid.var() / (decomposition.seasonal + decomposition.resid).var())

            # Сохраняем компоненты декомпозиции
            results['decomposition'] = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'trend_strength': trend_strength,
                'seasonal_strength': seasonal_strength
            }

        except Exception as e:
            print(f"Ошибка при декомпозиции: {e}")
            results['decomposition'] = None

        def check_stationarity(series):
            """Проверяет стационарность ряда с помощью теста Дики-Фуллера"""
            adf_test = adfuller(series.dropna())
            is_stationary = adf_test[1] <= 0.05

            if not is_stationary:
                print(
                    f"  Критические значения: 1%={adf_test[4]['1%']:.3f}, 5%={adf_test[4]['5%']:.3f}, 10%={adf_test[4]['10%']:.3f}")

            return adf_test, is_stationary

        # Проверяем исходный ряд
        adf_original, is_original_stationary = check_stationarity(ts)

        results['original_series'] = {
            'series': ts,
            'adf_test': adf_original,
            'is_stationary': is_original_stationary
        }

        # Если ряд нестационарен, применяем дифференцирование
        stationary_series = ts.copy()
        diff_order = 0
        adf_stationary = adf_original

        if not is_original_stationary:
            for i in range(1, max_diffs + 1):
                diff_series = ts.diff(i).dropna()
                adf_test, is_stationary = check_stationarity(diff_series)

                if is_stationary:
                    stationary_series = diff_series
                    diff_order = i
                    adf_stationary = adf_test
                    print(f"✓ Успешно стабилизирован после {i}-го дифференцирования")
                    break
                else:
                    print(f"✗ После {i}-го дифференцирования ряд все еще нестационарен")

            if diff_order == 0:
                print("⚠ Не удалось достичь стационарности после максимального количества дифференцирований")
                # Используем ряд после первого дифференцирования как лучший вариант
                stationary_series = ts.diff(1).dropna()
                diff_order = 1
        else:
            print("✓ Ряд стационарен, преобразование не требуется")

        # Анализ сезонности
        if hasattr(ts.index, 'month') and len(ts) >= 12:
            monthly_stats = ts.groupby(ts.index.month).agg(['mean', 'std', 'min', 'max'])
            month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                           'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']

            # Находим месяцы с максимальной и минимальной активностью
            max_month_idx = monthly_stats['mean'].idxmax()
            min_month_idx = monthly_stats['mean'].idxmin()

            results['seasonality_analysis'] = {
                'monthly_stats': monthly_stats,
                'peak_month': max_month_idx,
                'low_month': min_month_idx,
                'seasonal_amplitude': monthly_stats['mean'].max() - monthly_stats['mean'].min()
            }
        else:
            print("Недостаточно данных для анализа сезонности")
            results['seasonality_analysis'] = None

        # Сохраняем финальные результаты
        results['stationary_series'] = {
            'series': stationary_series,
            'diff_order': diff_order,
            'adf_test': adf_stationary,
            'is_stationary': adf_stationary[1] <= 0.05
        }

        results['analysis_summary'] = {
            'original_observations': len(ts),
            'stationary_observations': len(stationary_series),
            'required_differencing': diff_order,
            'is_final_series_stationary': adf_stationary[1] <= 0.05,
            'final_p_value': adf_stationary[1]
        }

        return results

    results = exploratory_data_analysis_ts(time_series)

    def calculate_metrics(actual, predicted):
        """Расчет метрик качества без использования sklearn"""
        actual = np.array(actual)
        predicted = np.array(predicted)

        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(actual - predicted))

        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        # MAPE (Mean Absolute Percentage Error) - избегаем деления на 0
        mask = actual != 0
        if np.any(mask):
            mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        else:
            mape = np.nan

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

    def analyze_forecast_seasonality(forecast_series):
        """Анализ сезонности в прогнозируемых данных"""

        # Группируем по месяцам
        monthly_forecast = forecast_series.groupby(forecast_series.index.month).mean()

        # Находим пиковые и минимальные месяцы
        peak_month = monthly_forecast.idxmax()
        low_month = monthly_forecast.idxmin()

        month_names = {
            1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
            5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
            9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
        }

        return {
            'monthly_means': monthly_forecast,
            'peak_month': peak_month,
            'peak_month_name': month_names.get(peak_month, 'Неизвестно'),
            'low_month': low_month,
            'low_month_name': month_names.get(low_month, 'Неизвестно'),
            'amplitude': monthly_forecast.max() - monthly_forecast.min(),
            'total_annual_demand': forecast_series.sum()
        }

    def build_ets_forecast(step_2_output, forecast_months=12, plot_results=True):
        """
        Построение модели ETS для прогнозирования спроса на специалистов по ИБ
        """
        # Извлекаем исходные данные
        original_series = step_2_output['original_series']['series']

        # Преобразуем в pandas Series с правильным индексом datetime
        if not isinstance(original_series.index, pd.DatetimeIndex):
            dates = pd.to_datetime(original_series.index)
            series_data = pd.Series(original_series.values, index=dates, name='vacancy_count')
        else:
            series_data = original_series.copy()

        # ИСПРАВЛЕНИЕ: Устанавливаем частоту и сортируем
        series_data = series_data.asfreq('MS').sort_index()

        # Заполняем пропущенные значения
        series_data = series_data.fillna(0)

        print(f"Исходные данные: {len(series_data)} наблюдений")
        print(f"Период данных: {series_data.index.min()} - {series_data.index.max()}")
        print(f"Среднее количество вакансий: {series_data.mean():.2f}")

        # Используем ВСЕ данные для обучения финальной модели
        train_data = series_data

        print(f"\nОбучающая выборка для прогноза: {len(train_data)} наблюдений")

        try:
            # Построение ETS модели
            print("\nПостроение ETS модели...")

            # Для небольшого количества данных используем простую модель без сезонности
            print("Используем простую модель ETS (без сезонности) из-за малого количества данных...")
            try:
                model = ETSModel(
                    train_data,
                    error='add',
                    trend='add',
                    seasonal=None,
                    damped_trend=True
                )
                best_model = model.fit(disp=False)
                best_config = {'error': 'add', 'trend': 'add', 'seasonal': None, 'damped_trend': True}
            except Exception as e:
                print(f"Ошибка при построении модели: {e}")
                # Пробуем еще более простую модель
                model = ETSModel(
                    train_data,
                    error='add',
                    trend=None,
                    seasonal=None
                )
                best_model = model.fit(disp=False)
                best_config = {'error': 'add', 'trend': None, 'seasonal': None}

            print(f"Конфигурация модели: {best_config}")
            print(f"AIC модели: {best_model.aic:.2f}")

            # Прогноз
            forecast_values = best_model.forecast(steps=forecast_months)
            forecast_values_data = forecast_values.values

            # Для доверительных интервалов используем симуляцию
            n_simulations = 1000
            try:
                simulations = best_model.simulate(
                    nsimulations=forecast_months,
                    repetitions=n_simulations,
                    anchor='end'
                )
                # Рассчитываем доверительные интервалы
                lower_bound = np.percentile(simulations, 2.5, axis=1)
                upper_bound = np.percentile(simulations, 97.5, axis=1)
            except:
                # Если симуляция не работает, используем приближенные доверительные интервалы
                print("Используем приближенные доверительные интервалы")
                std_error = np.std(forecast_values_data) * 0.5
                lower_bound = forecast_values_data - 1.96 * std_error
                upper_bound = forecast_values_data + 1.96 * std_error

            # ИСПРАВЛЕНИЕ: Создаем правильный индекс для прогноза
            last_date = train_data.index[-1]

            # Создаем индекс прогноза с правильной частотой
            forecast_index = pd.date_range(
                start=last_date + pd.offsets.MonthBegin(1),  # Следующий месяц
                periods=forecast_months,
                freq='MS'  # Monthly Start
            )

            print(f"Последняя дата в данных: {last_date.strftime('%Y-%m-%d')}")
            print(
                f"Период прогноза: {forecast_index[0].strftime('%Y-%m-%d')} - {forecast_index[-1].strftime('%Y-%m-%d')}")

            forecast_series = pd.Series(forecast_values_data, index=forecast_index)
            confidence_df = pd.DataFrame({
                'lower': lower_bound,
                'upper': upper_bound
            }, index=forecast_index)

            # Визуализация результатов
            if plot_results:
                plt.figure(figsize=(14, 8))

                # Основной график
                plt.plot(train_data.index, train_data.values, label='Исторические данные', color='blue', linewidth=2)
                plt.plot(forecast_series.index, forecast_series.values, label='Прогноз', color='red', linewidth=2)
                plt.fill_between(
                    forecast_series.index,
                    confidence_df['lower'],
                    confidence_df['upper'],
                    color='red', alpha=0.2, label='95% доверительный интервал'
                )

                plt.title('Прогноз спроса на специалистов по ИБ (ETS модель)', fontsize=14, fontweight='bold')
                plt.xlabel('Дата')
                plt.ylabel('Количество вакансий')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()

                plt.savefig(save_path)
                print('Сохранён график:', save_path)
                plt.close()

            # Анализ сезонности в прогнозе
            seasonal_analysis = analyze_forecast_seasonality(forecast_series)

            # Формируем результаты
            results = {
                'model': best_model,
                'forecast': forecast_series,
                'confidence_intervals': confidence_df,
                'model_summary': {
                    'model_type': f"ETS({best_config['error'][0].upper()},{best_config['trend'][0].upper() if best_config['trend'] else 'N'},{best_config['seasonal'][0].upper() if best_config['seasonal'] else 'N'})",
                    'aic': best_model.aic,
                    'bic': best_model.bic,
                    'config': best_config,
                },
                'test_metrics': {},
                'seasonal_analysis': seasonal_analysis
            }

            # Вывод ключевых insights
            print("\n" + "=" * 60)
            print("КЛЮЧЕВЫЕ ВЫВОДЫ ДЛЯ СПЕЦИАЛИСТОВ ПО ИБ")
            print("=" * 60)
            print(f"Средний прогнозируемый спрос: {forecast_series.mean():.1f} вакансий/мес")
            print(f"Пиковый месяц: {seasonal_analysis['peak_month_name']} ({forecast_series.max():.1f} вакансий)")
            print(f"Самый низкий спрос: {seasonal_analysis['low_month_name']} ({forecast_series.min():.1f} вакансий)")
            print(f"Сезонная амплитуда: {seasonal_analysis['amplitude']:.1f} вакансий")
            print(f"Общий годовой спрос: {seasonal_analysis['total_annual_demand']:.0f} вакансий")

            return results

        except Exception as e:
            print(f"Ошибка при построении модели: {e}")
            import traceback
            traceback.print_exc()
            return None

    build_ets_forecast(results)





def show_fields():
    # Извлечение данных из MongoDB
    db = get_db()
    coll = db['vacancies']

    # data = list(coll.find({}, {'vacancy_name': 1, 'description': 1, 'experience': 1}))

    # Проверим структуру одной записи
    sample_doc = coll.find_one()
    print("Пример документа:")
    print(sample_doc)

    # Посмотрим на все поля в коллекции
    all_fields = set()
    for doc in coll.find().limit(5):
        all_fields.update(doc.keys())
    print("\nВсе поля в коллекции:")
    print(all_fields)

def analyze_all():
    # top_cities_plot()
    experience_levels_plot()
    #show_fields()
    #ib_vacancies_filter()
    #forecast_cybersecurity_demand()



if __name__ == '__main__':
    analyze_all()
