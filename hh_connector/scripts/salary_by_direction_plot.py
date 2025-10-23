from db.mongo_connection import get_db
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

def _extract_directions(vacancy: dict) -> List[str]:
    """
    Попробовать достать список направлений/специализаций:
    priority: specializations -> professional_roles -> employer.industry
    specializations/professional_roles могут быть списками словарей с 'name' или списками строк.
    """
    # 1) specializations
    specs = vacancy.get('specializations')
    if specs:
        # может быть список словарей с 'name' или список строк
        names = []
        for s in specs:
            if isinstance(s, dict) and 'name' in s:
                names.append(s['name'])
            elif isinstance(s, str):
                names.append(s)
        if names:
            return names

    # 2) professional_roles
    roles = vacancy.get('professional_roles')
    if roles:
        names = []
        for r in roles:
            if isinstance(r, dict) and 'name' in r:
                names.append(r['name'])
            elif isinstance(r, str):
                names.append(r)
        if names:
            return names

    # 3) employer.industry
    industry = vacancy.get('employer', {}).get('industry')
    if industry:
        if isinstance(industry, list):
            # list of strings
            return [i for i in industry if isinstance(i, str)]
        elif isinstance(industry, str):
            return [industry]

    # fallback — использовать vacancy['name'] (например, название вакансии)
    title = vacancy.get('name')
    return [title] if isinstance(title, str) else []

def _salary_to_number(s: dict) -> Optional[float]:
    """
    Преобразовать поле salary в одно число:
    - если есть both 'from' и 'to' -> midpoint
    - если есть только один -> использовать его
    - если нет -> None
    """
    if not s:
        return None
    sf = s.get('from')
    st = s.get('to')
    # иногда значения приходят как строки — попытаться привести к float
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None
    f = _to_float(sf)
    t = _to_float(st)
    if f is not None and t is not None:
        return (f + t) / 2.0
    if f is not None:
        return f
    if t is not None:
        return t
    return None

def salary_by_direction_plot(
    top_n: int = 10,
    min_vacancies: int = 5,
    currency_allowed: List[str] = ['RUR', 'RUB'],
    save_path: str = 'salary_by_direction.png'
):
    """
    Построить график сравнения средней и медианной зарплаты по направлениям,
    выделив направление 'Информационная безопасность' (ИБ).
    """
    db = get_db()
    coll = db['vacancies']

    # Загружаем все документы (можно ограничить проекцией чтобы снизить память)
    cursor = coll.find({}, {
        'specializations': 1,
        'professional_roles': 1,
        'employer.industry': 1,
        'name': 1,
        'salary': 1
    })

    rows = []
    for doc in cursor:
        salary = doc.get('salary')
        sal_value = _salary_to_number(salary)
        currency = salary.get('currency') if isinstance(salary, dict) else None
        if sal_value is None:
            continue
        if currency is None:
            continue
        # нормализуем код валюты
        cur_norm = str(currency).upper()
        if cur_norm not in currency_allowed:
            continue

        directions = _extract_directions(doc)
        for d in directions:
            if not d:
                continue
            rows.append({
                'direction': d.strip(),
                'salary': sal_value
            })

    if not rows:
        print('Нет подходящих записей зарплат после фильтрации по валюте/наличию зарплаты.')
        return

    df = pd.DataFrame(rows)

    # Группировка: считаем count, mean, median
    agg = df.groupby('direction').salary.agg(['count', 'mean', 'median']).reset_index()
    # Оставляем направления с минимумом вакансий
    agg = agg[agg['count'] >= min_vacancies]

    if agg.empty:
        print('Нет направлений с достаточным количеством вакансий (min_vacancies={}).'.format(min_vacancies))
        return

    # Сортируем по count и берём top_n (но убедимся что ИБ присутствует)
    agg = agg.sort_values('count', ascending=False)
    top = agg.head(top_n).copy()

    # Попробуем найти направление ИБ (информационная безопасность)
    def is_infosec(name: str) -> bool:
        if not isinstance(name, str):
            return False
        n = name.lower()
        return ('безопас' in n) or ('информац' in n) or ('иб' == n.strip() or 'иб ' in n or ' иб' in n)

    infosec_row = agg[agg['direction'].apply(is_infosec)]
    if not infosec_row.empty:
        # если ИБ не в top, добавить её
        infosec_name = infosec_row.iloc[0]['direction']
        if infosec_name not in top['direction'].values:
            top = pd.concat([top, infosec_row.head(1)], ignore_index=True)

    # Для наглядности отсортируем по mean или по count — здесь по mean убыв.
    top = top.sort_values('mean', ascending=False).reset_index(drop=True)

    # Построение графика: две колонки (mean, median)
    x = np.arange(len(top))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, max(4, len(top)*0.6)))
    bars_mean = ax.bar(x - width/2, top['mean'], width, label='Средняя (mean)')
    bars_med = ax.bar(x + width/2, top['median'], width, label='Медиана (median)')

    # Подписи и оформление
    ax.set_xticks(x)
    ax.set_xticklabels(top['direction'], rotation=45, ha='right')
    ax.set_ylabel('Зарплата (RUB)')
    ax.set_title('Сравнение средней и медианной предлагаемой зарплаты по направлениям (RUB)')
    ax.legend()

    # Подписи значений над столбцами (необязательно)
    def autolabel(bars):
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax.annotate(f'{h:,.0f}',
                            xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    autolabel(bars_mean)
    autolabel(bars_med)

    plt.tight_layout()
    plt.savefig(save_path)
    print('Сохранён график:', save_path)
    plt.close()

    # Вывести таблицу результатов
    print('\nТаблица (direction, count, mean, median):')
    print(top[['direction', 'count', 'mean', 'median']].to_string(index=False))
