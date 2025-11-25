
from db.mongo_connection import get_db
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import Optional

# --- функция определения уровня (seniority) на основе текста вакансии ---
def detect_level(texts) -> Optional[int]:
    """
    Принимает строку или список строк (title, professional_roles, specializations и т.д.)
    Возвращает уровень в виде целого: 0=Intern/Trainee, 1=Junior, 2=Middle, 3=Senior,
    4=Lead/Principal/Head, 5=Manager/Director/CTO
    Возвращает None если уровень не определён.
    """
    if not texts:
        return None
    if isinstance(texts, (list, tuple)):
        combined = " ".join([str(t) for t in texts if t]).lower()
    else:
        combined = str(texts).lower()

    # шаблоны (англ/рус)
    mapping = [
        (r'\b(intern|trainee|стажер|стажёр)\b', 0),
        (r'\b(junior|jr\b|младш|младший|джуниор)\b', 1),
        (r'\b(middle|mid\b|мид|mид|средн)\b', 2),
        (r'\b(senior|sr\b|sen\b|старш|старший|сеньор)\b', 3),
        (r'\b(lead|principal|head|ведущ|ведущий|главн)\b', 4),
        (r'\b(manager|director|cto|chief|руководит|руководитель|директор)\b', 5),
    ]
    for patt, lvl in mapping:
        if re.search(patt, combined, flags=re.IGNORECASE):
            return lvl
    return None

# --- функция определения наличия предложения удалённой работы ---
def detect_remote(doc) -> bool:
    """
    Проверяет несколько полей вакансии на слова 'remote'/'удал'/'дистан' и т.п.
    doc: документ вакансии (словарь)
    """
    remote_keywords = ['remote', 'удал', 'дистант', 'дистанц', 'work from home', 'from home', 'home']
    fields_to_check = []

    # возможные места, где указано remote
    for key in ('schedule', 'employment', 'name', 'description', 'snippet', 'snippet.requirement', 'snippet.responsibility'):
        # защитимся от вложенных структур
        parts = key.split('.')
        val = doc
        try:
            for p in parts:
                if val is None:
                    val = None
                    break
                if isinstance(val, dict):
                    val = val.get(p)
                else:
                    # если поле не словарь — остановка
                    val = None
                    break
        except Exception:
            val = None
        if val:
            if isinstance(val, list):
                fields_to_check.extend([str(x).lower() for x in val])
            else:
                fields_to_check.append(str(val).lower())

    # также полевые списки 'professional_roles' и 'specializations'
    for maybe_list in ('professional_roles', 'specializations', 'key_skills', 'employer'):
        v = doc.get(maybe_list)
        if v:
            if isinstance(v, list):
                fields_to_check.extend([str(x).lower() for x in v])
            elif isinstance(v, dict):
                fields_to_check.extend([str(x).lower() for x in v.values()])
            else:
                fields_to_check.append(str(v).lower())

    text_all = " ".join(fields_to_check)
    for kw in remote_keywords:
        if kw in text_all:
            return True
    return False

# --- основная функция: сбор данных, расчёт и график ---
def level_remote_correlation_plot(save_path: str = 'level_remote_correlation.png', min_count_level:int = 10):
    db = get_db()
    coll = db['vacancies']

    # вытаскиваем только нужные поля (экономим трафик)
    cursor = coll.find(
        {},
        {
            'name': 1,
            'professional_roles': 1,
            'specializations': 1,
            'schedule': 1,
            'employment': 1,
            'description': 1,
            'snippet': 1,
            'key_skills': 1
        }
    )

    rows = []
    for doc in cursor:
        # составляем текст для определения уровня
        title = doc.get('name')
        roles = doc.get('professional_roles')
        specs = doc.get('specializations')
        combined_for_level = []
        if title:
            combined_for_level.append(title)
        if roles:
            # если это список словарей с полем 'name' — вытянем имена
            if isinstance(roles, list):
                combined_for_level.extend([r.get('name') if isinstance(r, dict) and r.get('name') else r for r in roles])
            elif isinstance(roles, dict):
                combined_for_level.append(str(roles))
            else:
                combined_for_level.append(str(roles))
        if specs:
            if isinstance(specs, list):
                combined_for_level.extend([s.get('name') if isinstance(s, dict) and s.get('name') else s for s in specs])
            else:
                combined_for_level.append(str(specs))

        level = detect_level(combined_for_level)

        remote_flag = detect_remote(doc)

        rows.append({'level': level, 'remote': int(remote_flag)})

    df = pd.DataFrame(rows)
    # отфильтруем неизвестные уровни
    df_known = df[df['level'].notna()].copy()
    if df_known.empty:
        print("Нет вакансий с определённым уровнем должности — ничего не построено.")
        return

    # статистика по уровням
    level_stats = df_known.groupby('level').agg(count=('remote', 'size'), remote_sum=('remote', 'sum'))
    level_stats['remote_rate'] = level_stats['remote_sum'] / level_stats['count']

    # отбрасываем уровни с очень малым количеством (шум)
    level_stats = level_stats[level_stats['count'] >= min_count_level]
    if level_stats.shape[0] < 2:
        print("Слишком мало уровней с >= {} вакансий для надёжной оценки.".format(min_count_level))
        print(level_stats)
        return

    # корреляция на уровне отдельных вакансий (point-biserial / Pearson между level и бинарной remote)
    x = df_known['level'].astype(float)
    y = df_known['remote'].astype(float)
    n = len(x)
    # Pearson r
    cov = np.cov(x, y, ddof=0)[0,1]
    r = cov / (x.std(ddof=0) * y.std(ddof=0))
    # попытаемся посчитать p-value для r (t-статистика)
    p_value = None
    try:
        from scipy import stats
        r_scipy, p_value = stats.pearsonr(x, y)
        r = r_scipy
    except Exception:
        # если scipy отсутствует — оценим p через t-статистику вручную (приближённо)
        try:
            t_stat = r * np.sqrt((n - 2) / (1 - r*r))
            # приближённое p через стандартное распределение Стьюдента
            import math
            # обращаемся к scipy только если есть; иначе p оставим None
            p_value = None
        except Exception:
            p_value = None

    # Построим график: bar (remote_rate по уровням) + точки (размер = количество вакансий) + линия тренда
    levels = level_stats.index.astype(int).to_list()
    rates = level_stats['remote_rate'].to_list()
    counts = level_stats['count'].to_list()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(levels, rates, width=0.6, label='Доля remote (по уровню)')
    # точки
    ax.scatter(levels, rates, s=np.clip(np.array(counts)/2, 20, 800), alpha=0.8, edgecolor='k', label='Уровни (размер = количество вакансий)')
    # тренд (линейная регрессия по агрегированным точкам)
    coef = np.polyfit(levels, rates, 1)
    poly1d_fn = np.poly1d(coef)
    xs = np.linspace(min(levels), max(levels), 100)
    ax.plot(xs, poly1d_fn(xs), linestyle='--', label='Линейный тренд')

    ax.set_xlabel('Уровень позиции (0=Intern, 1=Junior, 2=Middle, 3=Senior, 4=Lead, 5=Manager/Director)')
    ax.set_ylabel('Доля вакансий с remote')
    ax.set_title('Корреляция уровня позиции и вероятности предложения удалённой работы')
    ax.set_xticks(levels)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    # подпись с метриками
    text_stats = f"Pearson r ≈ {r:.3f}" + (f", p-value = {p_value:.3e}" if p_value is not None else "")
    ax.text(0.01, 0.01, text_stats, transform=ax.transAxes, fontsize=9, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # печатаем таблицу с уровнями
    print("Статистика по уровням (level -> count, remote_sum, remote_rate):")
    print(level_stats)
    print(text_stats)
    print("График сохранён:", save_path)
