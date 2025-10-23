# --- улучшённая версия: specialization_skills_analysis_refined ---
import re
import math
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from db.mongo_connection import get_db

# Токенизация/регекс для "технических" токенов (включая C++, C#, TCP/IP, IPv4, 3des, etc.)
_TECH_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9\+\#\/\.\-]{2,}", flags=re.UNICODE)

# Расширенный стоп-лист (рус+англ) для удаления не-навыков
BAD_TERMS = {
    # русскоязычные мусорные термины
    'опыт','опыта','опыт работы','требование','требования','обязанности','задачи',
    'работы','работа','ответственность','условия','условия работы','компания','команд',
    'проект','проекта','проекты','определение','знание','знания','образование','приветствуется',
    'желательно','выполнять','участие','поддержка','участвовать','сопровождение',
    'безопасность','безопасности','защита','защиты','информационной','информационные',
    # английские мусорные слова
    'experience','requirements','responsibilities','work','company','projects','project',
    'knowledge','skills','team','support','required','responsible','years','experience',
    # filler phrases
    'и','в','на','по','для','с','из','и т.п','и т.д','etc','other'
}

# whitelist — термины, которые даже если распространены, стоит сохранять
TECH_WHITELIST = {
    'python','java','c++','c#','c','sql','linux','aws','azure','gcp','docker',
    'kubernetes','k8s','splunk','elk','elasticsearch','kibana','wireshark','nmap',
    'metasploit','nessus','burpsuite','burp','firewall','iptables','snort','suricata',
    'tcp','udp','http','https','ssh','smb','ldap','rdp','sqlserver','postgresql','mysql',
    'git','ansible','terraform','powershell','bash','perl','ruby','go','golang','scala'
}

# нормализация распространённых вариантов
NORMALIZE_MAP = {
    'python3': 'python', 'python2': 'python',
    'py': 'python',
    'csharp': 'c#', 'c_sharp': 'c#', 'csharp.': 'c#',
    'mssql': 'sqlserver',
    'postgres': 'postgresql',
    'aws amazon': 'aws', 'amazon web services': 'aws',
    'k8s': 'kubernetes',
    'elk stack': 'elasticsearch',
    'elastic': 'elasticsearch',
    'js': 'javascript',
    'nodejs': 'node.js', 'node.js': 'node.js',
    'sql server': 'sqlserver',
}

def _normalize_term(t: str) -> str:
    if not t:
        return ''
    s = t.strip().lower()
    s = s.replace('(', '').replace(')', '')
    s = re.sub(r'[,;:]$', '', s)  # trim trailing punctuation
    # map common variants
    s = NORMALIZE_MAP.get(s, s)
    # collapse multiple spaces
    s = re.sub(r'\s+', ' ', s)
    return s

def _extract_text(doc):
    parts = []
    for key in ('description', 'snippet', 'snippet.requirement', 'snippet.responsibility'):
        parts_key = key.split('.')
        val = doc
        for p in parts_key:
            if not isinstance(val, dict):
                break
            val = val.get(p)
            if val is None:
                break
        if val:
            if isinstance(val, list):
                parts.extend([str(x) for x in val if x])
            else:
                parts.append(str(val))
    return " ".join(parts)

def _normalize_skill_item(x):
    if x is None:
        return None
    if isinstance(x, dict):
        # часто это {'name': 'Python'}
        return x.get('name') or x.get('title') or str(x)
    return str(x)

def specialization_skills_analysis_plot(save_png='spec_skills_refined.png',
                                          save_csv='spec_skills_refined.csv',
                                          top_specializations=8,
                                          top_k_terms=12,
                                          min_docs_per_spec=5,
                                          keyskill_weight=10,
                                          ubiquity_thresh=0.6):
    """
    Улучшённый анализ навыков по специализациям.
    - keyskill_weight: вес для явно указанных key_skills (по умолчанию = 10)
    - ubiquity_thresh: удаляем термины, которые встречаются в >= ubiquity_thresh доле выбранных специализаций,
      если только они не в TECH_WHITELIST.
    """
    db = get_db()
    coll = db['vacancies']
    cursor = coll.find({}, {'specializations':1, 'key_skills':1, 'description':1, 'snippet':1})

    # Группируем документы по специализациям
    spec_docs = defaultdict(list)
    total_docs = 0
    for doc in cursor:
        total_docs += 1
        specs = doc.get('specializations')
        if specs:
            spec_names = []
            if isinstance(specs, list):
                for s in specs:
                    if isinstance(s, dict):
                        nm = s.get('name') or s.get('title')
                        spec_names.append(nm if nm else str(s))
                    else:
                        spec_names.append(str(s))
            elif isinstance(specs, dict):
                nm = specs.get('name') or specs.get('title')
                spec_names = [nm] if nm else [str(specs)]
            else:
                spec_names = [str(specs)]
        else:
            spec_names = ['(NoSpecialization)']
        for sp in spec_names:
            spec_docs[sp].append(doc)

    if not spec_docs:
        print("Нет данных по specializations.")
        return

    # Выбираем топ специализаций по количеству документов (с >= min_docs_per_spec)
    spec_counts = {s: len(docs) for s, docs in spec_docs.items()}
    sorted_specs = sorted(spec_counts.items(), key=lambda x: x[1], reverse=True)
    selected_specs = [s for s,c in sorted_specs if c >= min_docs_per_spec][:top_specializations]
    if not selected_specs:
        print(f"Нет специализаций с >= {min_docs_per_spec} документов. Всего документов: {total_docs}.")
        return

    # Собираем частоты терминов: даём большой вес key_skills, меньший — токенам из текста
    per_spec_counter = {}
    global_spec_presence = Counter()  # в скольких специализациях встречается термин
    for spec in selected_specs:
        counter = Counter()
        docs = spec_docs[spec]
        seen_terms_in_spec = set()
        for doc in docs:
            # KEY SKILLS (явные) — дают большой вес
            ks = doc.get('key_skills') or doc.get('keySkills') or doc.get('key_skill')
            if ks:
                if isinstance(ks, list):
                    for k in ks:
                        norm = _normalize_skill_item(k)
                        if norm:
                            t = _normalize_term(norm)
                            if len(t) > 1 and t not in BAD_TERMS:
                                counter[t] += keyskill_weight
                                seen_terms_in_spec.add(t)
                elif isinstance(ks, dict):
                    for v in ks.values():
                        norm = _normalize_skill_item(v)
                        if norm:
                            t = _normalize_term(norm)
                            if len(t)>1 and t not in BAD_TERMS:
                                counter[t] += keyskill_weight
                                seen_terms_in_spec.add(t)
                else:
                    norm = _normalize_skill_item(ks)
                    t = _normalize_term(norm)
                    if len(t)>1 and t not in BAD_TERMS:
                        counter[t] += keyskill_weight
                        seen_terms_in_spec.add(t)

            # Текстовые токены (description/snippet)
            text = _extract_text(doc)
            if text:
                tokens = _TECH_TOKEN_RE.findall(text.lower())
                for tk in tokens:
                    tk_norm = _normalize_term(tk)
                    if not tk_norm or tk_norm in BAD_TERMS:
                        continue
                    # отбрасываем короткие «a», «в», двухсимвольные, если не tech-like
                    if len(tk_norm) < 2:
                        continue
                    counter[tk_norm] += 1
                    seen_terms_in_spec.add(tk_norm)

        # обновляем глобальное присутствие
        for t in seen_terms_in_spec:
            global_spec_presence[t] += 1

        per_spec_counter[spec] = counter

    # Удаляем слишком общие термины (встречаются почти во всех выбранных спеках),
    # кроме тех, что в whitelist (важные общие технологии).
    n_specs = len(selected_specs)
    ubiquity_cutoff = math.ceil(ubiquity_thresh * n_specs)
    filtered_per_spec = {}
    for spec, counter in per_spec_counter.items():
        filtered = Counter()
        for term, cnt in counter.items():
            doc_freq = global_spec_presence.get(term, 0)
            if term in TECH_WHITELIST:
                # сохраняем всегда
                filtered[term] = cnt
            else:
                if doc_freq >= ubiquity_cutoff:
                    # слишком распространён — пропускаем (рядом могут быть исключения)
                    continue
                else:
                    filtered[term] = cnt
        # дополнительно уберём «смысленные мусорные слова», если они ещё остались
        for bad in list(filtered.keys()):
            if bad in BAD_TERMS:
                del filtered[bad]
        filtered_per_spec[spec] = filtered

    # Формируем итоговую таблицу (top_k_terms)
    rows = []
    for spec in selected_specs:
        counter = filtered_per_spec.get(spec, Counter())
        if not counter:
            continue
        top = counter.most_common(top_k_terms)
        for term, score in top:
            rows.append({'specialization': spec, 'term': term, 'score': float(score)})

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        print("После фильтрации не осталось характерных терминов — попробуйте уменьшить ubiquity_thresh или min_docs_per_spec.")
        return

    # Сохраняем CSV
    df_out.to_csv(save_csv, index=False)
    print("Сохранён refined CSV:", save_csv)

    # Рисуем панели: по одной специализации — горизонтальный barh
    n = len(selected_specs)
    cols = 2
    rows_pl = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_pl, cols, figsize=(12, 4 * rows_pl))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, spec in enumerate(selected_specs):
        ax = axes[i]
        sub = df_out[df_out['specialization'] == spec].copy()
        if sub.empty:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
            ax.set_title(f"{spec} (n={spec_counts.get(spec,0)})")
            ax.axis('off')
            continue
        sub = sub.sort_values('score', ascending=True)
        ax.barh(sub['term'], sub['score'])
        ax.set_title(f"{spec} (n={spec_counts.get(spec,0)})")
        ax.set_xlabel('взвешенная частота (key_skills вес=10)')
        ax.tick_params(axis='y', labelsize=9)

    # выключаем лишние оси
    for j in range(len(selected_specs), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_png)
    plt.close()
    print("Сохранён refined PNG:", save_png)
    print("Готово. Топ-специализации:", selected_specs)
