"""
hh_fetch_ib_fixed_tz.py
Исправленная версия — фиксы User-Agent + timezone-aware сравнение дат.
"""

import requests
import time
import json
from datetime import datetime, timedelta, timezone
from dateutil import parser
from tqdm import tqdm
from requests.adapters import HTTPAdapter, Retry

BASE = "https://api.hh.ru/vacancies"

# --- ОБЯЗАТЕЛЬНО: укажите корректный контакт (email или сайт) внутри User-Agent ---
USER_AGENT = "StudentProject/1.0 (mailto:uhovad03@gmail.com)"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8"
}

IB_KEYWORDS = [
    "информационная безопасность", "информационной безопасности", "иб", "безопасность информации",
    "security", "information security", "infosec", "siem", "soc", "soc analyst",
    "penetration", "pentest", "пентест", "pentester", "security engineer", "devsecops",
    "forensics", "cybersecurity", "cyber security"
]

def is_ib_vacancy(vac):
    text = " ".join([
        vac.get("name","") or "",
        vac.get("snippet",{}).get("requirement","") or "",
        vac.get("snippet",{}).get("responsibility","") or "",
        vac.get("employer",{}).get("name","") or ""
    ]).lower()
    skills = " ".join([k.get("name","") for k in vac.get("key_skills",[]) ]).lower()
    combined = (text + " " + skills)
    for kw in IB_KEYWORDS:
        if kw.lower() in combined:
            return True
    for s in vac.get("specializations", []):
        if "безопас" in (s.get("name","").lower()):
            return True
    return False

def create_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET", "POST"])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(HEADERS)
    return s

def parse_pub_date(pub_str):
    """Возвращает timezone-aware datetime в UTC или None."""
    if not pub_str:
        return None
    try:
        dt = parser.parse(pub_str)
    except Exception:
        return None
    # если нет tzinfo — считаем, что это UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

def fetch_vacancies_by_text(text_query, start_date, out_file, max_pages_per_query=500):
    per_page = 100
    page = 0
    total_saved = 0
    session = create_session()
    pbar = tqdm(total=999999, desc=f"query='{text_query}'", unit="vac")
    while True:
        params = {
            "text": text_query,
            "page": page,
            "per_page": per_page,
            "order_by": "publication_time"
        }
        try:
            resp = session.get(BASE, params=params, timeout=30)
        except Exception as e:
            print("Request failed:", e)
            time.sleep(5)
            continue

        if resp.status_code == 400:
            print("HTTP 400 — Bad Request. Ответ сервера:", resp.text)
            pbar.close()
            return total_saved

        if resp.status_code == 429:
            wait = 10
            print("HTTP 429 — rate limited. Ждём", wait, "сек.")
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            print("HTTP", resp.status_code, resp.text)
            time.sleep(5)
            continue

        data = resp.json()
        items = data.get("items", [])
        if not items:
            break

        stop_when_older = False
        with open(out_file, "a", encoding="utf-8") as f:
            for vac in items:
                pub = vac.get("published_at")
                pub_dt = parse_pub_date(pub)
                # Теперь сравниваем aware (UTC) с aware start_date (UTC)
                if pub_dt and pub_dt < start_date:
                    stop_when_older = True
                    break
                if is_ib_vacancy(vac):
                    f.write(json.dumps(vac, ensure_ascii=False) + "\n")
                    total_saved += 1
                    pbar.update(1)

        page += 1
        pages_count = data.get("pages")
        if pages_count is not None and page >= pages_count:
            break
        if stop_when_older:
            break
        if page > max_pages_per_query:
            print("Достигнут предел страниц для одного запроса.")
            break
        time.sleep(0.35)
    pbar.close()
    print(f"Сохранено всего {total_saved} вакансий (для запроса '{text_query}')")
    return total_saved

if __name__ == "__main__":
    # делаем start_date timezone-aware (UTC)
    start_date = datetime.now(timezone.utc) - timedelta(days=365*3)
    out_file = "vacancies_ib.ndjson"
    open(out_file, "w", encoding="utf-8").close()

    queries = [
        "информационная безопасность",
        "информационной безопасности",
        "security engineer",
        "soc analyst",
        "pentester"
    ]

    total = 0
    for q in queries:
        total += fetch_vacancies_by_text(q, start_date, out_file)

    print("Готово. Всего сохранено:", total)
