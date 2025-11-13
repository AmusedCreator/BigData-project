#!/usr/bin/env python3
# coding: utf-8

"""
filter_it_vacancies.py

Пример использования:
python filter_it_vacancies.py --vacancies hh_vacancies_all.json --outdir ./ --log log.log --verbose
"""

import json
import re
import os
import argparse
import logging
from typing import List, Dict, Any

# -----------------------
# Настройки ключевых слов
# -----------------------
IT_KEYWORDS = [
    # роли / должности
    "разработчик", "программист",
    "инженер-программист", "разработчик мобильных приложений",
    "frontend", "frontend-разработчик", "фронтенд", "фронтэнд",
    "backend", "backend-разработчик", "бэкенд",
    "fullstack", "full-stack", "full stack", "фуллстек", "фулл-стек",
    "devops", "системный администратор", "системный админ", "системныйадминистратор",
    "qa", "тестировщик", "тестировщица", "инженер по тестированию",
    "аналитик данных", "data analyst", "data scientist", "мастер данных",
    "машинного обучения", "машинное обучение", "ml", "искусственный интеллект",
    # языки и технологии (часто пишут латиницей)
    "python", "java", "javascript", "js", "c#", "c\\+\\+", "php", "golang", "go",
    "ruby", "scala", "kotlin", "swift", "android", "ios",
    # фреймворки и инструменты
    "django", "flask", "react", "angular", "vue", "node", "nodejs", "node.js",
    "sql", "postgres", "postgresql", "mysql", "mongodb", "nosql",
    "docker", "kubernetes", "k8s",
    # общие IT-термины
    "embedded", "firmware", "devops", "ci/cd", "continuous integration",
    "selenium", "rest", "api", "graphql"
]


def compile_pattern(keywords: List[str]) -> re.Pattern:
    cleaned = []
    seen = set()
    for kw in keywords:
        if not kw:
            continue
        s = kw.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(re.escape(s))
    if not cleaned:
        return re.compile(r'(?!x)x')
    pattern = r'\b(?:' + '|'.join(cleaned) + r')\b'
    return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)


PATTERN = compile_pattern(IT_KEYWORDS)


def load_vacancies(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        vals = list(data.values())
        if vals and isinstance(vals[0], dict):
            return vals
    raise ValueError("Не удалось распознать структуру JSON (ожидается список или объект с 'items').")


def get_text_for_search(v: Dict[str, Any]) -> str:
    parts = []
    for key in ("name", "title", "description"):
        if key in v and v[key]:
            parts.append(str(v[key]))
    ks = v.get("key_skills") or v.get("skills")
    if isinstance(ks, list):
        parts.extend(str(x.get("name") if isinstance(x, dict) else x) for x in ks)
    for key in ("specializations", "professional_roles"):
        val = v.get(key)
        if val:
            if isinstance(val, list):
                parts.extend(str(x.get("name") if isinstance(x, dict) else x) for x in val)
            else:
                parts.append(str(val))
    emp = v.get("employer")
    if isinstance(emp, dict) and emp.get("name"):
        parts.append(str(emp.get("name")))
    area = v.get("area")
    if isinstance(area, dict) and area.get("name"):
        parts.append(str(area.get("name")))
    if not parts:
        parts.append(json.dumps(v, ensure_ascii=False))
    return " ".join(parts)


def is_it(vac: Dict[str, Any], pattern: re.Pattern) -> bool:
    txt = get_text_for_search(vac)
    return bool(pattern.search(txt))


def main():
    parser = argparse.ArgumentParser(description="Filter IT vacancies (simple).")
    parser.add_argument("--vacancies", "-v", required=True, help="JSON файл вакансий")
    parser.add_argument("--outdir", "-o", default="output", help="Папка вывода")
    parser.add_argument("--log", "-l", default=None, help="Путь к лог файлу (по умолчанию <outdir>/filter_it.log)")
    parser.add_argument("--verbose", "-V", action="store_true", help="Включить подробный DEBUG лог")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log_path = args.log if args.log else os.path.join(args.outdir, "filter_it.log")

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("filter_it_simple")

    try:
        logger.info("Start. vacancies=%s", args.vacancies)
        vacancies = load_vacancies(args.vacancies)
        logger.info("Loaded %d vacancies", len(vacancies))

        selected = []

        for vac in vacancies:
            try:
                if is_it(vac, PATTERN):
                    selected.append(vac)
            except Exception as e:
                vid = vac.get("id") or vac.get("vacancy_id") or vac.get("vacancyId")
                logger.error("Ошибка при проверке вакансии %s: %s", vid or "<no-id>", e)

        # сохранить JSON только
        out_json = os.path.join(args.outdir, "vacancies_it.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)

        logger.info("Saved %d IT vacancies -> %s", len(selected), out_json)
        logger.info("Finished successfully.")
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=False)
        raise


if __name__ == "__main__":
    main()
