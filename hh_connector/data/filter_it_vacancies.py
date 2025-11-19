#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import argparse
from pathlib import Path
from typing import Iterable, Iterator
from decimal import Decimal
import datetime


# --- (ваши константы без изменений) ---
IT_ROLE_IDS = {
    "10","25","36","96","104","107","112","113","114","116",
    "121","124","125","126","148","150","156","160","163","164","165"
}
ROLE_40_ID = "40"
IT_KEYWORDS = [
    "программист",  "software engineer", "software developer",
    "frontend", "backend", "fullstack", "web developer", "mobile developer",
    "android", "ios", "flutter", "react native",
    "devops", "dev ops", "sre", "site reliability engineer",
    "docker", "kubernetes", "k8s", "ansible", "terraform",
    "ci/cd", "jenkins", "gitlab ci", "github actions",
    "aws", "azure", "gcp", "cloud engineer", "cloud architect",
    "data scientist", "data science", "machine learning", "deep learning",
    "ai engineer", "ml engineer", "data engineer", "big data", "hadoop", "spark",
    "pytorch", "tensorflow",
    "qa engineer", "qa automation", "quality assurance", "тестировщик", "тестирование",
    "automation engineer", "selenium", "pytest", "unit test",
    "системный администратор", "system administrator", "system engineer",
    "сетевой инженер", "network engineer", "сетевой администратор",
    "информационная безопасность", "cybersecurity", "security engineer", "security analyst",
    "pentester", "ethical hacker", "soc analyst",
    "python", "java", "javascript", "typescript", "c\\+\\+", "c#", "csharp",
    "go", "golang", "php", "ruby", "rust", "kotlin", "swift", "scala",
    "sql", "mysql", "postgres", "postgresql", "mongodb", "redis",
    "linux", "unix", "bash", "powershell", "windows server",
    "react", "vue", "angular", "node.js", "express", "django", "flask", "fastapi",
    "spring boot", "laravel", "symfony", "asp.net", "wpf",
    "api developer", "backend engineer", "frontend engineer",
    "software architect", "solution architect", "technical writer",
    "test automation", "test engineer",
    "product analyst", "product manager it", "it project manager",
    "scrum master", "agile coach"
]
# --- end constants ---

def _json_default(obj):
    """
    Преобразует Decimal, datetime, bytes, set в сериализуемые типы.
    При необходимости замените float(obj) на str(obj) для сохранения точности.
    """
    if isinstance(obj, Decimal):
        # Если хотите сохранить точность - используйте str(obj) вместо float(obj)
        return float(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {obj.__class__.__name__} not serializable")


def _normalize_role_id(role_id):
    if role_id is None:
        return None
    return str(role_id)


def vacancy_has_role(vacancy: dict, role_id: str) -> bool:
    roles = vacancy.get("professional_roles") or vacancy.get("professions") or []
    if not isinstance(roles, list):
        return False
    for r in roles:
        rid = _normalize_role_id(r.get("id")) if isinstance(r, dict) else _normalize_role_id(r)
        if rid == role_id:
            return True
    return False


def vacancy_has_it_role(vacancy: dict, allowed_ids: Iterable[str]) -> bool:
    roles = vacancy.get("professional_roles") or vacancy.get("professions") or []
    if not isinstance(roles, list):
        return False
    for r in roles:
        if isinstance(r, dict):
            rid = _normalize_role_id(r.get("id"))
        else:
            rid = _normalize_role_id(r)
        if rid and rid in allowed_ids:
            return True
    return False


def is_it_by_text(vacancy: dict, logger: logging.Logger = None) -> bool:
    pieces = []
    pieces.append(vacancy.get("name") or "")
    snippet = vacancy.get("snippet") or {}
    pieces.append(snippet.get("requirement") or "")
    pieces.append(snippet.get("responsibility") or "")
    pieces.append(vacancy.get("description") or "")
    skills = vacancy.get("key_skills") or []
    if isinstance(skills, list):
        pieces.extend([s.get("name", "") if isinstance(s, dict) else str(s) for s in skills])
    employer = vacancy.get("employer") or {}
    pieces.append(employer.get("name") or "")
    text = " ".join([str(p) for p in pieces if p]).lower()
    for kw in IT_KEYWORDS:
        plain_kw = kw.replace("\\", "").lower()
        if plain_kw in text:
            if logger:
                logger.debug("Match keyword '%s' in vacancy id=%s", plain_kw, vacancy.get("id"))
            return True
    return False


def stream_vacancies(filename: Path, logger: logging.Logger) -> Iterator[dict]:
    """
    Итеративно возвращает объекты вакансий из файла.
    Поддерживает:
      - NDJSON (одна JSON-объект на строку) — читаем построчно
      - Большой JSON-массив / {items: [...]} — через ijson (рекомендуется ставить ijson)
    """
    filesize = filename.stat().st_size
    logger.info("stream_vacancies: file=%s size=%d bytes", filename, filesize)

    with filename.open("r", encoding="utf-8") as f:
        # пробуем определить формат по первым 1024 байтам
        sample = f.read(1024)
        f.seek(0)
        s = sample.lstrip()
        if not s:
            logger.warning("Входной файл пуст.")
            return

        # Если выглядит как JSON-массив/объект (начинается с '[' или '{'), попробуем потоковый парсер ijson
        if s.startswith("[") or s.startswith("{"):
            try:
                import ijson  # потоковый парсер
                # если это объект с ключом items: use 'items.item', иначе топ-уровнев массив -> 'item'
                # но ijson.items(f, 'item') всегда переберёт элементы массива; для 'items.item' — элементы в items
                # Для универсальности попробуем сначала items.item, затем fallback на item
                f.seek(0)
                # detect top-level token (very small peek)
                if s.startswith("{"):
                    # попробуем items.item (obj with items key)
                    try:
                        for obj in ijson.items(f, 'items.item'):
                            yield obj
                        # если итератор пустой (нет items), нужно попробовать найти массив на верхнем уровне:
                        f.seek(0)
                        for obj in ijson.items(f, 'item'):
                            yield obj
                    except Exception:
                        # fallback: ищем любой list на верхнем уровне (может бросить)
                        f.seek(0)
                        for prefix, event, value in ijson.parse(f):
                            # не реализуем сложную логику — пробуем простой items.item и item выше
                            pass
                else:
                    # начинаеся с '[' — читаем элементы массива
                    f.seek(0)
                    for obj in ijson.items(f, 'item'):
                        yield obj
                return
            except ImportError:
                logger.warning("ijson не установлен — если файл большой JSON-массив, это может вызвать MemoryError. Установите 'ijson' для потокового парсинга.")
            except Exception as e:
                logger.warning("ijson не смог распарсить файл (интеллектуальный fallback): %s", e)
                # будем пробовать дальше — но не будем пытаться json.load для огромного файла

        # Если не JSON-массив/объект или ijson недоступен — пробуем NDJSON построчно
        f.seek(0)
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield obj
            except json.JSONDecodeError:
                logger.warning("Строка %d не валидный JSON — пропускаю.", i)
                continue


def vacancy_matches(vac: dict, args, logger: logging.Logger) -> bool:
    try:
        if vacancy_has_it_role(vac, IT_ROLE_IDS):
            return True
        if vacancy_has_role(vac, ROLE_40_ID):
            if args.role40_include_all:
                logger.debug("Включаю вакансию id=%s поскольку role 40 и --role-40-include-all", vac.get("id"))
                return True
            if args.role40_by_keyword:
                if is_it_by_text(vac, logger):
                    logger.debug("Включаю вакансию id=%s поскольку role 40 и найдено IT-совпадение по тексту", vac.get("id"))
                    return True
                else:
                    logger.debug("Пропускаю вакансию id=%s с role 40 — IT-ключевые слова не найдены", vac.get("id"))
                    return False
        return False
    except Exception as e:
        logger.warning("Ошибка при проверке вакансии id=%s: %s", vac.get("id"), e)
        return False


def stream_filter_and_save(input_path: Path, out_path: Path, ndjson: bool, args, logger: logging.Logger):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_in = 0
    total_matched = 0

    if ndjson:
        with out_path.open("w", encoding="utf-8") as fout:
            for vac in stream_vacancies(input_path, logger):
                total_in += 1
                if vacancy_matches(vac, args, logger):
                    fout.write(json.dumps(vac, ensure_ascii=False, default=_json_default) + "\n")
                    total_matched += 1
                    if total_matched % 1000 == 0:
                        logger.info("Matched %d vacancies so far...", total_matched)
    else:
        # записываем потоково в JSON-массив: [item, item, ...]
        with out_path.open("w", encoding="utf-8") as fout:
            fout.write("[\n")
            first = True
            for vac in stream_vacancies(input_path, logger):
                total_in += 1
                if vacancy_matches(vac, args, logger):
                    if not first:
                        fout.write(",\n")
                    fout.write(json.dumps(vac, ensure_ascii=False, indent=2, default=_json_default))
                    first = False
                    total_matched += 1
                    if total_matched % 1000 == 0:
                        logger.info("Matched %d vacancies so far...", total_matched)
            fout.write("\n]\n")

    logger.info("Обработано входных записей: %d, отобрано: %d", total_in, total_matched)


def main():
    parser = argparse.ArgumentParser(description="Фильтр вакансий hh.ru по IT professional_roles (учитывает role 40).")
    parser.add_argument("--input", "-i", type=str, default="hh_vacancies_all.json",
                        help="Путь к входному файлу (JSON / NDJSON / API-ответ).")
    parser.add_argument("--output-name", "-o", type=str, default="hh_vacancies_it.json",
                        help="Имя выходного файла (по умолчанию hh_vacancies_it.json).")
    parser.add_argument("--output-dir", "-d", type=str, default=".",
                        help="Папка для сохранения выходного файла (по умолчанию текущая).")
    parser.add_argument("--log-file", "-l", type=str, default="filter_it_vacancies.log",
                        help="Путь к файлу лога (по умолчанию filter_it_vacancies.log).")
    parser.add_argument("--log-level", type=str, choices=["DEBUG","INFO","WARNING","ERROR"], default="INFO",
                        help="Уровень логирования (DEBUG/INFO/WARNING/ERROR).")
    parser.add_argument("--ndjson", action="store_true",
                        help="Сохранить результат в NDJSON (одна вакансия на строку).")
    parser.add_argument("--role-40-by-keyword", dest="role40_by_keyword", action="store_true",
                        help="Если у вакансии роль 40 (Другое), считать её IT только если в текстах найдены IT-ключевые слова (по умолчанию).")
    parser.add_argument("--role-40-include-all", dest="role40_include_all", action="store_true",
                        help="Включать все вакансии с ролью 40 без дополнительной текстовой проверки (опасно — может добавить нерелевантные вакансии).")
    parser.set_defaults(role40_by_keyword=True, role40_include_all=False)

    args = parser.parse_args()

    # Настройка логирования
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("filter_it_vacancies")
    logger.setLevel(getattr(logging, args.log_level))
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(getattr(logging, args.log_level))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Запуск фильтра. Input=%s output_dir=%s output_name=%s ndjson=%s log_file=%s role40_by_keyword=%s role40_include_all=%s",
                args.input, args.output_dir, args.output_name, args.ndjson, args.log_file, args.role40_by_keyword, args.role40_include_all)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Входной файл не найден: %s", input_path)
        return

    out_path = Path(args.output_dir) / args.output_name

    try:
        stream_filter_and_save(input_path, out_path, args.ndjson, args, logger)
    except Exception as e:
        logger.exception("Ошибка при обработке: %s", e)
        return

    logger.info("Фильтрация завершена.")

if __name__ == "__main__":
    main()
