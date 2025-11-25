"""
1) Проверяет, распакована ли MongoDB в db/mongodb/ — если нет, запускает db/
setup_mongo.py
2) Запускает mongod (как фоновый процесс), ждёт готовности (ping)
3) Загружает data/vacancies_ib.ndjson в коллекцию vacancies
4) Запускает анализ (scripts/analyze_data.analyze_all)
После завершения mongod остаётся запущенным (чтобы можно было посмотреть
данные в MongoDB Compass).
PID mongod сохраняется в mongo.pid
"""
import subprocess
import time
from pathlib import Path
import sys
from db import setup_mongo

PROJECT = Path(__file__).resolve().parent
DB_MONGODB = PROJECT / 'db' / 'mongodb'
MONGO_DATA = PROJECT / 'mongo_data'
PID_FILE = PROJECT / 'mongo.pid'

def find_mongod_exe():
    for p in DB_MONGODB.rglob('mongod.exe'):
        return p
    return None

def start_mongod(mongod_path: Path, dbpath: Path, log_path: Path):
    # Запускаем в фоне и сохраняем PID
    cmd = [str(mongod_path), '--dbpath', str(dbpath), '--bind_ip', '127.0.0.1']
    logf = open(log_path, 'a', encoding='utf-8')
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.pid

def wait_for_mongo_ready(timeout_s=30):
    from pymongo import MongoClient
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
            client.admin.command('ping')
            return True
        except Exception:
            time.sleep(1)
    return False


def main():
    # 1) если mongodb не распакована — запустить setup
    if not DB_MONGODB.exists():
        print('MongoDB не найдена в db/mongodb/. Попытаюсь распаковать...')
        subprocess.run([sys.executable, str(PROJECT / 'db' / 'setup_mongo.py')], check=True)
    mongod = find_mongod_exe()
    if not mongod:
        print('mongod.exe не найден. Проверь db/setup_mongo.py и распакованный архив.')
        return
    
    MONGO_DATA.mkdir(exist_ok=True)
    print('Запускаю mongod:', mongod)
    pid = start_mongod(mongod, MONGO_DATA, PROJECT / 'mongo.log')
    print('mongod PID:', pid)
    PID_FILE.write_text(str(pid))
    
    print('Ожидаю готовности MongoDB...')
    ok = wait_for_mongo_ready(30)
    if not ok:
        print('MongoDB не ответила в течение таймаута. Посмотри mongo.log')
        return
    
    print('MongoDB готова. Загружаю данные...')
    # Загружаем данные
    sys.path.insert(0, str(PROJECT / 'scripts'))
    from load_data import load_ndjson_to_mongo
    
    # Загружаем вакансии
    vacancies_path = PROJECT / 'data' / 'vacancies_it.json'
    load_ndjson_to_mongo(str(vacancies_path))
    
    # Загружаем работодателей
    employers_path = PROJECT / 'data' / 'employers.json'
    load_ndjson_to_mongo(str(employers_path ), 'employers')
    
    print('Данные загружены. Запускаю анализ...')
    
    from scripts.analyze_data import analyze_all as analyze_data1
    analyze_data1()
    from scripts.analyze_data_29 import analyze_all as analyze_data_29
    analyze_data_29()
    from scripts.analyze_data3 import analyze_all as analyze_data3
    analyze_data3()
    
    print('Всё выполнено. MongoDB остаётся запущенной. PID сохранён в', PID_FILE)

if __name__ == '__main__':
    main()
