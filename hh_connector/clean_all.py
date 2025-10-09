"""
Остановка mongod (если был запущен из run_all.py и записал PID в mongo.pid),
и удаление директорий db/mongodb/ и mongo_data/ и файла mongo.pid.
Скрипт просит подтверждение перед удалением.
"""
import shutil
from pathlib import Path
import subprocess
import os
import sys

PROJECT = Path(__file__).resolve().parent
PID_FILE = PROJECT / 'mongo.pid'
DB_MONGODB = PROJECT / 'db' / 'mongodb'
MONGO_DATA = PROJECT / 'mongo_data'

def stop_mongo_by_pidfile():
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        print('Найден PID:', pid)
        try:
            # Windows: taskkill
            subprocess.run(['taskkill', '/F', '/PID', str(pid)], check=False)
            print('Попытка завершить процесс PID', pid)
        except Exception as e:
            print('Ошибка при попытке завершить процесс:', e)
        try:
            PID_FILE.unlink()
        except Exception:
            print('Ошибка при попытке заверщить 2')
            pass
    else:
        print('PID-файл не найден. Процессы mongod будут пытаться завершаться по имени.')
        # Попробуем убить mongod по имени (Windows)
        subprocess.run(['taskkill', '/F', '/IM', 'mongod.exe'], check=False)

if __name__ == '__main__':
    print('ВНИМАНИЕ: это удалит локально распакованную MongoDB (db/mongodb/) и данные (mongo_data/)')
    ans = input("Подтверждаешь удаление (введи 'yes')? ")
    if ans.strip().lower() != 'yes':
        print('Отмена.')
        sys.exit(0)
    
    stop_mongo_by_pidfile()
    
    if DB_MONGODB.exists():
        print('Удаляю', DB_MONGODB)
        shutil.rmtree(DB_MONGODB, ignore_errors=True)
    if MONGO_DATA.exists():
        print('Удаляю', MONGO_DATA)
        shutil.rmtree(MONGO_DATA, ignore_errors=True)
    
    print('Готово.')

