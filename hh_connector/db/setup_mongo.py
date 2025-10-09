"""
setup_mongo.py
--------------
Если в папке db/ уже лежит архив MongoDB (например mongodb.zip),
этот скрипт просто распакует его в db/mongodb/ и проверит, что
mongod.exe найден.

Важно:
- Положи свой архив (например mongodb-windows-x86_64-7.0.5.zip)
  в папку db/ и переименуй в mongodb.zip
- После этого запусти: python db/setup_mongo.py
"""

from pathlib import Path
import zipfile
import os

def extract_zip(zip_path: Path, dest_dir: Path):
    """Распаковывает архив в указанную директорию."""
    print(f"Unpacking {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
    print("Unpacking is complete.")

def find_mongod(dest_dir: Path):
    """Ищет mongod.exe в каталоге dest_dir (рекурсивно)."""
    for p in dest_dir.rglob('mongod.exe'):
        return p
    return None

def main():
    project = Path(__file__).resolve().parents[1]
    db_dir = project / 'db'
    mongodb_dir = db_dir / 'mongodb'
    zip_path = db_dir / 'mongodb.zip'

    if not zip_path.exists():
        print(f"Archive not found {zip_path}.")
        print("Put MongoDB ZIP to the folder db/ under the name mongodb.zip and re-launch.")
        return

    if mongodb_dir.exists():
        print(f"Folder {mongodb_dir} already exists. Skipping unpacking.")
        exe = find_mongod(mongodb_dir)
        if exe:
            print("Found mongod.exe:", exe)
        else:
            print("mongod.exe not found, check the contents of the archive.")
        return

    mongodb_dir.mkdir(parents=True, exist_ok=True)
    extract_zip(zip_path, db_dir)

    exe = find_mongod(mongodb_dir)
    if exe:
        print(f" MongoDB has been successfully unpacked. Found mongod.exe: {exe}")
    else:
        print(" Unpacking went through, but mongod.exe not found, check the archive structure.")

if __name__ == '__main__':
    main()