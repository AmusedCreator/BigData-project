from db.mongo_connection import get_db
from pprint import pprint

def main():
    db = get_db()
    collection = db["vacancies"]

    print(f" Количество документов: {collection.count_documents({})}")
    print("\nПример записей:")
    for doc in collection.find().limit(5):
        pprint(doc)

if __name__ == "__main__":
    main()