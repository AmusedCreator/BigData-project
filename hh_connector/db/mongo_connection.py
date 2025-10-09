"""Небольшая обёртка для подключения к локальной MongoDB в проекте."""
from pymongo import MongoClient
from pathlib import Path

def get_mongo_client(uri: str = "mongodb://localhost:27017/", 
                server_selection_timeout_ms: int = 2000):
    client = MongoClient(uri,
                serverSelectionTimeoutMS=server_selection_timeout_ms)
    return client

def get_db(db_name: str = "hh_data"):
    client = get_mongo_client()
    return client[db_name]
