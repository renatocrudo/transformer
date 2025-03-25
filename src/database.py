from dotenv import load_dotenv
import os
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
DATABASE = "skynet"
COLLECTION_CEP = "cep"
COLLECTION_ITBI = "itbi_import"
COLLECTION_CONVERSATIONS = "conversations"

db = client[DATABASE]
collection_cep = db[COLLECTION_CEP]
collection_itbi = db[COLLECTION_ITBI]
collection_conversations = db[COLLECTION_CONVERSATIONS]
