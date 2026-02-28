import os
from dotenv import load_dotenv

load_dotenv()


# FAISS Vectorstore Configuration
VECTORSTORE_SAVE_PATH = "./vectorstore_cache"
DATA_PATH = "data/book.txt"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "langchain_qa"
COLLECTION_NAME = "conversations"