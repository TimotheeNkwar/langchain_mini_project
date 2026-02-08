import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FAISS Vectorstore Configuration
VECTORSTORE_SAVE_PATH = "./vectorstore_cache"
DATA_PATH = "data/data.txt"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"