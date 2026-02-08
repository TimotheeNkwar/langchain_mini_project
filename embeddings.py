
import os
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS 
from config import VECTORSTORE_SAVE_PATH, DATA_PATH , MODEL_NAME


def create_vectorstore():
    """Create a new FAISS vectorstore from documents and save it."""
    
    loader = TextLoader(DATA_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Automatically save the vectorstore
    save_vectorstore(vectorstore)
    print(f"Vectorstore created and saved to {VECTORSTORE_SAVE_PATH}")

    return vectorstore


def save_vectorstore(vectorstore):
    """Save the FAISS vectorstore to disk."""
    vectorstore.save_local(VECTORSTORE_SAVE_PATH)


def load_vectorstore():
    """Load a FAISS vectorstore from disk if it exists."""
    if os.path.exists(VECTORSTORE_SAVE_PATH):
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        vectorstore = FAISS.load_local(
            VECTORSTORE_SAVE_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Vectorstore loaded from {VECTORSTORE_SAVE_PATH}")
        return vectorstore
    return None


def get_or_create_vectorstore():
    """Load existing vectorstore or create a new one if not found."""
    vectorstore = load_vectorstore()
    if vectorstore is None:
        print("Creating vectorstore (first use)...")
        vectorstore = create_vectorstore()
    return vectorstore