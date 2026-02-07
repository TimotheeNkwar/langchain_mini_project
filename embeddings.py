from langchain_community.document_loaders import TextLoader  #type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  #type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings  #type: ignore
from langchain_community.vectorstores import FAISS  #type: ignore


def create_vectorstore():
    loader = TextLoader("data/data.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore