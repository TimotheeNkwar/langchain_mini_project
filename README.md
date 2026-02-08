# 🎯 LangChain Mini Project - Q&A Assistant

A simple and clean Python project demonstrating how **LangChain** works with a practical Q&A assistant that answers questions based on uploaded documents.

> 🚀 **Get started in 5 minutes!** | 📚 **Learn by example** | 🔧 **No API keys required**

## ✨ Features at a Glance

- ✅ **Zero API Dependencies** - Uses local embeddings and LLM
- ✅ **Production-Ready Architecture** - Modern LCEL (LangChain Expression Language) syntax
- ✅ **Comprehensive Knowledge Base** - 1000+ lines of LangChain documentation included
- ✅ **Easy to Understand** - Clean, well-documented code with step-by-step workflow
- ✅ **Fully Extensible** - Simple structure makes it easy to add features
- ✅ **Educational Value** - Includes 30 test questions with expert-level scoring rubric
- ✅ **Fast** - FAISS vector database for near-instant retrieval
- ✅ **Vectorstore Persistence** - Save and reload FAISS cache for faster startup times
- ✅ **Conversation Memory** - MongoDB integration for persistent multi-turn conversations
- ✅ **Document Conversion** - Built-in PDF/DOCX to text conversion with Docling

## 📚 Project Overview

This project implements a **Question & Answer system** that:

1. Loads a text document (`data.txt`) containing your knowledge base
2. Splits it into manageable chunks (500 chars with 50 char overlap)
3. Creates embeddings (numerical representations) of the text
4. Stores embeddings in a FAISS vector database for fast similarity search
5. Answers user questions by retrieving relevant context and using an LLM to generate responses

It's a practical implementation of the **Retrieval-Augmented Generation (RAG)** pattern, perfect for learning how modern NLP systems work.

---

## ⚡ Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
cd langchain_mini_project
uv pip install -r requirements.txt
```

### 2. Run the Q&A System

```bash
python app.py
```

### 3. Ask Questions

```
❓ Your question: What is LangChain?
🤖 Answer: LangChain is a framework for developing applications powered by language models. It enables applications that are: data-aware (connect a language model to other sources of data), agentic (allow a language model to interact with its environment), and modular...

❓ Your question: What are the main components?
🤖 Answer: The main components include: LLMs (Language Models), Prompts, Chains, Memory, Retrievers, Document Loaders, Text Splitters, Embeddings, and Vector Stores. Each serves a specific purpose in the knowledge pipeline...

❓ Your question: exit
[Program exits]
```

**That's it!** You now have a working Q&A system. 🎉

---

## 🛠️ How It Works

### Architecture Flow

```text
┌─────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  data/data.txt → TextLoader → RecursiveCharacterSplitter   │
│      ↓              ↓              ↓                         │
│   Raw Text    Split Chunks   Chunk Management               │
│                      ↓                                       │
│              HuggingFaceEmbeddings                           │
│                      ↓                                       │
│              FAISS Vector Store                              │
│                      ↓                                       │
│         (Ready for similarity search)                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          ↑
                          │
┌─────────────────────────────────────────────────────────────┐
│                     RETRIEVAL PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Question → Embed Question → Search Vector Store      │
│        ↓               ↓               ↓                     │
│   Input Text   Numerical Vector  Find K=4 Similar Chunks    │
│                                      ↓                       │
│                          Retrieve Context from K Chunks      │
│                                      ↓                       │
│                        Format Context + Question             │
│                                      ↓                       │
│                   OllamaLLM (Generate Answer)                │
│                                      ↓                       │
│                          Return Final Answer                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Step-by-Step Process

#### 1️⃣ **Document Loading** (`embeddings.py`)

- Reads `data/data.txt` using `TextLoader`
- Loads full document content into memory as raw text
- Validates file exists and contains expected content

#### 2️⃣ **Text Splitting** (`embeddings.py`)

- Splits document into overlapping chunks:
  - **Chunk size**: 500 characters per chunk
  - **Overlap**: 50 characters between chunks
  - **Why**: Maintains context at chunk boundaries, prevents losing meaning
- Uses `RecursiveCharacterTextSplitter` (respects paragraph breaks first, then sentences, then words)

**Example:**

```text
Original chunk ends with: "...Vector stores like FAISS" 
Next chunk starts with:   "...FAISS and Pinecone"
(50 char overlap preserves connection)
```

#### 3️⃣ **Embeddings Creation** (`embeddings.py`)

- Converts each text chunk to a numerical vector (384 dimensions)
- Uses `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2`
- All processing is **local** - no API calls, no cloud dependency
- First run downloads model (~50MB)

**Why HuggingFace?**

- ✅ Free and open-source
- ✅ Fast inference (CPU-based)
- ✅ Privacy-preserving (no data leaves machine)
- ✅ 384-dimensional embeddings (good quality for cost)

#### 4️⃣ **Vector Store Creation** (`embeddings.py`)

- Stores all embeddings in FAISS (Facebook AI Similarity Search)
- Creates searchable index of document chunks
- Enables fast nearest-neighbor search
- Runs entirely in-memory (perfect for this project size)

#### 5️⃣ **Retrieval Chain** (`chains.py`)

- When a question is asked:
  1. Converts question to an embedding
  2. Searches FAISS for 4 most similar chunks
  3. Retrieves full text of those chunks
  4. Formats them as context

- Uses **LCEL (LangChain Expression Language)**:

```python
qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

#### 6️⃣ **LLM Processing** (`chains.py`, `llm.py`, `app.py`)

- Sends formatted prompt with context to `OllamaLLM`
- LLM reads context chunks and generates answer
- Uses `temperature=0.9` for balanced creativity

**Flow:**

- Question + Context → LLM → Contextual Answer
- Model: `llama-2-7b-chat` (7 billion parameters)
- Response time: ~1-3 seconds on CPU

---

## 📁 Project Structure

```text
langchain_mini_project/
├── README.md                    # 📖 This file
├── app.py                       # 🚀 Main entry point
├── llm.py                       # 🧠 LLM initialization
├── embeddings.py                # 🔗 Document & vector store setup
├── chains.py                    # ⛓️  Retrieval chain (LCEL syntax)
├── memory.py                    # 💾 MongoDB conversation memory
├── doc.py                       # 📄 Document conversion (PDF/DOCX to text)
├── config.py                    # ⚙️  Configuration & env loading
├── utils.py                     # 🔧 Cache management utilities
├── requirements.txt             # 📦 Python dependencies
├── data/
│   └── data.txt                 # 📚 Knowledge base (2500+ lines)
├── vectorstore_cache/           # 💾 FAISS cache (auto-generated)
└── testing_questions.txt        # ❓ 30 test questions with rubric (including memory tests)
```

### Detailed File Descriptions

| File | Purpose | Key Features |
|------|---------|--------------|
| **app.py** | Main CLI interface | Interactive loop, error handling, vectorstore auto-loading |
| **llm.py** | LLM initialization | Loads `llama-2-7b-chat`, sets temperature=0.9 |
| **embeddings.py** | Document pipeline | Loading → Splitting → Embedding → Vector Store → Caching |
| **chains.py** | LCEL retrieval chain | Pipe syntax, context formatter, prompt template |
| **memory.py** | MongoDB conversation memory | Persistent storage, conversation tracking, history retrieval |
| **doc.py** | Document conversion | PDF/DOCX to text using Docling, markdown export |
| **config.py** | Configuration management | Environment variables, cache paths, MongoDB settings |
| **utils.py** | Cache utilities | Clear, rebuild, and check vectorstore cache status |
| **data/data.txt** | Knowledge base | Comprehensive LangChain documentation |
| **testing_questions.txt** | Test suite | 25 questions, 7 categories, scoring rubric |

### Code Overview

#### **app.py** - The Entry Point

```python
# Interactive Q&A loop
while True:
    question = input("❓ Your question: ")
    if question.lower() == "exit":
        break
    
    # Invoke the chain with the question
    response = chain.invoke({"question": question})
    print(f"🤖 Answer: {response}\n")
```

#### **llm.py** - LLM Configuration

```python
from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="llama-2-7b-chat",    # Local model
    temperature=0.9,             # 0=factual, 1=creative
)
```

#### **embeddings.py** - Document Processing Pipeline

```python
# 1. Load document
loader = TextLoader("data/data.txt")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(documents)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Build vector store
vector_store = FAISS.from_documents(
    chunks, 
    embeddings
)
```

#### **chains.py** - LCEL Retrieval Chain

```python
from langchain_core.runnables import RunnablePassthrough

# Modern LangChain syntax (LCEL)
qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()  # Parse LLM output to string
)

# Usage:
response = qa_chain.invoke({"question": "What is LangChain?"})
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- `uv` package manager (or `pip`)

### Setup

1. **Clone or navigate to the project**

```bash
cd langchain_mini_project
```

1. **Install dependencies with uv**

```bash
uv pip install -r requirements.txt
```

Or with pip:

```bash
pip install -r requirements.txt
```

1. **Download the embedding model (one-time)**
The first run will download the `sentence-transformers/all-MiniLM-L6-v2` model (~50MB).

---

## 💻 Usage

### Run the Q&A Assistant

```bash
python app.py
```

### Example Questions

```text
❓ Your question: What is LangChain?
🤖 Answer: LangChain is a powerful framework for developing applications...

❓ Your question: What are the key components?
🤖 Answer: The key components include LLMs, Prompts, Chains, Memory...

❓ Your question: exit
[Program exits]
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file if you need API keys (currently not needed):

```bash
OPENAI_API_KEY=your_key_here  # Not required for this project
```

### Model Parameters (in `llm.py`)

- `model="llama-2-7b-chat"` - The language model to use
- `temperature=0.9` - Creativity level (0=factual, 1=creative)

### Chunk Settings (in `embeddings.py`)

- `chunk_size=500` - Size of text chunks
- `chunk_overlap=50` - Overlap between chunks

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | Core framework |
| `langchain-community` | Additional integrations |
| `langchain-huggingface` | HuggingFace embeddings |
| `langchain-ollama` | Local LLM integration |
| `langchain-text-splitters` | Text chunking |
| `sentence-transformers` | Embedding model |
| `faiss-cpu` | Vector database |
| `pymongo` | MongoDB integration |
| `docling` | Document conversion (PDF/DOCX) |
| `python-dotenv` | Environment variables |

---

## 🎓 Learning Resources

This project teaches:

- **Document Processing**: Loading and splitting text
- **Embeddings**: Converting text to vectors
- **Vector Databases**: Storing and searching embeddings
- **Retrieval Chains**: Finding relevant context
- **LLM Integration**: Using local language models
- **LCEL**: LangChain Expression Language syntax

---

## 🔄 How the Chain Works (LCEL Syntax)

```python
qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

This creates a pipeline where:

1. The retriever gets relevant context chunks
2. The question passes through unchanged
3. Both are formatted into a prompt
4. The LLM processes and generates an answer

---

## 🌐 No API Keys Required

This project is completely local:

- ✅ Uses local embedding model (HuggingFace)
- ✅ Uses local LLM (Ollama/Llama2)
- ✅ No OpenAI API calls
- ✅ Optional MongoDB for conversation memory
- ✅ Runs entirely offline (after first model download)

---

## 📝 Notes

- First run creates and caches the vectorstore (~30-60 seconds)
- Subsequent runs load the cached vectorstore in ~2-3 seconds
- Cache is stored in `vectorstore_cache/` directory (auto-created)
- To update the cache, run: `python utils.py rebuild`
- Responses depend on the quality of documents in `data/data.txt`
- You can replace `data.txt` with any text file you want to analyze

---

## 🤝 Common Issues & Solutions

| Issue                   | Solution                                           |
| ----------------------- | -------------------------------------------------- |
| `ModuleNotFoundError`   | Run `uv pip install -r requirements.txt`           |
| Slow first run          | First request downloads embedding model, be patient |
| No response from LLM    | Ensure Ollama is running (if using Ollama locally) |
| MongoDB connection error | Ensure MongoDB is running on localhost:27017       |
| Type warnings           | Disable Pylance type checking in VS Code settings  |

---

---

## �️ MongoDB Integration (Conversation Memory)

### Setup

1. **Start MongoDB locally**

```bash
# Docker (recommended)
docker run -d -p 27017:27017 --name mongodb mongo:latest

# macOS (Homebrew)
brew services start mongodb-community

# Linux (Ubuntu/Debian)
sudo systemctl start mongod
```

1. **Configure MongoDB in config.py**

```python
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "langchain_qa"
COLLECTION_NAME = "conversations"
```

1. **Use Memory in app.py**

```python
from memory import create_memory

memory = create_memory(
    mongo_uri=MONGO_URI,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME
)
```

### Memory Features

- **Persistent Storage**: All conversations saved to MongoDB
- **Unique Conversation IDs**: Each session has a unique identifier
- **Timestamped Exchanges**: Every question-answer pair is timestamped
- **Conversation Retrieval**: Load past conversations by ID
- **Statistics**: View conversation metrics

### Memory Methods

```python
# Save interaction
memory.save_context(
    {"question": "What is LangChain?"},
    {"output": "LangChain is..."}
)

# Load recent history (last 5 exchanges)
memory.load_memory_variables({})

# Load conversation from MongoDB
memory.load_from_mongodb(conversation_id)

# Get all conversations
memory.get_all_conversations()

# Get conversation statistics
memory.get_conversation_stats()

# Clear memory
memory.clear()
```

---

## 📄 Document Conversion (doc.py)

Convert PDF, DOCX, and other document formats to text for use as knowledge bases.

### Usage

```python
from doc import convert_doc_to_txt

# Convert PDF to Markdown
convert_doc_to_txt("data/resume.pdf", "data/resume.md")
```

### Supported Formats

- PDF
- DOCX (Microsoft Word)
- PPTX (PowerPoint)
- Images (PNG, JPG)
- HTML
- Markdown
- And more with Docling

---

## �💾 Vectorstore Cache Management

The project automatically caches the FAISS vectorstore for faster startup times.

### Available Commands

```bash
# Check cache status
python utils.py status

# Rebuild the vectorstore from scratch
python utils.py rebuild

# Clear the cache (vectorstore will be recreated on next run)
python utils.py clear
```

### How It Works

1. **First Run**: Creates embeddings and saves to `vectorstore_cache/`
2. **Subsequent Runs**: Loads cached vectorstore (much faster)
3. **Cache Location**: `./vectorstore_cache/` (added to `.gitignore`)
4. **Cache Size**: Typically 10-50 MB depending on document size

### When to Rebuild

- After updating `data/data.txt` with new documents
- If embeddings settings change (chunk size, overlap, embedding model)
- If cache corruption is suspected

---

## 📚 Further Learning

To extend this project, you could:

- Add web interface (Flask/FastAPI)
- Use different LLMs (GPT-4, Claude, Llama, etc.)
- Support multiple vector stores (Pinecone, Weaviate, etc.)
- Add authentication for multiple users
- Implement conversation search and filtering
- Add document grouping and management
- Export conversations to PDF/JSON
- Add RAG evaluation metrics

---

## � MongoDB Integration (Conversation Memory)

### Setup

1. **Start MongoDB locally**

```bash
# Docker (recommended)
docker run -d -p 27017:27017 --name mongodb mongo:latest

# macOS
brew services start mongodb-community

# Linux
sudo systemctl start mongod
```

1. **Configure in config.py**

```python
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "langchain_qa"
COLLECTION_NAME = "conversations"
```

### Memory Features

- **Persistent Storage**: All conversations saved to MongoDB
- **Unique Conversation IDs**: Each session has identifier
- **Timestamped Exchanges**: Every Q&A is timestamped  
- **Conversation Retrieval**: Load past conversations by ID
- **Statistics**: View conversation metrics

---

## 📄 Document Conversion (doc.py)

Convert PDF, DOCX, and other formats to text for knowledge bases.

### Usage

```python
from doc import convert_doc_to_txt
convert_doc_to_txt("data/resume.pdf", "data/resume.md")
```

### Supported Formats

- PDF, DOCX, PPTX
- Images (PNG, JPG)
- HTML, Markdown
- And more with Docling

---

## �📄 License

This project is provided as a learning resource.

Happy coding!
