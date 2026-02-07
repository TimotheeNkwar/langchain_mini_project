# 🎯 LangChain Mini Project - Q&A Assistant

A simple and clean Python project demonstrating how **LangChain** works with a practical Q&A assistant that answers questions based on uploaded documents.

## 📚 Project Overview

This project implements a **Question & Answer system** that:

1. Loads a text document (`data.txt`)
2. Splits it into manageable chunks
3. Creates embeddings (numerical representations) of the text
4. Stores embeddings in a vector database
5. Answers user questions based on the document content

It's a practical implementation of the classic LangChain workflow, perfect for learning how modern NLP systems work.

---

## 🛠️ How It Works

### Architecture Flow

```text
User Question
     ↓
Vector Store (FAISS) ← Document Processing
     ↓
Retriever (finds relevant chunks)
     ↓
LLM (generates answer from context)
     ↓
User gets Answer
```

### Step-by-Step Process

1. **Document Loading** (`embeddings.py`)
   - Reads `data/data.txt` using `TextLoader`
   - Loads raw text from the file

2. **Text Splitting** (`embeddings.py`)
   - Splits document into chunks of 500 characters
   - Overlaps chunks by 50 characters (maintains context)
   - Uses `RecursiveCharacterTextSplitter`

3. **Embeddings Creation** (`embeddings.py`)
   - Converts each text chunk to a numerical vector
   - Uses `HuggingFaceEmbeddings` (sentence-transformers model)
   - All processing is **local** (no API calls needed)

4. **Vector Store** (`embeddings.py`)
   - Stores embeddings in `FAISS` (Facebook AI Similarity Search)
   - Enables fast similarity search
   - Creates an in-memory index

5. **Retrieval Chain** (`chains.py`)
   - When you ask a question, it:
     - Converts your question to an embedding
     - Searches for similar chunks in the vector store
     - Retrieves the most relevant chunks
     - Feeds them as context to the LLM

6. **LLM Processing** (`chains.py`)
   - Uses `OllamaLLM` (local, no API keys required)
   - Receives the question + retrieved context
   - Generates a contextual answer

---

## 📁 Project Structure

```text
langchain_mini_project/
├── app.py              # Main entry point - interactive Q&A loop
├── llm.py              # Loads and configures the language model
├── embeddings.py       # Document loading, chunking & embeddings
├── chains.py           # Creates the retrieval chain
├── config.py           # Configuration & environment variables
├── data/
│   └── data.txt        # Source document for Q&A
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### File Details

| File | Purpose |
|------|---------|
| `app.py` | Interactive CLI interface - takes user questions in a loop |
| `llm.py` | Initializes OllamaLLM with specific model & parameters |
| `embeddings.py` | Handles document loading, splitting, and vector store creation |
| `chains.py` | Builds the LCEL (LangChain Expression Language) chain |
| `config.py` | Loads environment variables (for API keys if needed) |
| `data/data.txt` | Sample document about LangChain (your knowledge base) |

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

```bash
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
| `langchain-ollama` | Local LLM integration |
| `langchain-text-splitters` | Text chunking |
| `sentence-transformers` | Embedding model |
| `faiss-cpu` | Vector database |
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
- ✅ Runs entirely offline (after first model download)

---

## 📝 Notes

- The first run will download the embedding model (~50MB)
- Responses depend on the quality of documents in `data/data.txt`
- You can replace `data.txt` with any text file you want to analyze
- The vector store is created in-memory each time (not persisted)

---

## 🤝 Common Issues & Solutions

| Issue                   | Solution                                           |
| ----------------------- | -------------------------------------------------- |
| `ModuleNotFoundError`   | Run `uv pip install -r requirements.txt`           |
| Slow first run          | First request downloads embedding model, be patient |
| No response from LLM    | Ensure Ollama is running (if using Ollama locally) |
| Type warnings           | Disable Pylance type checking in VS Code settings  |

---

## 📚 Further Learning

To extend this project, you could:

- Add memory (remember previous conversations)
- Support multiple documents
- Add a web interface (Flask/FastAPI)
- Use different LLMs (GPT, Claude, etc.)
- Persist the vector store to disk
- Add document chat features

---

## 📄 License

This project is provided as a learning resource.

Happy coding! 🚀
