# 🎯 Mini‑projet simple : comprendre comment LangChain fonctionne

Objectif : comprendre **le fonctionnement de LangChain** avec un **petit projet clair**, une **architecture propre** et des **fichiers séparés**.

---

## 🧠 Idée du projet

Un **assistant Q&A sur un fichier texte** :

* Tu donnes un fichier `data.txt`
* LangChain le découpe
* Crée des embeddings
* Les stocke dans un vector store
* Un LLM répond aux questions en se basant sur le texte

C’est le workflow LangChain classique 🔥

---

## 🗂️ Structure du projet

```
langchain_mini_project/
│
├── app.py                # Point d’entrée
├── config.py             # Configuration globale
├── llm.py                # Chargement du LLM
├── embeddings.py         # Embeddings + Vector Store
├── chain.py              # Chaîne LangChain
├── data/
│   └── data.txt          # Texte source
└── requirements.txt
```

---

## ⚙️ 1. requirements.txt

```txt
langchain
langchain-community
langchain-openai
faiss-cpu
python-dotenv
```

---

## ⚙️ 2. config.py

Centralise la config (bonne pratique)

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

---

## 🤖 3. llm.py – Modèle local avec Ollama

On utilise un **LLM 100% local** via Ollama (ex: Mistral, Llama3).

```python
from langchain_community.llms import Ollama


def load_llm():
    return Ollama(
        model="mistral",
        temperature=0
    )
```

➡️ Avantages :

* Pas d’API Key
* Données locales
* Gratuit

---

## 🧬 4. embeddings.py – Texte → vecteurs

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def create_vectorstore():
    loader = TextLoader("data/data.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore
```

➡️ Ici tu vois le cœur **RAG** :
Texte → chunks → embeddings → base vectorielle

---

## 🔗 5. chain.py – La magie LangChain

```python
from langchain.chains import RetrievalQA


def create_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    return qa_chain
```

➡️ LangChain connecte :
**Question → recherche vectorielle → contexte → LLM → réponse**

---

## ▶️ 6. app.py – Point d’entrée

```python
from llm import load_llm
from embeddings import create_vectorstore
from chain import create_chain


def main():
    llm = load_llm()
    vectorstore = create_vectorstore()
    qa_chain = create_chain(llm, vectorstore)

    while True:
        question = input("❓ Ta question : ")
        if question.lower() == "exit":
            break

        answer = qa_chain.run(question)
        print("\n🤖 Réponse :", answer, "\n")


if __name__ == "__main__":
    main()
```

---

## 🧪 Exemple de data/data.txt

```txt
LangChain est une bibliothèque Python permettant de construire des applications basées sur des modèles de langage.
Elle facilite la création de chaînes combinant LLM, outils, mémoire et bases vectorielles.
```

---

## 🔍 Résumé mental (super important)

```
Texte
  ↓
Découpage
  ↓
Embeddings
  ↓
Vector Store (FAISS)
  ↓
Retriever
  ↓
LLM
  ↓
Réponse
```

---

## 🚀 Étapes suivantes (si tu veux aller plus loin)

* Ajouter **mémoire de conversation**
* Passer à **Streamlit**
* Utiliser un **LLM local (Ollama)**
* Sauvegarder FAISS sur disque

Si tu veux, je peux te faire :
👉 la **version Streamlit**
👉 une **version 100% open‑source**
👉 ou t’expliquer **chaque ligne lentement**
