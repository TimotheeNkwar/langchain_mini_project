

from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def create_chain(llm, vectorstore, memory):
    retriever = vectorstore.as_retriever()
    
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""Use the following context and chat history to answer the question.

Chat History:
{chat_history}

Context: {context}

Question: {question}

Réponse:"""
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    qa_chain = (
        {
            "context": retriever | RunnableLambda(format_docs), 
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(lambda x: memory.load_memory_variables({}).get("chat_history", ""))
        }  
        | prompt
        | llm
    )

    return qa_chain