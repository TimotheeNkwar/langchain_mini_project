

from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnablePassthrough  


def create_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever()
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""use the following context to answer the question.

Context: {context}

Question: {question}

Réponse:"""
    )
    
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}  
        | prompt
        | llm
    )

    return qa_chain