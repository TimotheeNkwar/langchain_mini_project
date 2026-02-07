from langchain_core.prompts import PromptTemplate  #type: ignore
from langchain_core.runnables import RunnablePassthrough  #type: ignore


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
        {"context": retriever, "question": RunnablePassthrough()}  #type: ignore
        | prompt
        | llm
    )

    return qa_chain