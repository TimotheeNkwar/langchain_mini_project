import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from llm import load_llm
from embeddings import get_or_create_vectorstore
from chains import create_chain
from memory import create_memory
from config import MONGO_URI, DB_NAME, COLLECTION_NAME


def main():
    print("Starting Q&A Assistant...\n")
    llm = load_llm()
    vectorstore = get_or_create_vectorstore()
    memory = create_memory(mongo_uri=MONGO_URI)
    qa_chain = create_chain(llm, vectorstore, memory)

    while True:
        question = input("Your question: ")
        if question.lower() == "exit":
            break

        answer = qa_chain.invoke(question)
        print("\nAnswer:", answer, "\n")
        
        # Save to memory
        memory.save_context({"question": question}, {"output": answer})
        
        # Display memory for testing
        #memory.print_history()


if __name__ == "__main__":
    main()
