import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from llm import load_llm
from embeddings import create_vectorstore
from chains import create_chain


def main():
    llm = load_llm()
    vectorstore = create_vectorstore()
    qa_chain = create_chain(llm, vectorstore)

    while True:
        question = input("Your question: ")
        if question.lower() == "exit":
            break

        answer = qa_chain.invoke(question)
        print("\nAnswer:", answer, "\n")


if __name__ == "__main__":
    main()