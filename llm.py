from langchain_ollama import OllamaLLM #type: ignore

def load_llm():
    
    return OllamaLLM(model="mistral", temperature=0.9)