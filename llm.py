from langchain_ollama import OllamaLLM #type: ignore

def load_llm():
    
    """
    Loads a pre-trained language model (OllamaLLM) for use in the Q&A system.

    The model is the latest version of the "neural-chat" model, and the temperature is set to 0.9 which is a good balance between determinism and creativity.

    Returns:
        OllamaLLM: The pre-trained language model.
    """
    return OllamaLLM(model="neural-chat:latest", temperature=0.9)