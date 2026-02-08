"""Utility functions for managing FAISS vectorstore cache."""

import os
import shutil
from config import VECTORSTORE_SAVE_PATH
from embeddings import create_vectorstore


def clear_vectorstore_cache():
    """Delete the cached vectorstore to force recreation."""
    if os.path.exists(VECTORSTORE_SAVE_PATH):
        shutil.rmtree(VECTORSTORE_SAVE_PATH)
        print(f"Vectorstore cache deleted: {VECTORSTORE_SAVE_PATH}")
    else:
        print(f"No cache found: {VECTORSTORE_SAVE_PATH}")


def rebuild_vectorstore():
    """Rebuild the vectorstore from scratch."""
    clear_vectorstore_cache()
    print("\nRebuilding vectorstore...")
    vectorstore = create_vectorstore()
    print("Vectorstore rebuilt successfully!\n")
    return vectorstore


def cache_status():
    """Display the status of the vectorstore cache."""
    if os.path.exists(VECTORSTORE_SAVE_PATH):
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(VECTORSTORE_SAVE_PATH)
            for filename in filenames
        )
        size_mb = cache_size / (1024 * 1024)
        print(f"Cache found: {VECTORSTORE_SAVE_PATH}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Files: {sum(1 for _, _, files in os.walk(VECTORSTORE_SAVE_PATH) for _ in files)}")
    else:
        print(f"✗ No cache found: {VECTORSTORE_SAVE_PATH}\n  Vectorstore will be created on next startup")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "clear":
            clear_vectorstore_cache()
        elif command == "rebuild":
            rebuild_vectorstore()
        elif command == "status":
            cache_status()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: clear, rebuild, status")
    else:
        print("Cache Status:\n")
        cache_status()
