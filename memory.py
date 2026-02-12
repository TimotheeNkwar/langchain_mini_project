#type: ignore
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId
from config import MONGO_URI, DB_NAME, COLLECTION_NAME


class ConversationMemory:
    """Conversation memory with MongoDB persistence."""
    
    def __init__(self, mongo_uri=MONGO_URI, db_name=DB_NAME, collection_name=COLLECTION_NAME):
        """
        Initializes a ConversationMemory instance.

        Args:
            mongo_uri (str): MongoDB connection URI. Defaults to MONGO_URI.
            db_name (str): Name of the MongoDB database. Defaults to DB_NAME.
            collection_name (str): Name of the MongoDB collection. Defaults to COLLECTION_NAME.
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        
        # Connect to MongoDB
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        # Create index for better performance
        self.collection.create_index("conversation_id")
        
        self.conversation_id = str(ObjectId())
        self.chat_history = []
    
    def save_context(self, input_dict, output_dict):
        """Save interaction to MongoDB and memory."""
        question = input_dict.get("question", "")
        answer = output_dict.get("output", "")
        
        # Save locally
        exchange = f"Q: {question}\nA: {answer}"
        self.chat_history.append(exchange)
        
        # Save to MongoDB
        doc = {
            "conversation_id": self.conversation_id,
            "timestamp": datetime.utcnow(),
            "question": question,
            "answer": answer
        }
        self.collection.insert_one(doc)
    
    def load_memory_variables(self, inputs):
        """Load memory variables from local cache (last 5)."""
        history_text = "\n\n".join(self.chat_history[-5:]) if self.chat_history else ""
        return {"chat_history": history_text}
    
    def load_from_mongodb(self, conversation_id=None):
        """Load conversation history from MongoDB."""
        cid = conversation_id or self.conversation_id
        documents = list(self.collection.find(
            {"conversation_id": cid}
        ).sort("timestamp", 1))
        
        self.chat_history = []
        for doc in documents:
            exchange = f"Q: {doc['question']}\nA: {doc['answer']}"
            self.chat_history.append(exchange)
        
        return self.chat_history
    
    def clear(self):
        """Clear memory."""
        self.chat_history = []
        # Optionally delete from MongoDB
        # self.collection.delete_many({"conversation_id": self.conversation_id})
    
    def print_history(self):
        """Affiche l'historique pour déboguer."""
        print("\n=== Memory History (MongoDB) ===")
        for i, exchange in enumerate(self.chat_history, 1):
            print(f"\n[{i}] {exchange}")
        print(f"\nTotal: {len(self.chat_history)} interactions")
        print(f"Conversation ID: {self.conversation_id}")
        print("=======================\n")
    
    def get_all_conversations(self):
        """Get list of all conversations."""
        conversations = self.collection.distinct("conversation_id")
        return conversations
    
    def get_conversation_stats(self, conversation_id=None):
        """Get stats for a conversation."""
        cid = conversation_id or self.conversation_id
        count = self.collection.count_documents({"conversation_id": cid})
        return {"conversation_id": cid, "total_exchanges": count}


def create_memory(mongo_uri="mongodb://localhost:27017/"):
    """Create a conversation memory buffer with MongoDB."""
    memory = ConversationMemory(mongo_uri=mongo_uri)
    return memory


def get_memory_variables(memory):
    """Get the current memory as a formatted string."""
    return memory.load_memory_variables({})

