import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import numpy as np
from cryptography.fernet import Fernet
import hashlib
import os

from config import settings

class MemoryEngine:
    """Core memory system for EchoSoul"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encryption = Fernet(self._generate_encryption_key())
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(ChromaSettings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=f"{settings.DATA_DIR}/chroma/{user_id}"
        ))
        
        # Create or get collections
        self.memory_collection = self.chroma_client.get_or_create_collection(
            name=f"memories_{user_id}",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.personality_collection = self.chroma_client.get_or_create_collection(
            name=f"personality_{user_id}",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Ensure directories exist
        os.makedirs(f"{settings.MEMORIES_DIR}/{user_id}", exist_ok=True)
        os.makedirs(f"{settings.VAULT_DIR}/{user_id}", exist_ok=True)
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key from user ID and app secret"""
        key_material = f"{self.user_id}:{settings.ENCRYPTION_KEY}"
        return hashlib.sha256(key_material.encode()).digest()[:32]
    
    def store_memory(self, memory: Dict[str, Any], is_vault: bool = False) -> str:
        """Store a new memory"""
        memory_id = str(uuid.uuid4())
        memory["id"] = memory_id
        memory["timestamp"] = datetime.now().isoformat()
        memory["user_id"] = self.user_id
        
        # Encrypt if vault memory
        if is_vault:
            memory["encrypted"] = True
            memory_data = json.dumps(memory).encode()
            encrypted_data = self.encryption.encrypt(memory_data)
            
            vault_path = f"{settings.VAULT_DIR}/{self.user_id}/{memory_id}.enc"
            with open(vault_path, 'wb') as f:
                f.write(encrypted_data)
        else:
            # Store in regular memory
            memory["encrypted"] = False
            memory_path = f"{settings.MEMORIES_DIR}/{self.user_id}/{memory_id}.json"
            with open(memory_path, 'w') as f:
                json.dump(memory, f, indent=2)
            
            # Create embedding and store in vector DB
            memory_text = f"{memory.get('title', '')} {memory.get('content', '')} {memory.get('emotion', '')}"
            embedding = self.encoder.encode(memory_text).tolist()
            
            self.memory_collection.add(
                embeddings=[embedding],
                documents=[json.dumps(memory)],
                metadatas=[{
                    "type": memory.get("type", "conversation"),
                    "emotion": memory.get("emotion", "neutral"),
                    "timestamp": memory["timestamp"],
                    "is_vault": False
                }],
                ids=[memory_id]
            )
        
        return memory_id
    
    def retrieve_memories(self, query: str, n_results: int = 5, 
                         memory_type: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant memories based on query"""
        query_embedding = self.encoder.encode(query).tolist()
        
        results = self.memory_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"is_vault": False} if not memory_type else 
                  {"is_vault": False, "type": memory_type}
        )
        
        memories = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            memory = json.loads(doc)
            memory["similarity"] = metadata.get("distance", 0)
            memories.append(memory)
        
        return memories
    
    def get_timeline(self, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> List[Dict]:
        """Get chronological timeline of memories"""
        memories = []
        memory_dir = f"{settings.MEMORIES_DIR}/{self.user_id}"
        
        if os.path.exists(memory_dir):
            for filename in os.listdir(memory_dir):
                if filename.endswith('.json'):
                    with open(f"{memory_dir}/{filename}", 'r') as f:
                        memory = json.load(f)
                        
                        # Filter by date range
                        mem_date = datetime.fromisoformat(memory["timestamp"])
                        if start_date:
                            start = datetime.fromisoformat(start_date)
                            if mem_date < start:
                                continue
                        if end_date:
                            end = datetime.fromisoformat(end_date)
                            if mem_date > end:
                                continue
                        
                        memories.append(memory)
        
        # Sort by timestamp
        memories.sort(key=lambda x: x["timestamp"])
        return memories
    
    def update_personality_trait(self, trait: str, value: Any):
        """Update personality traits"""
        personality_path = f"{settings.USERS_DIR}/{self.user_id}/personality.json"
        os.makedirs(os.path.dirname(personality_path), exist_ok=True)
        
        if os.path.exists(personality_path):
            with open(personality_path, 'r') as f:
                personality = json.load(f)
        else:
            personality = {}
        
        personality[trait] = value
        
        with open(personality_path, 'w') as f:
            json.dump(personality, f, indent=2)
        
        # Also store in vector DB for semantic search
        embedding = self.encoder.encode(f"{trait}: {value}").tolist()
        self.personality_collection.upsert(
            embeddings=[embedding],
            documents=[json.dumps({trait: value})],
            metadatas=[{"trait": trait}],
            ids=[str(uuid.uuid4())]
        )
    
    def get_vault_memories(self) -> List[Dict]:
        """Retrieve all vault memories (encrypted)"""
        vault_memories = []
        vault_dir = f"{settings.VAULT_DIR}/{self.user_id}"
        
        if os.path.exists(vault_dir):
            for filename in os.listdir(vault_dir):
                if filename.endswith('.enc'):
                    try:
                        with open(f"{vault_dir}/{filename}", 'rb') as f:
                            encrypted_data = f.read()
                        
                        decrypted_data = self.encryption.decrypt(encrypted_data)
                        memory = json.loads(decrypted_data.decode())
                        vault_memories.append(memory)
                    except:
                        continue
        
        return vault_memories
