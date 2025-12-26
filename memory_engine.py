import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from cryptography.fernet import Fernet
import hashlib
import os

from config import settings

class MemoryEngine:
    """Core memory system for EchoSoul with Pinecone + Gemini"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # Initialize Google Gemini
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.gemini = genai.GenerativeModel(settings.GEMINI_MODEL)
        else:
            self.gemini = None
        
        # Initialize embedding model (using Google's or local fallback)
        if settings.GOOGLE_API_KEY:
            # Using Google's embedding model
            self.embedding_model = "google"  # We'll use API for embeddings
        else:
            # Fallback to local model
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.encryption = Fernet(self._generate_encryption_key())
        
        # Initialize Pinecone
        if settings.PINECONE_API_KEY:
            pinecone.init(
                api_key=settings.PINECONE_API_KEY,
                environment=settings.PINECONE_ENVIRONMENT
            )
            
            # Create or connect to index
            index_name = f"{settings.PINECONE_INDEX_NAME}-{user_id}"
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=768,  # Adjust based on your embedding model
                    metric="cosine",
                    metadata_config={"indexed": ["type", "emotion", "timestamp", "is_vault", "user_id"]}
                )
            
            self.pinecone_index = pinecone.Index(index_name)
            self.use_pinecone = True
        else:
            # Fallback to ChromaDB
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            self.chroma_client = chromadb.Client(ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=f"{settings.DATA_DIR}/chroma/{user_id}"
            ))
            
            self.memory_collection = self.chroma_client.get_or_create_collection(
                name=f"memories_{user_id}",
                metadata={"hnsw:space": "cosine"}
            )
            self.use_pinecone = False
        
        # Ensure directories exist
        os.makedirs(f"{settings.MEMORIES_DIR}/{user_id}", exist_ok=True)
        os.makedirs(f"{settings.VAULT_DIR}/{user_id}", exist_ok=True)
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key from user ID and app secret"""
        key_material = f"{self.user_id}:{settings.ENCRYPTION_KEY}"
        return hashlib.sha256(key_material.encode()).digest()[:32]
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Google's API or local model"""
        if self.gemini and settings.GOOGLE_API_KEY:
            try:
                # Using Google's embedding API
                import google.generativeai as genai
                result = genai.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            except Exception as e:
                print(f"Google embedding failed: {e}, using local model")
                return self.encoder.encode(text).tolist()
        else:
            # Use local model
            return self.encoder.encode(text).tolist()
    
    def store_memory(self, memory: Dict[str, Any], is_vault: bool = False) -> str:
        """Store a new memory in Pinecone/Chroma and local storage"""
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
            embedding = self._get_embedding(memory_text)
            
            # Store metadata
            metadata = {
                "type": memory.get("type", "conversation"),
                "emotion": memory.get("emotion", "neutral"),
                "timestamp": memory["timestamp"],
                "is_vault": False,
                "user_id": self.user_id,
                "id": memory_id
            }
            
            if self.use_pinecone:
                # Store in Pinecone
                self.pinecone_index.upsert(
                    vectors=[(memory_id, embedding, metadata)],
                    namespace=self.user_id
                )
            else:
                # Store in ChromaDB
                self.memory_collection.add(
                    embeddings=[embedding],
                    documents=[json.dumps(memory)],
                    metadatas=[metadata],
                    ids=[memory_id]
                )
        
        return memory_id
    
    def retrieve_memories(self, query: str, n_results: int = 5, 
                         memory_type: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant memories based on query"""
        query_embedding = self._get_embedding(query)
        
        if self.use_pinecone:
            # Query Pinecone
            filter_dict = {
                "user_id": {"$eq": self.user_id},
                "is_vault": {"$eq": False}
            }
            
            if memory_type:
                filter_dict["type"] = {"$eq": memory_type}
            
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=n_results,
                include_metadata=True,
                filter=filter_dict,
                namespace=self.user_id
            )
            
            memories = []
            for match in results.matches:
                memory_id = match.id
                # Load full memory from local storage
                memory_path = f"{settings.MEMORIES_DIR}/{self.user_id}/{memory_id}.json"
                if os.path.exists(memory_path):
                    with open(memory_path, 'r') as f:
                        memory = json.load(f)
                        memory["similarity"] = match.score
                        memories.append(memory)
        else:
            # Query ChromaDB
            where_filter = {"is_vault": False}
            if memory_type:
                where_filter["type"] = memory_type
            
            results = self.memory_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter
            )
            
            memories = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                memory = json.loads(doc)
                memory["similarity"] = metadata.get("distance", 0)
                memories.append(memory)
        
        return memories
    
    def get_timeline(self, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> List[Dict]:
        """Get chronological timeline of memories from local storage"""
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
