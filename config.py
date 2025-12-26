
import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "EchoSoul"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Google AI Studio (Gemini) Settings
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-pro"  # or "gemini-pro-vision" for multimodal
    
    # Pinecone Settings
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = "us-west1-gcp"  # or your environment
    PINECONE_INDEX_NAME: str = "echosoul-memories"
    
    # Memory Settings
    VECTOR_DB_TYPE: str = "pinecone"  # Changed to pinecone
    USE_GEMINI: bool = True  # Use Gemini instead of local LLM
    
    # Emotion Model
    EMOTION_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"
    
    # Storage
    DATA_DIR: str = "data"
    MEMORIES_DIR: str = "data/memories"
    VAULT_DIR: str = "data/vault"
    USERS_DIR: str = "data/users"
    
    # Encryption
    ENCRYPTION_KEY: str = "your-secret-key-here-change-in-production"
    
    # Embeddings
    EMBEDDING_MODEL: str = "models/embedding-001"  # Google's embedding model
    
    class Config:
        env_file = ".env"

settings = Settings()
