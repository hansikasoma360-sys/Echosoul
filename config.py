
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
    GEMINI_MODEL: str = "gemini-pro"
    
    # Memory Settings (using ChromaDB instead of Pinecone)
    VECTOR_DB_TYPE: str = "chroma"
    USE_GEMINI: bool = True
    
    # Emotion Model
    EMOTION_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"
    
    # Storage
    DATA_DIR: str = "data"
    MEMORIES_DIR: str = "data/memories"
    VAULT_DIR: str = "data/vault"
    USERS_DIR: str = "data/users"
    
    # Encryption
    ENCRYPTION_KEY: str = "your-secret-key-here-change-in-production"
    
    # Embeddings (using local model)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"

settings = Settings()
