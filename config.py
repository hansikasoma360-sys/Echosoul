import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "EchoSoul"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # AI Settings
    OPENAI_API_KEY: Optional[str] = None
    USE_LOCAL_LLM: bool = True  # Use local Llama models
    LOCAL_LLM_PATH: str = "TheBloke/Llama-2-7B-Chat-GGUF"
    
    # Memory Settings
    VECTOR_DB_TYPE: str = "chroma"  # chroma or pinecone
    PINE_CONE_API_KEY: Optional[str] = None
    PINE_CONE_ENV: Optional[str] = None
    
    # Emotion Model
    EMOTION_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"
    
    # Storage
    DATA_DIR: str = "data"
    MEMORIES_DIR: str = "data/memories"
    VAULT_DIR: str = "data/vault"
    USERS_DIR: str = "data/users"
    
    # Encryption
    ENCRYPTION_KEY: str = "your-secret-key-here-change-in-production"
    
    class Config:
        env_file = ".env"

settings = Settings()
