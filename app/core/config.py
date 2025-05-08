from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Doubt Solver API"
    
    # CORS settings
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]
    
    # MongoDB settings
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    
    # Gemini API settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 25 * 1024 * 1024  # 25 MB
    ALLOWED_EXTENSIONS: list[str] = [
        ".txt", ".pdf", ".doc", ".docx", 
        ".jpg", ".jpeg", ".png", 
        ".mp3", ".wav"
    ]
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "supersecretkey")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Validate required settings
if not settings.MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable not set")
if not settings.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")