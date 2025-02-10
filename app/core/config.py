
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    PROJECT_NAME: str = "Financial Advisor Pro"
    VERSION: str = "1.0.0"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DB_DIR: Path = BASE_DIR / "vector_db"
    CACHE_DIR: Path = DB_DIR / "cache"
    
    # API Keys
    OPENAI_API_KEY: str
    NEWSAPI_KEY: str
    FINANCE_API_KEY: str
    
    # Database
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    
    # Model Settings
    MODEL_NAME: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0
    
    class Config:
        env_file = ".env"

settings = Settings()