import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load .env file from backend directory
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

def get_qdrant_client() -> QdrantClient:
    """
    Returns QdrantClient connected to Qdrant Cloud if credentials exist,
    otherwise initializes persistent local disk storage in ./qdrant_storage.
    """
    if QDRANT_URL and QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Fallback to local storage (works out of the box with zero cloud setup)
    storage_dir = Path(__file__).parent / "qdrant_storage"
    storage_dir.mkdir(exist_ok=True)
    return QdrantClient(path=str(storage_dir))
