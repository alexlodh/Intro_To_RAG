import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def load_config():
    """Load configuration settings for the RAG system."""
    return {
        'file_paths': {
            'MARKDOWN_FILE': 'headlines.md',
            'QDRANT_STORAGE_PATH': 'qdrant_data'
        },
        'database': {
            'QDRANT_COLLECTION_NAME': 'pe_headlines'
        },
        'models': {
            'embedding_model': 'text-embedding-3-small',
            'chat_model': 'gpt-4o-mini'
        }
    }
