import os
from dotenv import load_dotenv

def load_api_key():
    """Load API keys from environment variables."""
    # Try to load from .env file first
    load_dotenv()
    
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        # If not in environment, try to get from any existing .env
        from pathlib import Path
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        
        # If still not available, warn but continue
        if not os.getenv('OPENAI_API_KEY'):
            print("Warning: OPENAI_API_KEY not found in environment variables.")
            print("Please set your OpenAI API key:")
            print("export OPENAI_API_KEY='your_api_key_here'")


def load_secrets():
    """Load all secrets from environment variables."""
    # Load environment variables
    load_dotenv()
    
    # Try to load from .env file in project root
    from pathlib import Path
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    
    # Return dictionary of secrets
    secrets = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'LANGSMITH_API_KEY': os.getenv('LANGSMITH_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
    }
    
    # Warn if OpenAI API key is missing
    if not secrets['OPENAI_API_KEY']:
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your_api_key_here'")
    
    return secrets
