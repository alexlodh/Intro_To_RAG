#!/usr/bin/env python3
"""
Focused test for year filtering in self-query RAG.
"""

import os
import sys
sys.path.append('/Users/Sasha/Intro_To_RAG')

from utils.config_loader import PROJECT_ROOT, load_config
from utils.secrets_loader import load_api_key
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load configuration
load_api_key()
config = load_config()

QDRANT_STORAGE_PATH = os.path.join(PROJECT_ROOT, config['file_paths']['QDRANT_STORAGE_PATH'])
QDRANT_COLLECTION_NAME = "test_year_filtering"  # Use a test collection
EMBEDDING_MODEL = config['models']['embedding_model']

# Initialize components
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def create_test_documents():
    """Create test documents with different years for testing."""
    test_docs = [
        Document(
            page_content="OpenAI launches GPT-5 with advanced reasoning capabilities",
            metadata={"Date": "20250115", "Category": "New Product", "Year": 2025, "Vendor(s)": "OpenAI"}
        ),
        Document(
            page_content="Google announces new AI partnerships with major investment firms",
            metadata={"Date": "20240803", "Category": "Partnerships & Integrations", "Year": 2024, "Vendor(s)": "Google"}
        ),
        Document(
            page_content="Microsoft hires former Goldman Sachs partner as AI strategy director",
            metadata={"Date": "20240621", "Category": "Personnel / Office", "Year": 2024, "Vendor(s)": "Microsoft"}
        ),
        Document(
            page_content="Amazon Web Services unveils new private equity analytics platform",
            metadata={"Date": "20230912", "Category": "New Product", "Year": 2023, "Vendor(s)": "Amazon"}
        ),
        Document(
            page_content="BlackRock completes acquisition of AI-powered analytics startup",
            metadata={"Date": "20230408", "Category": "Deal Activity", "Year": 2023, "Vendor(s)": "BlackRock"}
        )
    ]
    return test_docs

def setup_test_collection():
    """Set up a test collection with sample documents."""
    client = QdrantClient(path=QDRANT_STORAGE_PATH)
    
    # Delete collection if it exists
    try:
        collections = [col.name for col in client.get_collections().collections]
        if QDRANT_COLLECTION_NAME in collections:
            print(f"Deleting existing test collection {QDRANT_COLLECTION_NAME}")
            client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
    except Exception as e:
        print(f"Error checking/deleting collection: {e}")
    
    # Create new collection
    print(f"Creating test collection {QDRANT_COLLECTION_NAME}")
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    
    # Initialize vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings
    )
    
    # Add test documents
    test_docs = create_test_documents()
    print(f"Adding {len(test_docs)} test documents...")
    vector_store.add_documents(test_docs)
    
    return vector_store

def create_test_self_query_retriever(vector_store):
    """Create self-query retriever with proper year filtering support."""
    metadata_field_info = [
        AttributeInfo(
            name="Category",
            description="Business category of the headline",
            type="string",
        ),
        AttributeInfo(
            name="Date",
            description="Publication date in YYYYMMDD format",
            type="string",
        ),
        AttributeInfo(
            name="Year",
            description="Publication year as an integer (e.g., 2025, 2024, 2023). Use this for year-based filtering and comparisons.",
            type="integer",
        ),
        AttributeInfo(
            name="Vendor(s)",
            description="Companies or vendors mentioned",
            type="string",
        ),
    ]
    
    document_content_description = "Technology and business headlines"
    
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        document_content_description,
        metadata_field_info,
        verbose=True,
        search_kwargs={"k": 5}
    )
    
    return retriever

def test_year_filtering():
    """Test year filtering functionality."""
    print("Setting up test collection...")
    vector_store = setup_test_collection()
    
    print("\nCreating self-query retriever...")
    retriever = create_test_self_query_retriever(vector_store)
    
    # Test queries with year filtering
    test_queries = [
        "What new products were released in 2025?",
        "Show me partnerships from 2024",
        "What deal activity happened in 2023?",
        "Tell me about personnel changes in 2024"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('-' * 60)
        
        try:
            results = retriever.get_relevant_documents(query)
            print(f"Found {len(results)} documents")
            
            # Show results with year information
            for i, doc in enumerate(results):
                year = doc.metadata.get('Year', 'N/A')
                date = doc.metadata.get('Date', 'N/A')
                category = doc.metadata.get('Category', 'N/A')
                print(f"  Result {i+1}: Year {year} ({date}) - {category}")
                print(f"    Content: {doc.page_content}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Cleanup
    client = QdrantClient(path=QDRANT_STORAGE_PATH)
    try:
        print(f"\nCleaning up test collection {QDRANT_COLLECTION_NAME}")
        client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
        print("Test collection deleted successfully")
    except Exception as e:
        print(f"Error cleaning up: {e}")

if __name__ == "__main__":
    test_year_filtering()
