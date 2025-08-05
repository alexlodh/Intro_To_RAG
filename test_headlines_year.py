#!/usr/bin/env python3
"""
Quick test of year filtering with actual headlines data.
"""

import os
import sys
sys.path.append('/Users/Sasha/Intro_To_RAG')

from utils.config_loader import PROJECT_ROOT, load_config
from utils.secrets_loader import load_api_key
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Load configuration
load_api_key()
config = load_config()

QDRANT_STORAGE_PATH = os.path.join(PROJECT_ROOT, config['file_paths']['QDRANT_STORAGE_PATH'])
QDRANT_COLLECTION_NAME = config['database']['QDRANT_COLLECTION_NAME']
EMBEDDING_MODEL = config['models']['embedding_model']

# Initialize components
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def test_headlines_year_filtering():
    """Test year filtering with the headlines collection."""
    # Check if collection exists and has data
    client = QdrantClient(path=QDRANT_STORAGE_PATH)
    try:
        collection_info = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        count = client.count(collection_name=QDRANT_COLLECTION_NAME).count
        print(f"Collection {QDRANT_COLLECTION_NAME} exists with {count} documents")
        
        if count == 0:
            print("Collection is empty. Please run the main script first to load documents.")
            return
            
    except Exception as e:
        print(f"Collection doesn't exist or error accessing it: {e}")
        print("Please run the main script first to create and load the collection.")
        return
    
    # Set up vector store and self-query retriever
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings
    )
    
    # Create self-query retriever with Year metadata
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
    
    document_content_description = "Private Equity industry headlines"
    
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        document_content_description,
        metadata_field_info,
        verbose=True,
        search_kwargs={"k": 5}
    )
    
    # Test year-based queries
    test_queries = [
        "What new products were released in 2025?",
        "Show me personnel changes from 2024", 
        "What AI developments happened in 2025?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('-' * 70)
        
        try:
            results = retriever.invoke(query)
            print(f"Found {len(results)} documents")
            
            # Check if results are filtered by year
            years_found = set()
            for i, doc in enumerate(results[:3]):  # Show first 3
                year = doc.metadata.get('Year', 'N/A')
                date = doc.metadata.get('Date', 'N/A')
                category = doc.metadata.get('Category', 'N/A')
                years_found.add(year)
                print(f"  Result {i+1}: Year {year} ({date}) - {category}")
                print(f"    Content: {doc.page_content[:100]}...")
            
            print(f"Years found in results: {sorted(years_found)}")
                
        except Exception as e:
            print(f"Error: {e}")
            # Fallback to basic search
            try:
                results = vector_store.similarity_search(query, k=3)
                print(f"Fallback search found {len(results)} documents")
                for i, doc in enumerate(results):
                    year = doc.metadata.get('Year', 'N/A') 
                    date = doc.metadata.get('Date', 'N/A')
                    print(f"  Fallback {i+1}: Year {year} ({date})")
            except Exception as fallback_error:
                print(f"Fallback search also failed: {fallback_error}")

if __name__ == "__main__":
    test_headlines_year_filtering()
