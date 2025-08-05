#!/usr/bin/env python3
"""
Simple demo of tool-based conversational RAG for testing.
This demonstrates the core functionality without complex LangGraph setup.
"""

import os
import sys
sys.path.append('/Users/Sasha/Intro_To_RAG')

from utils.config_loader import PROJECT_ROOT, load_config
from utils.secrets_loader import load_api_key
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
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

# Global vector store
vector_store = None

@tool
def search_headlines(query: str, category: str = None, year: int = None) -> str:
    """
    Search for relevant headlines based on query, with optional filtering.
    
    Args:
        query: The search query for headlines
        category: Optional business category filter 
        year: Optional year filter (e.g., 2025, 2024, 2023)
    
    Returns:
        String summary of relevant documents found
    """
    try:
        # Use basic semantic search for simplicity
        retrieved_docs = vector_store.similarity_search(query, k=5)
        
        if not retrieved_docs:
            return f"No headlines found for query: {query}"
        
        results = []
        for i, doc in enumerate(retrieved_docs[:5], 1):
            date = doc.metadata.get('Date', 'N/A')
            doc_category = doc.metadata.get('Category', 'N/A')
            doc_year = doc.metadata.get('Year', 'N/A')
            content = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            
            # Apply filters if specified
            if category and doc_category != category:
                continue
            if year and doc_year != year:
                continue
                
            results.append(f"{i}. Date: {date} ({doc_year}) | Category: {doc_category}\n   {content}")
        
        if not results:
            return f"No headlines found matching filters. Query: {query}, Category: {category}, Year: {year}"
        
        return f"Found {len(results)} headlines:\n\n" + "\n\n".join(results)
        
    except Exception as e:
        return f"Error searching headlines: {e}"

@tool
def get_categories() -> str:
    """Get available business categories."""
    categories = [
        "Personnel / Office", "New Client", "Updated Product", "Research",
        "New Product", "Partnerships & Integrations", "Deal Activity", "Awards"
    ]
    return "Available categories:\n" + "\n".join([f"- {cat}" for cat in categories])

@tool
def get_available_years() -> str:
    """Get available years in the dataset."""
    return "Available years: 2025, 2024, 2023"

# Tools list
tools = [search_headlines, get_categories, get_available_years]

def setup_vector_store():
    """Initialize vector store connection."""
    global vector_store
    
    client = QdrantClient(path=QDRANT_STORAGE_PATH)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings
    )
    print(f"Connected to vector store with collection: {QDRANT_COLLECTION_NAME}")

def simple_tool_chat():
    """Simple chat interface that can use tools."""
    setup_vector_store()
    
    print("\n" + "="*70)
    print("SIMPLE TOOL-BASED RAG DEMO")
    print("="*70)
    print("\nAvailable commands:")
    print("- Ask questions about PE headlines")
    print("- Try: 'What categories are available?'")
    print("- Try: 'Show me AI developments from 2025'")
    print("- Try: 'What personnel changes happened recently?'")
    print("- Type 'quit' to exit")
    
    # Bind tools to the model
    model_with_tools = llm.bind_tools(tools)
    
    while True:
        try:
            user_input = input("\nAsk a question: ").strip()
            
            if not user_input or user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            print("-" * 50)
            
            # Create messages
            system_msg = SystemMessage(content="""You are an assistant for Private Equity industry headlines. 
You have access to tools to search headlines and get information about categories and years.
Use the tools to find relevant information before answering questions.""")
            
            human_msg = HumanMessage(content=user_input)
            
            # Get model response with potential tool calls
            response = model_with_tools.invoke([system_msg, human_msg])
            
            # Check if model wants to use tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print("Using tools to search for information...")
                
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    # Find and execute the tool
                    for tool in tools:
                        if tool.name == tool_name:
                            try:
                                result = tool.invoke(tool_args)
                                tool_results.append(f"{tool_name}: {result}")
                                print(f"✓ Used {tool_name}")
                            except Exception as e:
                                tool_results.append(f"{tool_name}: Error - {e}")
                                print(f"✗ Error with {tool_name}: {e}")
                            break
                
                # Generate final response with tool results
                final_prompt = f"""Based on the tool results below, provide a comprehensive answer to: {user_input}

Tool Results:
{chr(10).join(tool_results)}

Please provide a clear, informative answer:"""
                
                final_response = llm.invoke([HumanMessage(content=final_prompt)])
                print(f"\nAnswer: {final_response.content}")
                
            else:
                # No tools needed, direct response
                print(f"Answer: {response.content}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    simple_tool_chat()
