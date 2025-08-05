# RAG Application: Part 1 - Build a Q&A app that answers questions about PE industry headlines
# Based on: https://python.langchain.com/docs/tutorials/rag/

import os
import hashlib
import sys
from pathlib import Path
from typing import List, Literal, Union
from typing_extensions import TypedDict, Annotated
from loguru import logger

# Add parent directory to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# LangChain components
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Qdrant vector store
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

# LangGraph for orchestration
from langgraph.graph import START, StateGraph

# Project utilities
from utils.config_loader import PROJECT_ROOT, load_config
from utils.secrets_loader import load_api_key

# LangSmith integration
try:
    from langsmith import Client as LangSmithClient
    langsmith_enabled = True
except ImportError:
    langsmith_enabled = False

# Setup logging with loguru
logger.remove()  # Remove default handler
logger.add(
    sink=lambda message: print(message, end=""),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO"
)

# Load configuration and secrets
try:
    load_api_key()
    config = load_config()
except Exception as e:
    logger.error(f"Failed to load configuration or API keys: {e}")
    logger.error("Please ensure you have proper API keys set up and config files available.")
    sys.exit(1)

# Verify OpenAI API key is available
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable is not set. Please set it before running the application.")
    sys.exit(1)

# Configuration
MARKDOWN_FILE = os.path.join(PROJECT_ROOT, config['file_paths']['MARKDOWN_FILE'])
QDRANT_STORAGE_PATH = os.path.join(PROJECT_ROOT, config['file_paths']['QDRANT_STORAGE_PATH'])
QDRANT_COLLECTION_NAME = config['database']['QDRANT_COLLECTION_NAME']
EMBEDDING_MODEL = config['models']['embedding_model']

# Verify required files exist
if not os.path.exists(MARKDOWN_FILE):
    logger.error(f"Required markdown file not found: {MARKDOWN_FILE}")
    logger.error("Please ensure the headlines.md file exists in the project root.")
    sys.exit(1)

# Initialize components with error handling
try:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    logger.info("Successfully initialized OpenAI components")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI components: {e}")
    logger.error("Please check your OpenAI API key and internet connection.")
    sys.exit(1)

def load_md_table_as_documents(file_path: str) -> List[Document]:
    """Load markdown table as LangChain documents with metadata."""
    documents = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[4:]  # Skip first 4 lines (header)
    except Exception as e:
        logger.error(f"Error reading markdown file: {e}")
        return []

    # Extract rows from markdown table, skipping the header separator
    rows = [line.strip().strip("|").split("|") for line in lines if line.strip().startswith("|") and '---' not in line]
    headers = ["Date", "Category", "Headline", "Vendor(s)"]
    
    for row in rows:
        row = [col.strip() for col in row]
        if len(row) != len(headers):
            continue  # skip malformed rows
        data = dict(zip(headers, row))
        headline = data.pop("Headline")
        
        # Add Year metadata for self-query filtering
        date_str = data.get("Date", "")
        if date_str and len(date_str) >= 4:
            try:
                year = int(date_str[:4])
                data["Year"] = year
            except ValueError:
                logger.warning(f"Could not extract year from date: {date_str}")
        
        documents.append(Document(page_content=headline, metadata=data))
    
    logger.info(f"Loaded {len(documents)} documents from {file_path}")
    return documents

def delete_collection() -> None:
    """Delete the existing collection for testing purposes."""
    client = None
    try:
        client = QdrantClient(path=QDRANT_STORAGE_PATH)
        collections = [col.name for col in client.get_collections().collections]
        if QDRANT_COLLECTION_NAME in collections:
            logger.info(f"Deleting collection {QDRANT_COLLECTION_NAME}")
            client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
            logger.success(f"Collection {QDRANT_COLLECTION_NAME} deleted successfully")
        else:
            logger.info(f"Collection {QDRANT_COLLECTION_NAME} does not exist")
    except RuntimeError as e:
        if "already accessed by another instance" in str(e):
            logger.error("Qdrant database is already in use by another process. Please stop other instances or use the server mode.")
            logger.info("You can use the VS Code task 'Start Qdrant Database' to run Qdrant in server mode.")
            raise
        else:
            logger.error(f"Error deleting collection: {e}")
            raise
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise
    finally:
        if client:
            try:
                client.close()
            except:
                pass

def setup_vector_store() -> QdrantVectorStore:
    """Initialize and populate the vector store with documents."""
    client = None
    try:
        client = QdrantClient(path=QDRANT_STORAGE_PATH)
    except RuntimeError as e:
        if "already accessed by another instance" in str(e):
            logger.error("Qdrant database is already in use by another process.")
            logger.info("Trying to use Qdrant server mode instead...")
            try:
                # Try to connect to Qdrant server if local file is locked
                client = QdrantClient(host="localhost", port=6333)
                logger.info("Successfully connected to Qdrant server")
            except Exception as server_error:
                logger.error(f"Could not connect to Qdrant server: {server_error}")
                logger.error("Please start Qdrant server using: docker-compose up -d qdrant")
                logger.error("Or stop other Qdrant instances and try again.")
                raise RuntimeError("Qdrant database unavailable") from e
        else:
            raise
    
    # Check if collection exists, create if it doesn't
    try:
        collections = [col.name for col in client.get_collections().collections]
        if QDRANT_COLLECTION_NAME not in collections:
            logger.info(f"Creating new collection {QDRANT_COLLECTION_NAME}")
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            logger.info(f"Collection {QDRANT_COLLECTION_NAME} created successfully")
        else:
            logger.info(f"Using existing collection {QDRANT_COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Error with collection setup: {e}")
        if client:
            try:
                client.close()
            except:
                pass
        raise
    
    # Initialize vector store
    try:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=embeddings
        )
        logger.info("Vector store initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        if client:
            try:
                client.close()
            except:
                pass
        raise
    
    # Load and add documents with deduplication
    docs = load_md_table_as_documents(MARKDOWN_FILE)
    if not docs:
        logger.error("No documents loaded. Cannot proceed.")
        raise ValueError("No documents loaded")
    
    # Check initial count
    initial_count = client.count(collection_name=QDRANT_COLLECTION_NAME).count
    logger.info(f"Initial documents in collection: {initial_count}")
    
    # Add hash to documents and check for duplicates
    for doc in docs:
        text = doc.page_content[:50]  # Use first 50 chars for hash
        doc.metadata["hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    # Get existing hashes from collection if collection is not empty
    existing_hashes = set()
    if initial_count > 0:
        try:
            # Scroll through all documents to get existing hashes
            scroll_result = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=10000,  # Large limit to get all docs
                with_payload=True
            )
            for point in scroll_result[0]:
                if point.payload and 'hash' in point.payload:
                    existing_hashes.add(point.payload['hash'])
            logger.info(f"Found {len(existing_hashes)} existing document hashes")
            
            # If no hashes found but documents exist, this might be an old collection
            if len(existing_hashes) == 0 and initial_count > 0:
                logger.warning(f"Found {initial_count} documents but no hash metadata. This appears to be an old collection.")
                logger.info("Will check for duplicates by comparing document content instead.")
        except Exception as e:
            logger.warning(f"Could not retrieve existing hashes: {e}")
    
    # Filter out documents that already exist
    new_docs = []
    duplicate_count = 0
    
    # If we have existing hashes, use hash-based deduplication
    if existing_hashes:
        for doc in docs:
            if doc.metadata["hash"] not in existing_hashes:
                new_docs.append(doc)
            else:
                duplicate_count += 1
    # If no hashes found but documents exist, check by content
    elif initial_count > 0:
        logger.info("Performing content-based duplicate detection...")
        # Get existing document content for comparison
        existing_content = set()
        try:
            scroll_result = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=10000,
                with_payload=True
            )
            for point in scroll_result[0]:
                if point.payload and 'page_content' in point.payload:
                    existing_content.add(point.payload['page_content'])
            logger.info(f"Retrieved {len(existing_content)} existing document contents for comparison")
        except Exception as e:
            logger.warning(f"Could not retrieve existing content: {e}")
            # If we can't retrieve existing content, assume all docs are new
            new_docs = docs
        
        # Check for duplicates by content
        for doc in docs:
            if doc.page_content not in existing_content:
                new_docs.append(doc)
            else:
                duplicate_count += 1
    else:
        # Empty collection, all docs are new
        new_docs = docs
    
    logger.info(f"Found {duplicate_count} duplicate documents (skipping)")
    logger.info(f"Preparing to add {len(new_docs)} new documents")
    
    if new_docs:
        # Documents already have proper Category metadata from table parsing
        # The metadata["Category"] field contains the actual business categories:
        # "Personnel / Office", "New Client", "Updated Product", "Research", 
        # "New Product", "Partnerships & Integrations", "Deal Activity", "Awards"
        logger.info(f"Documents will use existing Category metadata for filtering")
        
        # Add new documents to vector store
        logger.info(f"Adding {len(new_docs)} new documents to vector store...")
        try:
            vector_store.add_documents(new_docs)
            logger.success(f"Successfully added {len(new_docs)} new documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    else:
        logger.info("No new documents to add - all documents already exist in collection")
    
    # Verify final count
    final_count = client.count(collection_name=QDRANT_COLLECTION_NAME).count
    logger.info(f"Final documents in collection: {final_count}")
    logger.info(f"Net documents added: {final_count - initial_count}")
    
    # Verify count matches expected (initial + new documents added)
    expected_final_count = initial_count + len(new_docs)
    if final_count != expected_final_count:
        logger.warning(f"Count mismatch! Expected {expected_final_count}, got {final_count}")
    else:
        logger.success(f"Count verification passed: {final_count} documents total ({len(new_docs)} added)")
    
    return vector_store

def create_self_query_retriever(vector_store: QdrantVectorStore) -> SelfQueryRetriever:
    """Create a self-querying retriever with metadata field descriptions."""
    
    # Define metadata fields that can be filtered
    metadata_field_info = [
        AttributeInfo(
            name="Category",
            description="Business category of the headline. One of: Personnel / Office, New Client, Updated Product, Research, New Product, Partnerships & Integrations, Deal Activity, Awards",
            type="string",
        ),
        AttributeInfo(
            name="Date",
            description="Publication date in YYYYMMDD format (e.g., 20250725, 20240415). For year filtering, use format like 'YYYY' (e.g., '2025', '2024')",
            type="string",
        ),
        AttributeInfo(
            name="Year",
            description="Publication year as an integer extracted from the date (e.g., 2025, 2024, 2023). Use this for year-based filtering and comparisons.",
            type="integer",
        ),
        AttributeInfo(
            name="Vendor(s)",
            description="Companies or vendors mentioned in the headline",
            type="string",
        ),
    ]
    
    # Document content description for the retriever
    document_content_description = "Private Equity industry headlines containing news about personnel changes, new clients, product updates, research, partnerships, deals, and awards"
    
    # Create the self-query retriever
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        document_content_description,
        metadata_field_info,
        verbose=True,
        search_kwargs={"k": 5}
    )
    
    return retriever

# Define RAG application state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define Query Analysis schema for metadata filtering
class Search(TypedDict):
    """Search query with category filtering."""
    query: Annotated[str, "Search query to run."]
    category: Annotated[
        Literal["Personnel / Office", "New Client", "Updated Product", "Research", 
                "New Product", "Partnerships & Integrations", "Deal Activity", "Awards"],
        "Business category to filter by - Personnel / Office, New Client, Updated Product, Research, New Product, Partnerships & Integrations, Deal Activity, or Awards."
    ]
    year: Annotated[
        Union[int, None], 
        "Year to filter by (e.g., 2024, 2023). Extracted from dates in YYYYMMDD format. Use None for no year filtering. If there are no results from the requested year, or insufficient results, only use the small number of sources that match the year (or if needed, return 'no results' if there are no results from the requested year), rather than use inaccurate sources."
    ]


# Enhanced state for self-query RAG
class EnhancedState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    query: dict  # Stores analyzed query information

def retrieve(state: State) -> dict:
    """Retrieve relevant documents based on the question."""
    retrieved_docs = vector_store.similarity_search(
        state["question"],
        k=5  # Get top 5 most relevant documents
    )
    return {"context": retrieved_docs}

def generate(state: State) -> dict:
    """Generate answer using retrieved context and question."""
    # Load RAG prompt from LangChain hub
    try:
        prompt = hub.pull("rlm/rag-prompt")
    except Exception as e:
        # Fallback to custom prompt if hub is unavailable
        logger.warning(f"Could not load prompt from hub: {e}. Using custom prompt.")
        template = """You are an assistant for question-answering tasks about Private Equity industry headlines. 

Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""
        prompt = PromptTemplate.from_template(template)
    
    # Format context
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Generate response
    try:
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {"answer": "I apologize, but I encountered an error while generating the response."}

# Enhanced functions for self-query RAG
def enhanced_generate(state: EnhancedState) -> dict:
    """Generate answer for enhanced RAG with self-querying."""
    # Check if we have any context documents
    if not state["context"]:
        return {"answer": "I don't have any relevant information to answer your question based on the available data."}
    
    # Load RAG prompt
    try:
        prompt = hub.pull("rlm/rag-prompt")
    except Exception:
        template = """You are an assistant for question-answering tasks about Private Equity industry headlines.

Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""
        prompt = PromptTemplate.from_template(template)
    
    # Format context
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Generate response
    try:
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {"answer": "I apologize, but I encountered an error while generating the response."}

def enhanced_retrieve(state: EnhancedState) -> dict:
    """Retrieve relevant documents using self-query retriever."""
    try:
        # Get the structured query from self-query retriever
        retrieved_docs = self_query_retriever.invoke(state["question"])
        
        # Try to get the structured query for analysis
        try:
            structured_query = self_query_retriever.query_constructor.invoke({"query": state["question"]})
            query_info = {
                "query": structured_query.query if hasattr(structured_query, 'query') else state["question"],
                "category": "All"
            }
            
            # Extract category from filter if available
            if hasattr(structured_query, 'filter') and structured_query.filter:
                if hasattr(structured_query.filter, 'condition') and hasattr(structured_query.filter.condition, 'value'):
                    query_info["category"] = structured_query.filter.condition.value
                elif hasattr(structured_query.filter, 'comparator') and hasattr(structured_query.filter, 'attribute'):
                    query_info["category"] = f"{structured_query.filter.attribute}: {structured_query.filter.comparator}"
                    
        except Exception as query_analysis_error:
            logger.warning(f"Could not analyze query structure: {query_analysis_error}")
            query_info = {
                "query": state["question"],
                "category": "All"
            }
        
        return {
            "context": retrieved_docs,
            "query": query_info
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced_retrieve: {e}")
        # Fallback to basic retrieval
        try:
            retrieved_docs = vector_store.similarity_search(
                state["question"],
                k=5
            )
            return {
                "context": retrieved_docs,
                "query": {
                    "query": state["question"],
                    "category": "Fallback"
                }
            }
        except Exception as fallback_error:
            logger.error(f"Fallback retrieval also failed: {fallback_error}")
            return {
                "context": [],
                "query": {
                    "query": state["question"],
                    "category": "Error"
                }
            }

def create_basic_rag_chain():
    """Create basic RAG chain without query analysis."""
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()

def create_enhanced_rag_chain():
    """Create enhanced RAG chain with self-querying."""
    graph_builder = StateGraph(EnhancedState).add_sequence([enhanced_retrieve, enhanced_generate])
    graph_builder.add_edge(START, "enhanced_retrieve")
    return graph_builder.compile()

def interactive_query_loop(basic_rag, enhanced_rag):
    """Interactive loop for user queries."""
    print("\n" + "="*80)
    print("INTERACTIVE RAG QUERY SYSTEM")
    print("="*80)
    print("\nAvailable modes:")
    print("1. Basic RAG - Simple semantic search and generation")
    print("2. Enhanced RAG - Query analysis with category filtering")
    print("\nAvailable categories for Enhanced RAG:")
    print("- Personnel / Office")
    print("- New Client") 
    print("- Updated Product")
    print("- Research")
    print("- New Product")
    print("- Partnerships & Integrations")
    print("- Deal Activity")
    print("- Awards")
    print("\nCommands:")
    print("- Type 'quit' or 'exit' to stop")
    print("- Type 'mode basic' or 'mode enhanced' to switch modes")
    print("- Type 'help' to see this message again")
    
    current_mode = "enhanced"  # Default to enhanced mode
    
    while True:
        try:
            print(f"\n[{current_mode.upper()} MODE]")
            user_input = input("Enter your question: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
                
            if user_input.lower() == 'help':
                print("\nAvailable modes:")
                print("1. Basic RAG - Simple semantic search and generation")
                print("2. Enhanced RAG - Query analysis with category filtering")
                print("\nAvailable categories for Enhanced RAG:")
                print("- Personnel / Office, New Client, Updated Product, Research")
                print("- New Product, Partnerships & Integrations, Deal Activity, Awards")
                print("\nCommands: 'quit', 'exit', 'mode basic', 'mode enhanced', 'help'")
                continue
                
            if user_input.lower().startswith('mode '):
                new_mode = user_input.lower().replace('mode ', '').strip()
                if new_mode in ['basic', 'enhanced']:
                    current_mode = new_mode
                    print(f"Switched to {current_mode.upper()} mode")
                else:
                    print("Invalid mode. Use 'basic' or 'enhanced'")
                continue
            
            print("-" * 60)
            
            if current_mode == "basic":
                try:
                    result = basic_rag.invoke({"question": user_input})
                    print(f"Answer: {result['answer']}")
                    print(f"Sources: {len(result['context'])} documents retrieved")
                    for i, doc in enumerate(result['context'][:3]):  # Show first 3 sources
                        print(f"  Source {i+1}: {doc.metadata.get('Date', 'N/A')} - {doc.metadata.get('Category', 'N/A')}")
                        print(f"    \"{doc.page_content[:100]}...\"")
                except Exception as e:
                    logger.error(f"Error processing question '{user_input}': {e}")
                    print(f"Error: {e}")
                    
            else:  # enhanced mode
                try:
                    result = enhanced_rag.invoke({"question": user_input})
                    
                    # Safely access query information
                    query_info = result.get('query', {})
                    analyzed_query = query_info.get('query', user_input)
                    category = query_info.get('category', 'Unknown')
                    
                    print(f"Analyzed Query: {analyzed_query}")
                    print(f"Category Filter: {category}")
                    print(f"Answer: {result['answer']}")
                    print(f"Sources: {len(result['context'])} documents retrieved")
                    
                    for i, doc in enumerate(result['context'][:3]):  # Show first 3 sources
                        print(f"  Source {i+1}: {doc.metadata.get('Date', 'N/A')} - {doc.metadata.get('Category', 'N/A')}")
                        print(f"    \"{doc.page_content[:100]}...\"")
                        
                    if len(result['context']) == 0:
                        print(f"  Note: No documents found for the query.")
                        print(f"  Try switching to basic mode for broader search.")
                        
                except Exception as e:
                    logger.error(f"Error processing question '{user_input}': {e}")
                    print(f"Error: {e}")
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"An unexpected error occurred: {e}")

def run_demo(basic_rag, enhanced_rag):
    """Run the demo with predefined questions."""
    test_questions = [
        "How is AI being used by private market investors?",
        "What new product features are related to IPO analysis?",
        "Which companies are expanding their technology platforms?",
        "What are recent personnel changes in the industry?"
    ]
    
    logger.info("Running Demo Mode with predefined questions:")
    print("\n" + "="*80)
    print("DEMO MODE - BASIC RAG RESULTS")
    print("="*80)
    
    for question in test_questions[:2]:  # Test first 2 questions with basic RAG
        print(f"\nQuestion: {question}")
        print("-" * 60)
        
        try:
            result = basic_rag.invoke({"question": question})
            print(f"Answer: {result['answer']}")
            print(f"Sources: {len(result['context'])} documents retrieved")
            for i, doc in enumerate(result['context'][:2]):  # Show first 2 sources
                print(f"  Source {i+1}: {doc.metadata.get('Date', 'N/A')} - {doc.metadata.get('Category', 'N/A')}")
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            print(f"Error: {e}")
        
        print()
    
    print("\n" + "="*80)
    print("DEMO MODE - ENHANCED RAG RESULTS (WITH QUERY ANALYSIS)")
    print("="*80)
    
    for question in test_questions[2:]:  # Test remaining questions with enhanced RAG
        print(f"\nQuestion: {question}")
        print("-" * 60)
        
        try:
            result = enhanced_rag.invoke({"question": question})
            
            # Safely access query information
            query_info = result.get('query', {})
            analyzed_query = query_info.get('query', question)
            category = query_info.get('category', 'Unknown')
            
            print(f"Analyzed Query: {analyzed_query}")
            print(f"Category Filter: {category}")
            print(f"Answer: {result['answer']}")
            print(f"Sources: {len(result['context'])} documents retrieved")
            for i, doc in enumerate(result['context'][:2]):  # Show first 2 sources  
                print(f"  Source {i+1}: {doc.metadata.get('Date', 'N/A')} - {doc.metadata.get('Category', 'N/A')}")
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            print(f"Error: {e}")
        
        print()

def main():
    """Main function for the RAG application."""
    global vector_store, self_query_retriever
    
    try:
        # Setup vector store
        logger.info("Setting up vector store...")
        vector_store = setup_vector_store()
        
        # Create self-query retriever
        logger.info("Creating self-query retriever...")
        self_query_retriever = create_self_query_retriever(vector_store)
        
        # Create RAG chains
        logger.info("Initializing RAG chains...")
        basic_rag = create_basic_rag_chain()
        enhanced_rag = create_enhanced_rag_chain()
        
        logger.info("RAG system ready!")
        
        # Check command line arguments
        demo_mode = len(sys.argv) > 1 and sys.argv[1] == "--demo"
        
        if demo_mode:
            # Run demo mode
            run_demo(basic_rag, enhanced_rag)
        else:
            # Start interactive query loop
            interactive_query_loop(basic_rag, enhanced_rag)
        
        # LangSmith logging for session completion
        if langsmith_enabled:
            try:
                ls_client = LangSmithClient()
                session_type = "demo" if demo_mode else "interactive"
                ls_client.create_run(
                    name=f"RAG_{session_type.title()}_Session",
                    run_type="chain",
                    inputs={
                        "collection": QDRANT_COLLECTION_NAME,
                        "session_type": session_type,
                        "embedding_model": EMBEDDING_MODEL
                    },
                    outputs={"status": "completed"},
                    project_name="RAG_Demo"
                )
                logger.info("Logged session completion to LangSmith")
            except Exception as e:
                logger.warning(f"LangSmith logging failed: {e}")
        
        logger.info("RAG application session completed!")
        
    except Exception as e:
        logger.error(f"Fatal error in RAG application: {e}")
        raise

if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--delete-collection":
            delete_collection()
        elif sys.argv[1] == "--demo":
            main()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python 02_rag_pt1.py                 # Interactive mode (default)")
            print("  python 02_rag_pt1.py --demo          # Demo mode with predefined questions")
            print("  python 02_rag_pt1.py --delete-collection  # Delete existing collection")
            print("  python 02_rag_pt1.py --help          # Show this help")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help to see available options")
    else:
        main()



"""
=============================================================================
RAG APPLICATION - COMPREHENSIVE EXPLANATION
=============================================================================

This module implements a complete Retrieval Augmented Generation (RAG) system
for Private Equity industry headlines using LangChain and LangGraph, following
the official LangChain RAG tutorial architecture.

CORE COMPONENTS:
1. Document Loading & Processing
2. Vector Store Setup & Management
3. Basic RAG Chain Implementation
4. Enhanced RAG with Self-Query Retriever
5. LangGraph Orchestration
6. Deduplication & Error Handling
7. LangSmith Integration

=============================================================================
DETAILED WORKFLOW:
=============================================================================

1. DOCUMENT LOADING:
   - Reads PE headlines from markdown table (headlines.md)
   - Parses table structure: Date | Category | Headline | Vendor(s)
   - Creates LangChain Document objects with structured metadata
   - Implements robust error handling for file operations

2. VECTOR STORE MANAGEMENT:
   - Uses Qdrant vector database with persistent local file storage
   - Creates collection with 1536-dimensional vectors (OpenAI embedding size)
   - Implements intelligent collection management: preserves existing collections
   - Dual deduplication system: hash-based for new collections, content-based for legacy
   - Supports incremental document addition with smart duplicate detection
   - Backward compatible with existing collections lacking hash metadata
   - Provides comprehensive logging for all collection operations and count verification

3. BASIC RAG CHAIN:
   Architecture: Question -> Retrieve -> Generate -> Answer
   
   Components:
   - State: TypedDict containing question, context, and answer
   - Retrieve Function: Semantic search using OpenAI embeddings
   - Generate Function: LLM-based answer generation with context
   - Orchestration: LangGraph StateGraph for workflow management
   
   Process Flow:
   a) User provides a question
   b) System retrieves top 5 most relevant documents using similarity search
   c) Documents are formatted as context for the LLM
   d) GPT-4o-mini generates a concise answer using RAG prompt
   e) Returns structured response with answer and source documents

4. ENHANCED RAG WITH SELF-QUERY RETRIEVER:
   Architecture: Question -> Self-Query -> Generate -> Answer
   
   Advanced Features:
   - Self-Query Retriever: Automatically analyzes questions for metadata filtering
   - Metadata Filtering: Filters documents by Category, Date, and Vendor
   - Smart Retrieval: Uses natural language to structured query translation
   - Year-based Filtering: Automatically extracts and filters by year from dates
   
   Self-Query Process:
   a) User question is automatically analyzed by the self-query retriever
   b) Question is decomposed into search terms and metadata filters
   c) System retrieves documents matching both content and metadata criteria
   d) LLM generates targeted answer with filtered context
   
   Metadata Fields:
   - Category: Business categories (Personnel, New Product, etc.)
   - Date: Publication dates in YYYYMMDD format for year filtering
   - Vendor(s): Companies mentioned for entity-based filtering

5. LANGRAPH ORCHESTRATION:
   Benefits:
   - Automatic support for sync, async, and streaming invocations
   - Built-in state management and error handling
   - Streamlined deployment capabilities via LangGraph Platform
   - Automatic LangSmith tracing and observability
   - Easy addition of persistence and human-in-the-loop features
   
   Graph Structure:
   Basic RAG: START -> retrieve -> generate
   Enhanced RAG: START -> enhanced_retrieve -> enhanced_generate

6. PROMPT ENGINEERING:
   - Primary: Uses LangChain Hub RAG prompt (rlm/rag-prompt)
   - Fallback: Custom prompt optimized for PE industry content
   - Instructions: Concise answers (3 sentences max), acknowledge uncertainty
   - Context Integration: Seamless incorporation of retrieved documents

ARCHITECTURAL EVOLUTION:
Semantic Search: Query -> Embed -> Search -> Rank -> Return Documents
Basic RAG: Query -> Embed -> Search -> Rank -> Generate Answer
Enhanced RAG: Query -> Self-Query -> Search -> Filter -> Generate Answer

This implementation serves as a robust foundation for PE industry knowledge
management with intelligent data persistence and can be extended to support
various enterprise use cases including incremental data ingestion, legacy
system integration, and high-availability deployments.
"""


