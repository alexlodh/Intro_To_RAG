#!/usr/bin/env python3
"""
Test Self-Query RAG implementation with proper year filtering.
"""

import os
import sys
from typing import List
from typing_extensions import TypedDict

# Add project root to path
sys.path.append('.')

from utils.config_loader import PROJECT_ROOT, load_config
from utils.secrets_loader import load_api_key
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain import hub
from qdrant_client import QdrantClient
from langgraph.graph import START, StateGraph

# Load configuration
load_api_key()
config = load_config()

# Configuration
QDRANT_STORAGE_PATH = os.path.join(PROJECT_ROOT, config['file_paths']['QDRANT_STORAGE_PATH'])
QDRANT_COLLECTION_NAME = config['database']['QDRANT_COLLECTION_NAME']
EMBEDDING_MODEL = config['models']['embedding_model']

# Global variables
vector_store = None
self_query_retriever = None

# Initialize components
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# RAG State
class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str

def setup_self_query_retriever():
    """Set up the self-query retriever with proper metadata configuration."""
    global vector_store, self_query_retriever
    
    # Initialize vector store
    client = QdrantClient(path=QDRANT_STORAGE_PATH)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings
    )
    
    # Define metadata fields for self-querying
    metadata_field_info = [
        AttributeInfo(
            name="Category",
            description="Business category of the headline. One of: Personnel / Office, New Client, Updated Product, Research, New Product, Partnerships & Integrations, Deal Activity, Awards",
            type="string",
        ),
        AttributeInfo(
            name="Date",
            description="Publication date in YYYYMMDD format. For 2025 data, dates start with '2025'. For 2024 data, dates start with '2024'.",
            type="string",
        ),
    ]
    
    document_content_description = "Private Equity industry headlines containing news about personnel changes, new clients, product updates, research, partnerships, deals, and awards"
    
    # Create self-query retriever
    self_query_retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        document_content_description,
        metadata_field_info,
        verbose=True,
        search_kwargs={"k": 5}
    )
    
    print("Self-query retriever setup complete!")

def self_query_retrieve(state: RAGState) -> dict:
    """Retrieve documents using self-query retriever."""
    try:
        retrieved_docs = self_query_retriever.invoke(state["question"])
        print(f"Self-query retriever found {len(retrieved_docs)} documents")
        
        # Show what was retrieved
        if retrieved_docs:
            print("Retrieved documents:")
            for i, doc in enumerate(retrieved_docs):
                date = doc.metadata.get("Date", "N/A")
                category = doc.metadata.get("Category", "N/A")
                print(f"  {i+1}. Date: {date}, Category: {category}")
                print(f"     Content: {doc.page_content[:80]}...")
        
        return {"context": retrieved_docs}
    except Exception as e:
        print(f"Self-query retriever failed: {e}")
        # Fallback to basic search
        retrieved_docs = vector_store.similarity_search(state["question"], k=5)
        return {"context": retrieved_docs}

def generate_answer(state: RAGState) -> dict:
    """Generate answer using retrieved context."""
    if not state["context"]:
        return {"answer": "I don't have any relevant information to answer your question."}
    
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
        print(f"Error generating response: {e}")
        return {"answer": "I encountered an error while generating the response."}

def create_self_query_rag():
    """Create RAG chain with self-querying."""
    graph_builder = StateGraph(RAGState).add_sequence([self_query_retrieve, generate_answer])
    graph_builder.add_edge(START, "self_query_retrieve")
    return graph_builder.compile()

def test_self_query_rag():
    """Test the self-query RAG system."""
    print("Setting up self-query RAG system...")
    setup_self_query_retriever()
    
    rag_chain = create_self_query_rag()
    
    # Test questions with year filtering expectations
    test_questions = [
        "What new products were released in 2025?",
        "Show me personnel changes from 2024",
        "What partnerships happened in 2025?",
        "Tell me about Updated Product news from 2025",
        "Find Deal Activity from 2024"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('-' * 60)
        
        try:
            result = rag_chain.invoke({"question": question})
            print(f"Answer: {result['answer']}")
            print(f"Sources: {len(result['context'])} documents")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_self_query_rag()
