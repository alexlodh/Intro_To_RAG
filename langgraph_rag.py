"""
LangGraph-based RAG implementation for PE Headlines Analysis System.
This integrates with the existing Qdrant vector store and headline processing system.
"""

import os
from typing import List, TypedDict
from langchain import hub
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# Initialize components
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant_client = QdrantClient(path="./qdrant_data")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

# Define state for the RAG application
class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve_headlines(state: RAGState) -> dict:
    """
    Retrieve relevant headlines from Qdrant vector store based on the question.
    """
    try:
        # Convert question to embedding
        question_embedding = embedding_model.encode(state["question"]).tolist()
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name="pe_headlines",
            query_vector=question_embedding,
            limit=5,
            with_payload=True
        )
        
        # Convert search results to Document objects
        retrieved_docs = []
        for result in search_results:
            payload = result.payload
            # Create content from headline and metadata
            content = f"""Date: {payload.get('date', 'Unknown')}
Category: {payload.get('category', 'Unknown')}
Headline: {payload.get('headline', '')}
Vendors: {', '.join(payload.get('vendors', []))}
Score: {result.score:.3f}"""
            
            doc = Document(
                page_content=content,
                metadata=payload
            )
            retrieved_docs.append(doc)
        
        console.print(f"[green]✓[/green] Retrieved {len(retrieved_docs)} relevant headlines")
        return {"context": retrieved_docs}
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error retrieving headlines: {str(e)}")
        return {"context": []}


def generate_answer(state: RAGState) -> dict:
    """
    Generate an answer using the retrieved context and OpenAI GPT.
    """
    try:
        # Get the RAG prompt from LangChain Hub
        prompt = hub.pull("rlm/rag-prompt")
        
        # Combine context documents
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        # Create system context specific to PE headlines
        pe_context = f"""You are an expert analyst specializing in Private Equity industry news and trends. 
You have access to recent PE headlines and should provide insights based on this data.

Context from PE Headlines:
{docs_content}"""
        
        # Generate response
        messages = prompt.invoke({
            "question": state["question"], 
            "context": pe_context
        })
        response = llm.invoke(messages)
        
        console.print("[green]✓[/green] Generated AI response")
        return {"answer": response.content}
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error generating answer: {str(e)}")
        return {"answer": f"Error generating response: {str(e)}"}


def create_rag_graph():
    """
    Create and compile the LangGraph RAG application.
    """
    # Build the graph with retrieve -> generate sequence
    graph_builder = StateGraph(RAGState)
    graph_builder.add_sequence([retrieve_headlines, generate_answer])
    graph_builder.add_edge(START, "retrieve_headlines")
    
    return graph_builder.compile()


@click.command()
@click.option('--question', '-q', help='Question to ask about PE headlines')
@click.option('--interactive', '-i', is_flag=True, help='Run in interactive mode')
def main(question: str, interactive: bool):
    """
    LangGraph-based RAG system for PE Headlines Analysis.
    """
    console.print(Panel.fit(
        "[bold blue]PE Headlines LangGraph RAG System[/bold blue]",
        subtitle="Powered by LangGraph + Qdrant + OpenAI"
    ))
    
    # Check if Qdrant collection exists
    try:
        collections = qdrant_client.get_collections()
        if not any(col.name == "pe_headlines" for col in collections.collections):
            console.print("[red]✗[/red] Collection 'pe_headlines' not found. Please run:")
            console.print("  [cyan]python main.py embed --file-path headlines.md[/cyan]")
            return
    except Exception as e:
        console.print(f"[red]✗[/red] Cannot connect to Qdrant: {str(e)}")
        console.print("Please ensure Qdrant is running:")
        console.print("  [cyan]docker-compose up -d qdrant[/cyan]")
        return
    
    # Create the RAG graph
    rag_graph = create_rag_graph()
    
    if interactive:
        console.print("\n[yellow]Interactive Mode[/yellow] - Type 'quit' to exit")
        while True:
            try:
                user_question = input("\n🤔 Ask a question about PE headlines: ").strip()
                if user_question.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                    
                if not user_question:
                    continue
                
                # Run the RAG pipeline
                with console.status("[bold green]Processing your question..."):
                    result = rag_graph.invoke({"question": user_question})
                
                # Display results
                console.print("\n" + "="*60)
                console.print(f"[bold cyan]Question:[/bold cyan] {user_question}")
                console.print(f"\n[bold green]Answer:[/bold green]")
                console.print(Panel(result["answer"], expand=False))
                
                # Show retrieved context
                if result.get("context"):
                    console.print(f"\n[bold yellow]Retrieved Context ({len(result['context'])} headlines):[/bold yellow]")
                    for i, doc in enumerate(result["context"], 1):
                        console.print(f"  {i}. {doc.metadata.get('headline', 'No headline')[:100]}...")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")
    
    elif question:
        # Single question mode
        console.print(f"\n[bold cyan]Question:[/bold cyan] {question}")
        
        with console.status("[bold green]Processing your question..."):
            result = rag_graph.invoke({"question": question})
        
        console.print(f"\n[bold green]Answer:[/bold green]")
        console.print(Panel(result["answer"], expand=False))
        
        # Show retrieved context
        if result.get("context"):
            console.print(f"\n[bold yellow]Retrieved Context ({len(result['context'])} headlines):[/bold yellow]")
            for i, doc in enumerate(result["context"], 1):
                console.print(f"  {i}. {doc.metadata.get('headline', 'No headline')[:100]}...")
    
    else:
        console.print("[yellow]Please provide a question with --question or use --interactive mode[/yellow]")
        console.print("\nExample usage:")
        console.print("  [cyan]python langgraph_rag.py --question 'What are recent AI trends in PE?'[/cyan]")
        console.print("  [cyan]python langgraph_rag.py --interactive[/cyan]")


if __name__ == "__main__":
    main()
