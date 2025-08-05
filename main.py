import os
import re
import hashlib
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, Range, GeoBoundingBox,
    UpdateStatus, CollectionInfo, SearchParams, MatchAny
)
from sentence_transformers import SentenceTransformer

# Import QA system
from qa_system import MarkdownQASystem, display_stats

try:
    from langsmith import Client
    from langsmith.run_helpers import traceable
    from dotenv import load_dotenv
    LANGSMITH_AVAILABLE = True
    load_dotenv()
except ImportError:
    LANGSMITH_AVAILABLE = False
    def traceable(func):
        return func

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_system.log'),
        logging.StreamHandler()
    ]
)

# Create logs directory if it doesn't exist
Path('logs').mkdir(exist_ok=True)
logger = logging.getLogger(__name__)

console = Console()

@dataclass
class HeadlineEntry:
    """Data class for a single headline entry"""
    date: str
    category: str
    headline: str
    vendors: List[str]
    date_obj: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Qdrant payload"""
        return {
            "date": self.date,
            "category": self.category,
            "headline": self.headline,
            "vendors": self.vendors,
            "year": self.date_obj.year,
            "month": self.date_obj.month,
            "day": self.date_obj.day,
            "vendor_count": len(self.vendors),
            "headline_length": len(self.headline),
            "has_vendors": len(self.vendors) > 0
        }
    
    def get_hash(self) -> str:
        """Generate a unique hash for this headline entry"""
        content = f"{self.date}|{self.category}|{self.headline}|{'|'.join(sorted(self.vendors))}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

class HeadlineProcessor:
    """Processes headlines markdown file and extracts structured data"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.entries: List[HeadlineEntry] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @traceable
    def parse_markdown_table(self) -> List[HeadlineEntry]:
        """Parse the markdown table and extract headline entries"""
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.logger.info(f"Successfully read file: {self.file_path}")
        except FileNotFoundError:
            self.logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"Headlines file not found: {self.file_path}")
        except PermissionError:
            self.logger.error(f"Permission denied reading file: {self.file_path}")
            raise PermissionError(f"Cannot read file: {self.file_path}")
        except Exception as e:
            self.logger.error(f"Unexpected error reading file {self.file_path}: {e}")
            raise RuntimeError(f"Error reading file: {e}")
        
        # Find the table content (skip header and separator)
        lines = content.split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            if line.startswith('| Date'):
                in_table = True
                continue
            elif line.startswith('| ---'):
                continue
            elif in_table and line.startswith('|'):
                table_lines.append(line)
        
        entries = []
        parse_errors = 0
        
        for line_num, line in enumerate(table_lines, 1):
            if not line.strip():
                continue
                
            try:
                # Split by | and clean up
                parts = [part.strip() for part in line.split('|')[1:-1]]  # Remove empty first/last elements
                
                if len(parts) >= 4:
                    date_str = parts[0].strip()
                    category = parts[1].strip()
                    headline = parts[2].strip()
                    vendors_str = parts[3].strip()
                    
                    # Parse date
                    try:
                        date_obj = datetime.strptime(date_str, "%Y%m%d")
                    except ValueError as e:
                        self.logger.warning(f"Could not parse date {date_str} on line {line_num}: {e}")
                        parse_errors += 1
                        continue
                    
                    # Parse vendors
                    vendors = []
                    if vendors_str:
                        vendors = [v.strip() for v in vendors_str.split(',') if v.strip()]
                    
                    entry = HeadlineEntry(
                        date=date_str,
                        category=category,
                        headline=headline,
                        vendors=vendors,
                        date_obj=date_obj
                    )
                    entries.append(entry)
                else:
                    self.logger.warning(f"Insufficient columns on line {line_num}: {line}")
                    parse_errors += 1
                    
            except Exception as e:
                self.logger.error(f"Error parsing line {line_num}: {e}")
                parse_errors += 1
        
        self.entries = entries
        self.logger.info(f"Successfully parsed {len(entries)} entries with {parse_errors} errors")
        
        if parse_errors > 0:
            console.print(f"[yellow]Warning: {parse_errors} lines could not be parsed[/yellow]")
        
        return entries

class QdrantEmbedder:
    """Handles embedding and storing headlines in Qdrant database"""
    
    def __init__(self, 
                 collection_name: str = "pe_headlines",
                 model_name: str = "all-MiniLM-L6-v2",
                 qdrant_url: str = "./qdrant_data"):
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize LangSmith if available
        if LANGSMITH_AVAILABLE and os.getenv("LANGCHAIN_API_KEY"):
            self.langsmith_client = Client()
            self.logger.info("LangSmith integration enabled")
        else:
            self.langsmith_client = None
            self.logger.info("LangSmith integration not available or not configured")
        
        # Use local file storage instead of in-memory for persistence
        try:
            if qdrant_url == "./qdrant_data":
                self.client = QdrantClient(path=qdrant_url)
                self.logger.info(f"Connected to local Qdrant at {qdrant_url}")
            else:
                self.client = QdrantClient(url=qdrant_url)
                self.logger.info(f"Connected to remote Qdrant at {qdrant_url}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Cannot connect to Qdrant: {e}")
        
    def create_collection(self, force_recreate: bool = False) -> bool:
        """Create or recreate the Qdrant collection"""
        
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists:
                if force_recreate:
                    self.logger.info(f"Force recreating collection '{self.collection_name}'")
                    console.print(f"[yellow]Deleting existing collection '{self.collection_name}'[/yellow]")
                    self.client.delete_collection(self.collection_name)
                    collection_exists = False
                else:
                    self.logger.info(f"Collection '{self.collection_name}' already exists, skipping creation")
                    console.print(f"[blue]Collection '{self.collection_name}' already exists[/blue]")
                    return False
            
            if not collection_exists:
                # Get embedding dimension from model
                try:
                    sample_embedding = self.model.encode(["sample text"])
                    vector_size = len(sample_embedding[0])
                    self.logger.info(f"Determined vector size: {vector_size}")
                except Exception as e:
                    self.logger.error(f"Failed to determine vector size: {e}")
                    raise RuntimeError(f"Cannot determine embedding size: {e}")
                
                console.print(f"[green]Creating collection '{self.collection_name}' with vector size {vector_size}[/green]")
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Successfully created collection '{self.collection_name}'")
                return True
                
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            console.print(f"[red]Error creating collection: {e}[/red]")
            return False
    
    def _get_existing_hashes(self) -> set:
        """Get existing document hashes from the collection"""
        try:
            # Try to get all points with just the hash field
            points, next_page_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["document_hash"],
                limit=10000  # Adjust based on your expected data size
            )
            
            existing_hashes = set()
            for point in points:
                if point.payload and "document_hash" in point.payload:
                    existing_hashes.add(point.payload["document_hash"])
            
            # If we have more points, continue scrolling
            while next_page_offset:
                points, next_page_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    with_payload=["document_hash"],
                    offset=next_page_offset,
                    limit=10000
                )
                for point in points:
                    if point.payload and "document_hash" in point.payload:
                        existing_hashes.add(point.payload["document_hash"])
            
            self.logger.info(f"Found {len(existing_hashes)} existing documents in collection")
            return existing_hashes
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve existing hashes, assuming empty collection: {e}")
            return set()
    
    @traceable
    def embed_headlines(self, entries: List[HeadlineEntry], skip_duplicates: bool = True) -> bool:
        """Embed headlines and store in Qdrant with metadata, avoiding duplicates"""
        
        self.logger.info(f"Starting to embed {len(entries)} headlines")
        console.print(f"[blue]Processing {len(entries)} headlines...[/blue]")
        
        # Get existing hashes to avoid duplicates
        existing_hashes = set()
        if skip_duplicates:
            existing_hashes = self._get_existing_hashes()
        
        # Filter out duplicates
        new_entries = []
        duplicate_count = 0
        
        for entry in entries:
            entry_hash = entry.get_hash()
            if skip_duplicates and entry_hash in existing_hashes:
                duplicate_count += 1
                self.logger.debug(f"Skipping duplicate entry: {entry.headline[:50]}...")
            else:
                new_entries.append((entry, entry_hash))
        
        if duplicate_count > 0:
            console.print(f"[yellow]Skipping {duplicate_count} duplicate entries[/yellow]")
            self.logger.info(f"Skipped {duplicate_count} duplicate entries")
        
        if not new_entries:
            console.print("[yellow]No new entries to process[/yellow]")
            self.logger.info("No new entries to process")
            return True
        
        console.print(f"[blue]Embedding {len(new_entries)} new headlines...[/blue]")
        
        # Prepare texts for embedding
        texts = [entry.headline for entry, _ in new_entries]
        
        # Generate embeddings
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Generating embeddings...", total=None)
                embeddings = self.model.encode(texts, show_progress_bar=False)
                progress.advance(task)
                
            self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            console.print(f"[red]Error generating embeddings: {e}[/red]")
            return False
        
        # Prepare points for upload
        points = []
        for i, ((entry, entry_hash), embedding) in enumerate(zip(new_entries, embeddings)):
            payload = entry.to_dict()
            payload["document_hash"] = entry_hash  # Add hash for deduplication
            
            point = PointStruct(
                id=entry_hash,  # Use hash as ID for consistent deduplication
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
        
        # Upload to Qdrant
        try:
            console.print(f"[blue]Uploading {len(points)} points to Qdrant...[/blue]")
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            if operation_info.status == UpdateStatus.COMPLETED:
                success_msg = f"Successfully uploaded {len(points)} headlines to Qdrant!"
                console.print(f"[green]{success_msg}[/green]")
                self.logger.info(success_msg)
                
                # Log to LangSmith if available
                if self.langsmith_client:
                    try:
                        self.langsmith_client.create_run(
                            name="embed_headlines",
                            run_type="tool",
                            inputs={"num_entries": len(entries), "new_entries": len(points)},
                            outputs={"status": "success", "uploaded": len(points)}
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to log to LangSmith: {e}")
                
                return True
            else:
                error_msg = f"Upload failed with status: {operation_info.status}"
                console.print(f"[red]{error_msg}[/red]")
                self.logger.error(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"Error uploading to Qdrant: {e}"
            console.print(f"[red]{error_msg}[/red]")
            self.logger.error(error_msg)
            return False
    
    @traceable
    def search_headlines(self, 
                        query: str,
                        limit: int = 10,
                        filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search headlines with optional metadata filters"""
        
        self.logger.info(f"Searching for query: '{query}' with limit: {limit}")
        
        # Generate query embedding
        try:
            query_embedding = self.model.encode([query])[0]
            self.logger.debug("Successfully generated query embedding")
        except Exception as e:
            self.logger.error(f"Failed to generate query embedding: {e}")
            console.print(f"[red]Error generating query embedding: {e}[/red]")
            return []
        
        # Build filter if provided
        qdrant_filter = None
        if filters:
            self.logger.info(f"Applying filters: {filters}")
            conditions = []
            
            try:
                # Category filter
                if "category" in filters:
                    conditions.append(
                        FieldCondition(key="category", match=MatchValue(value=filters["category"]))
                    )
                
                # Date range filter
                if "date_from" in filters or "date_to" in filters:
                    date_range = {}
                    if "date_from" in filters:
                        date_range["gte"] = int(filters["date_from"])
                    if "date_to" in filters:
                        date_range["lte"] = int(filters["date_to"])
                    
                    conditions.append(
                        FieldCondition(key="date", range=Range(**date_range))
                    )
                
                # Year filter
                if "year" in filters:
                    conditions.append(
                        FieldCondition(key="year", match=MatchValue(value=filters["year"]))
                    )
                
                # Vendor filter
                if "vendor" in filters:
                    conditions.append(
                        FieldCondition(key="vendors", match=MatchAny(any=[filters["vendor"]]))
                    )
                
                # Has vendors filter
                if "has_vendors" in filters:
                    conditions.append(
                        FieldCondition(key="has_vendors", match=MatchValue(value=filters["has_vendors"]))
                    )
                
                if conditions:
                    qdrant_filter = Filter(must=conditions)
                    
            except Exception as e:
                self.logger.error(f"Error building filters: {e}")
                console.print(f"[red]Error building filters: {e}[/red]")
                return []
        
        # Perform search
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True
            )
            
            results = []
            for point in search_result:
                result = {
                    "score": point.score,
                    "id": point.id,
                    **point.payload
                }
                results.append(result)
            
            self.logger.info(f"Search completed, found {len(results)} results")
            
            # Log to LangSmith if available
            if self.langsmith_client:
                try:
                    self.langsmith_client.create_run(
                        name="search_headlines",
                        run_type="retriever",
                        inputs={"query": query, "filters": filters, "limit": limit},
                        outputs={"num_results": len(results), "top_score": results[0]["score"] if results else 0}
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to log search to LangSmith: {e}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error searching: {e}"
            self.logger.error(error_msg)
            console.print(f"[red]{error_msg}[/red]")
            return []
    
    def get_collection_info(self) -> Optional[CollectionInfo]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            self.logger.info(f"Retrieved collection info for '{self.collection_name}'")
            return info
        except Exception as e:
            error_msg = f"Error getting collection info: {e}"
            self.logger.error(error_msg)
            console.print(f"[red]{error_msg}[/red]")
            return None

def display_search_results(results: List[Dict[str, Any]]):
    """Display search results in a formatted table"""
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Score", width=8)
    table.add_column("Date", width=10)
    table.add_column("Category", width=15)
    table.add_column("Headline", width=60)
    table.add_column("Vendors", width=30)
    
    for result in results:
        vendors_str = ", ".join(result.get("vendors", []))[:30] + "..." if len(", ".join(result.get("vendors", []))) > 30 else ", ".join(result.get("vendors", []))
        headline_str = result["headline"][:60] + "..." if len(result["headline"]) > 60 else result["headline"]
        
        table.add_row(
            f"{result['score']:.3f}",
            result["date"],
            result["category"],
            headline_str,
            vendors_str
        )
    
    console.print(table)

@click.group()
def cli():
    """PE Headlines RAG System with Qdrant"""
    pass

@cli.command()
@click.option("--file-path", default="headlines.md", help="Path to headlines markdown file")
@click.option("--force-recreate", is_flag=True, help="Force recreate collection")
@click.option("--skip-duplicates/--allow-duplicates", default=True, help="Skip duplicate entries (default: True)")
def embed(file_path: str, force_recreate: bool, skip_duplicates: bool):
    """Embed headlines into Qdrant database"""
    
    logger.info(f"Starting embed command with file: {file_path}")
    
    # Parse headlines
    try:
        processor = HeadlineProcessor(file_path)
        entries = processor.parse_markdown_table()
    except Exception as e:
        logger.error(f"Failed to process headlines file: {e}")
        console.print(f"[red]Error processing file: {e}[/red]")
        return
    
    if not entries:
        console.print("[red]No headlines found in file[/red]")
        logger.warning("No headlines found in file")
        return
    
    console.print(f"[green]Found {len(entries)} headlines[/green]")
    logger.info(f"Found {len(entries)} headlines")
    
    # Create embedder and collection
    try:
        embedder = QdrantEmbedder()
        collection_created = embedder.create_collection(force_recreate=force_recreate)
        
        # If collection already exists and we're not forcing recreation, inform user
        if not collection_created and not force_recreate:
            console.print("[blue]Using existing collection. Use --force-recreate to start fresh.[/blue]")
    except Exception as e:
        logger.error(f"Failed to initialize embedder or create collection: {e}")
        console.print(f"[red]Error setting up Qdrant: {e}[/red]")
        return
    
    # Embed and store
    try:
        success = embedder.embed_headlines(entries, skip_duplicates=skip_duplicates)
    except Exception as e:
        logger.error(f"Failed to embed headlines: {e}")
        console.print(f"[red]Error embedding headlines: {e}[/red]")
        return
    
    if success:
        # Show collection info
        info = embedder.get_collection_info()
        if info:
            console.print(f"[green]Collection info:[/green]")
            console.print(f"  Points count: {info.points_count}")
            console.print(f"  Vectors count: {info.vectors_count}")
            logger.info(f"Embed completed successfully. Collection has {info.points_count} points")
    else:
        logger.error("Embed operation failed")

@cli.command()
@click.option("--query", prompt="Search query", help="Query to search for")
@click.option("--limit", default=5, help="Number of results to return")
@click.option("--category", help="Filter by category")
@click.option("--vendor", help="Filter by vendor (partial match)")
@click.option("--year", type=int, help="Filter by year")
@click.option("--date-from", help="Filter from date (YYYYMMDD)")
@click.option("--date-to", help="Filter to date (YYYYMMDD)")
@click.option("--has-vendors", type=bool, help="Filter by has vendors (true/false)")
def search(query: str, limit: int, category: str, vendor: str, year: int, 
           date_from: str, date_to: str, has_vendors: bool):
    """Search headlines with optional filters"""
    
    logger.info(f"Starting search command with query: '{query}'")
    
    # Build filters
    filters = {}
    if category:
        filters["category"] = category
    if vendor:
        filters["vendor"] = vendor
    if year:
        filters["year"] = year
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to
    if has_vendors is not None:
        filters["has_vendors"] = has_vendors
    
    # Perform search
    try:
        embedder = QdrantEmbedder()
        results = embedder.search_headlines(query, limit=limit, filters=filters)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        console.print(f"[red]Search error: {e}[/red]")
        return
    
    console.print(f"[blue]Found {len(results)} results for query: '{query}'[/blue]")
    if filters:
        console.print(f"[blue]Filters applied: {filters}[/blue]")
    
    display_search_results(results)
    logger.info(f"Search completed, returned {len(results)} results")

@cli.command()
def info():
    """Show collection information"""
    logger.info("Getting collection info")
    
    try:
        embedder = QdrantEmbedder()
        info = embedder.get_collection_info()
        
        if info:
            console.print(f"[green]Collection: {embedder.collection_name}[/green]")
            console.print(f"  Points count: {info.points_count}")
            console.print(f"  Vectors count: {info.vectors_count}")
            console.print(f"  Status: {info.status}")
            logger.info(f"Collection info retrieved: {info.points_count} points")
        else:
            console.print("[red]Collection not found or error occurred[/red]")
            logger.warning("Collection not found or error occurred")
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        console.print(f"[red]Error getting collection info: {e}[/red]")

@cli.command()
@click.option("--file-path", default="headlines.md", help="Path to markdown file")
@click.option("--question", help="Single question to ask (optional)")
@click.option("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
@click.option("--max-tokens", default=1500, help="Maximum tokens in response")
@click.option("--temperature", default=0.7, help="Temperature for response generation")
def ask(file_path: str, question: str, model: str, max_tokens: int, temperature: float):
    """Ask questions about the markdown content using OpenAI GPT-3.5-turbo"""
    
    logger.info(f"Starting QA session with file: {file_path}")
    
    # Initialize QA system
    try:
        qa_system = MarkdownQASystem(
            model_name=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    except Exception as e:
        logger.error(f"Error initializing QA system: {e}")
        console.print(f"[red]Error initializing QA system: {e}[/red]")
        return
    
    # Load content
    if not qa_system.load_markdown_content(file_path):
        return
    
    # Display content stats
    stats = qa_system.get_summary_stats()
    display_stats(stats)
    
    # Handle single question or interactive mode
    if question:
        console.print(f"\n[bold blue]Question:[/bold blue] {question}")
        answer = qa_system.ask_question(question)
        
        if answer:
            console.print("\n" + "="*80)
            from rich.panel import Panel
            from rich.markdown import Markdown
            console.print(Panel(
                Markdown(answer),
                title="[bold green]Answer[/bold green]",
                border_style="green"
            ))
            console.print("="*80)
            logger.info(f"Successfully answered question: {question[:50]}...")
    else:
        qa_system.interactive_session()

@cli.command()
@click.option("--file-path", default="headlines.md", help="Path to markdown file")
def content_stats(file_path: str):
    """Show statistics about the markdown content"""
    logger.info(f"Getting content statistics for: {file_path}")
    
    try:
        # Create a minimal QA system that doesn't require OpenAI API for stats
        from pathlib import Path
        
        file_path = Path(file_path)
        if not file_path.exists():
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            return
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        lines = content.split('\n')
        
        # Count table rows (excluding header and separator)
        table_rows = 0
        in_table = False
        
        for line in lines:
            if line.startswith('| Date'):
                in_table = True
                continue
            elif line.startswith('| ---'):
                continue
            elif in_table and line.startswith('|'):
                table_rows += 1
            elif in_table and not line.startswith('|'):
                break
        
        stats = {
            "total_characters": len(content),
            "total_lines": len(lines),
            "estimated_headlines": table_rows,
            "estimated_tokens": len(content) // 4
        }
        
        display_stats(stats)
        logger.info(f"Content stats retrieved: {stats.get('estimated_headlines', 0)} headlines")
        
    except Exception as e:
        logger.error(f"Error getting content stats: {e}")
        console.print(f"[red]Error getting content statistics: {e}[/red]")

if __name__ == "__main__":
    cli()
