#!/usr/bin/env python3
"""
LangChain RAG Package Main Entry Point

This module provides a unified command-line interface for running all RAG compon    try:
        # Check for dependencies first
        if not check_and_install_dependencies(['loguru', 'langchain_openai', 'qdrant_client', 'langchain_qdrant', 'langgraph']):
            return 1
        
        # Import the RAG module using importlib1_semantic_search_engine.py: Basic semantic search setup and testing
- 02_rag_pt1.py: Advanced RAG with query analysis and metadata filtering  
- 03_rag_pt2.py: Conversational RAG with memory and tools

Usage:
    python -m langchain_rag <module> [options]

Examples:
    python -m langchain_rag search               # Run semantic search engine
    python -m langchain_rag rag --demo           # Run RAG demo mode
    python -m langchain_rag conv                 # Run conversational RAG
    python -m langchain_rag conv --test          # Test conversational RAG
    python -m langchain_rag --help               # Show this help
"""

import sys
import os
import argparse
import importlib.util
import traceback
from typing import Optional, List

# Add the parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def check_dependencies(required_packages: List[str]) -> List[str]:
    """Check if required packages are installed."""
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    return missing

def install_dependencies() -> bool:
    """Install dependencies from requirements.txt if it exists."""
    requirements_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print("❌ requirements.txt not found in project root")
        return False
    
    print("📦 Installing dependencies from requirements.txt...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_path
        ], capture_output=True, text=True, check=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully!")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during installation: {e}")
        return False

def check_and_install_dependencies(required_packages: List[str]) -> bool:
    """Check dependencies and offer to install them if missing."""
    missing_deps = check_dependencies(required_packages)
    
    if not missing_deps:
        return True
    
    print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
    
    # In interactive mode, ask user if they want to install
    if sys.stdin.isatty():
        try:
            response = input("Would you like to install missing dependencies? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                return install_dependencies()
        except (KeyboardInterrupt, EOFError):
            print("\n⚠️ Installation cancelled by user")
    
    print("Please install them manually with: pip install -r requirements.txt")
    return False

def setup_search_engine():
    """Run the semantic search engine setup and demo."""
    print("🔍 Setting up Semantic Search Engine...")
    print("=" * 60)
    
    try:
        # Check for dependencies first
        if not check_and_install_dependencies(['langchain_openai', 'qdrant_client', 'langchain_qdrant']):
            return 1
        
        # Import and run the semantic search engine using importlib
        # Get the path to the semantic search engine module
        current_dir = os.path.dirname(__file__)
        module_path = os.path.join(current_dir, "01_semantic_search_engine.py")
        
        if not os.path.exists(module_path):
            print(f"❌ Module not found: {module_path}")
            return 1
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("semantic_search", module_path)
        search_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(search_module)
        
        print(f"Loading documents from: {search_module.MARKDOWN_FILE}")
        docs = search_module.load_md_table_as_documents(search_module.MARKDOWN_FILE)
        print(f"Loaded {len(docs)} documents")
        
        print("Setting up Qdrant collection and running demos...")
        search_module.setup_qdrant_collection()
        
        print("✅ Semantic search engine setup completed!")
        print("\nNext steps:")
        print("- Run 'python -m langchain_rag rag' for advanced RAG")
        print("- Run 'python -m langchain_rag conv' for conversational RAG")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies with: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"❌ Error setting up search engine: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def run_rag_system(demo: bool = False, delete_collection: bool = False):
    """Run the advanced RAG system (02_rag_pt1.py)."""
    if demo:
        print("🤖 Running Advanced RAG Demo...")
    else:
        print("🤖 Starting Advanced RAG Interactive Mode...")
    print("=" * 60)
    
    try:
        # Check for dependencies first
        missing_deps = check_dependencies(['loguru', 'langchain_openai', 'qdrant_client', 'langchain_qdrant', 'langgraph'])
        if missing_deps:
            print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
            print("Please install them with: pip install -r requirements.txt")
            return 1
        
                # Import the RAG module using importlib
        current_dir = os.path.dirname(__file__)
        module_path = os.path.join(current_dir, "02_rag_pt1.py")
        
        if not os.path.exists(module_path):
            print(f"❌ Module not found: {module_path}")
            return 1
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("rag_pt1", module_path)
        rag_pt1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag_pt1)
        
        if delete_collection:
            print("🗑️ Deleting existing collection...")
            rag_pt1.delete_collection()
            return 0
        
        # Set up command line arguments for the module
        original_argv = sys.argv[:]
        try:
            if demo:
                sys.argv = ['02_rag_pt1.py', '--demo']
            else:
                sys.argv = ['02_rag_pt1.py']
            
            # Run the main function
            rag_pt1.main()
            
        finally:
            sys.argv = original_argv
        
        print("✅ RAG system session completed!")
        
    except KeyboardInterrupt:
        print("\n👋 RAG session interrupted by user")
        return 0
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies with: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"❌ Error running RAG system: {e}")
        traceback.print_exc()
        return 1
    
    return 0

def run_conversational_rag(mode: str = "interactive"):
    """Run the conversational RAG system (03_rag_pt2.py)."""
    mode_descriptions = {
        "interactive": "🗣️ Starting Conversational RAG Interactive Mode...",
        "demo": "🗣️ Running Conversational RAG Demo...",
        "test": "🗣️ Running Conversational RAG Test..."
    }
    
    print(mode_descriptions.get(mode, "🗣️ Starting Conversational RAG..."))
    print("=" * 60)
    
    try:
        # Check for dependencies first
        if not check_and_install_dependencies(['loguru', 'langchain_openai', 'qdrant_client', 'langchain_qdrant', 'langgraph']):
            return 1
        
        # Import the conversational RAG module using importlib
        current_dir = os.path.dirname(__file__)
        module_path = os.path.join(current_dir, "03_rag_pt2.py")
        
        if not os.path.exists(module_path):
            print(f"❌ Module not found: {module_path}")
            return 1
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("rag_pt2", module_path)
        rag_pt2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag_pt2)
        
        # Set up command line arguments for the module
        original_argv = sys.argv[:]
        try:
            if mode == "demo":
                sys.argv = ['03_rag_pt2.py', '--demo']
            elif mode == "test":
                sys.argv = ['03_rag_pt2.py', '--test']
            elif mode == "delete":
                sys.argv = ['03_rag_pt2.py', '--delete-collection']
                rag_pt2.delete_collection()
                return 0
            else:
                sys.argv = ['03_rag_pt2.py']
            
            # Run the main function
            rag_pt2.main()
            
        finally:
            sys.argv = original_argv
        
        print("✅ Conversational RAG session completed!")
        
    except KeyboardInterrupt:
        print("\n👋 Conversational RAG session interrupted by user")
        return 0
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies with: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"❌ Error running conversational RAG: {e}")
        traceback.print_exc()
        return 1
    
    return 0

def show_module_help(module: Optional[str] = None):
    """Show help for specific modules."""
    if module == "search":
        print("""
🔍 SEMANTIC SEARCH ENGINE (01_semantic_search_engine.py)
================================================

Basic semantic search setup and testing module.

Features:
- Document loading from markdown tables
- Qdrant vector store setup
- Embedding generation with OpenAI
- Similarity search demonstrations
- Deduplication and error handling

Usage:
    python -m langchain_rag search

This module sets up the foundation for RAG systems by:
1. Loading PE industry headlines from markdown
2. Creating embeddings using OpenAI's text-embedding-3-small
3. Storing vectors in local Qdrant database
4. Running search demos to verify functionality
        """)
    
    elif module == "rag":
        print("""
🤖 ADVANCED RAG SYSTEM (02_rag_pt1.py)
=======================================

Advanced RAG with query analysis and metadata filtering.

Features:
- Structured query analysis with function calling
- Self-querying retriever with metadata filtering
- Category and year-based filtering
- Interactive Q&A mode
- LangGraph orchestration

Usage:
    python -m langchain_rag rag [--demo] [--delete-collection]

Options:
    --demo              Run demo mode with predefined questions
    --delete-collection Delete existing vector collection

Interactive mode supports:
- Natural language questions about PE headlines
- Automatic category detection and filtering
- Year-based temporal filtering
- Multi-step query processing
        """)
    
    elif module == "conv":
        print("""
🗣️ CONVERSATIONAL RAG (03_rag_pt2.py)
======================================

Conversational RAG with memory and tool-based architecture.

Features:
- Persistent conversation memory across turns
- Tool-based search with category/year filtering
- Thread-based conversation isolation
- Context-aware follow-up question handling
- Advanced conversation state management

Usage:
    python -m langchain_rag conv [--demo] [--test] [--delete-collection]

Options:
    --demo              Run demo with sample conversation
    --test              Quick test mode
    --delete-collection Delete existing vector collection

Interactive features:
- Multi-turn conversations with memory
- Context from previous exchanges
- Natural follow-up question handling
- Tools for search, categories, and available years
- Support for multiple conversation threads
        """)
    
    else:
        print("""
🚀 LANGCHAIN RAG PACKAGE
========================

Comprehensive RAG (Retrieval Augmented Generation) implementation
for Private Equity industry headlines analysis.

Available Modules:

1. 🔍 search   - Semantic Search Engine (Basic setup)
2. 🤖 rag      - Advanced RAG System (Query analysis + filtering)  
3. 🗣️ conv     - Conversational RAG (Memory + tools)
4. 📦 install  - Install project dependencies

Usage:
    python -m langchain_rag <module> [options]

Examples:
    python -m langchain_rag install              # Install dependencies
    python -m langchain_rag search               # Setup semantic search
    python -m langchain_rag rag                  # Interactive RAG mode
    python -m langchain_rag rag --demo           # RAG demo mode
    python -m langchain_rag conv                 # Conversational RAG
    python -m langchain_rag conv --test          # Test conversational RAG

Get module-specific help:
    python -m langchain_rag help <module>        # Detailed module help

System Requirements:
- OpenAI API key (set in environment)
- Python 3.8+
- Dependencies from requirements.txt

The modules build upon each other:
search → rag → conv (increasing sophistication)
        """)

def main():
    """Main entry point for the langchain_rag package."""
    parser = argparse.ArgumentParser(
        description="LangChain RAG Package - Semantic Search and Q&A for PE Headlines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s install                   # Install project dependencies
  %(prog)s search                    # Run semantic search engine setup
  %(prog)s rag                       # Interactive RAG mode
  %(prog)s rag --demo                # RAG demo mode
  %(prog)s conv                      # Conversational RAG
  %(prog)s conv --test               # Test conversational RAG
  %(prog)s help conv                 # Get help for conversational module
        """
    )
    
    parser.add_argument(
        'module',
        choices=['search', 'rag', 'conv', 'help', 'install'],
        help='Module to run: search (basic), rag (advanced), conv (conversational), help, or install'
    )
    
    parser.add_argument(
        'help_module',
        nargs='?',
        choices=['search', 'rag', 'conv'],
        help='Show help for specific module (use with help command)'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode (for rag and conv modules)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run quick test (for conv module)'
    )
    
    parser.add_argument(
        '--delete-collection',
        action='store_true',
        help='Delete existing vector collection'
    )
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    # Handle help command
    if args.module == 'help':
        show_module_help(args.help_module)
        return 0
    
    # Handle install command
    if args.module == 'install':
        if install_dependencies():
            print("✅ All dependencies installed successfully!")
            return 0
        else:
            print("❌ Failed to install dependencies")
            return 1
    
    # Run the appropriate module
    try:
        if args.module == 'search':
            if args.demo or args.test or args.delete_collection:
                print("⚠️ Search module doesn't support --demo, --test, or --delete-collection options")
                return 1
            return setup_search_engine()
        
        elif args.module == 'rag':
            if args.test:
                print("⚠️ RAG module doesn't support --test option (use --demo instead)")
                return 1
            return run_rag_system(demo=args.demo, delete_collection=args.delete_collection)
        
        elif args.module == 'conv':
            if args.delete_collection:
                return run_conversational_rag(mode="delete")
            elif args.demo:
                return run_conversational_rag(mode="demo")
            elif args.test:
                return run_conversational_rag(mode="test")
            else:
                return run_conversational_rag(mode="interactive")
        
        else:
            print(f"❌ Unknown module: {args.module}")
            parser.print_help()
            return 1
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
