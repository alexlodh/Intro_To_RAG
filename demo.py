#!/usr/bin/env python3
"""
Demo script for PE Headlines RAG System
"""

import time
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display its output"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run the demo"""
    print("🎯 PE Headlines RAG System Demo")
    print("This demo will show you how to embed and search the headlines data.")
    
    # Check if Qdrant is running
    print("\n📊 Checking if Qdrant is running...")
    qdrant_check = subprocess.run(
        "curl -s http://localhost:6333/collections >/dev/null 2>&1", 
        shell=True
    )
    
    if qdrant_check.returncode != 0:
        print("❌ Qdrant is not running. Please start it first:")
        print("   docker-compose up -d qdrant")
        print("   or")
        print("   docker run -p 6333:6333 qdrant/qdrant:latest")
        return
    else:
        print("✅ Qdrant is running!")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        return
    
    # Embed the headlines
    if not run_command(
        "python main.py embed --file-path headlines.md --force-recreate", 
        "Embedding headlines into Qdrant"
    ):
        print("❌ Failed to embed headlines")
        return
    
    # Show collection info
    run_command("python main.py info", "Showing collection information")
    
    # Demo searches
    searches = [
        ("AI artificial intelligence", "Finding AI-related headlines"),
        ("portfolio management --vendor BlackRock", "Finding BlackRock portfolio management headlines"), 
        ("partnership --category 'Partnerships & Integrations'", "Finding partnership headlines"),
        ("investment platform --year 2025 --limit 3", "Finding recent investment platform headlines"),
        ("fund --date-from 20250720 --date-to 20250725", "Finding headlines from specific date range"),
    ]
    
    for query, description in searches:
        time.sleep(1)  # Brief pause between searches
        run_command(f"python main.py search --query {query}", description)
    
    print(f"\n{'='*60}")
    print("🎉 Demo completed!")
    print("You can now run your own searches with:")
    print("python main.py search --query 'your search terms'")
    print("Use --help to see all available options.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
