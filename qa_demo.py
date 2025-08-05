#!/usr/bin/env python3
"""
Demo script for the PE Headlines Question-Answering System
This script demonstrates how to use the QA system with example questions
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Add current directory to path so we can import qa_system
sys.path.append(str(Path(__file__).parent))

from qa_system import MarkdownQASystem, display_stats

# Load environment variables
load_dotenv()

console = Console()

def check_prerequisites():
    """Check if all prerequisites are met"""
    issues = []
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY environment variable is not set")
    
    # Check for headlines file
    if not Path("headlines.md").exists():
        issues.append("headlines.md file not found")
    
    return issues

def run_demo():
    """Run the QA system demo with example questions"""
    
    console.print(Panel.fit(
        "[bold green]PE Headlines Q&A System Demo[/bold green]\n"
        "This demo will ask several example questions about the PE industry headlines data.",
        border_style="blue"
    ))
    
    # Check prerequisites
    issues = check_prerequisites()
    if issues:
        console.print("[red]Prerequisites not met:[/red]")
        for issue in issues:
            console.print(f"  • {issue}")
        console.print("\n[yellow]Please ensure you have:[/yellow]")
        console.print("  1. Set OPENAI_API_KEY environment variable")
        console.print("  2. headlines.md file in the current directory")
        return False
    
    # Initialize QA system
    try:
        console.print("[blue]Initializing QA system...[/blue]")
        qa_system = MarkdownQASystem()
        
        # Load content
        if not qa_system.load_markdown_content("headlines.md"):
            return False
        
        # Display content stats
        stats = qa_system.get_summary_stats()
        display_stats(stats)
        
    except Exception as e:
        console.print(f"[red]Error initializing QA system: {e}[/red]")
        return False
    
    # Example questions to demonstrate the system
    example_questions = [
        "What are the recent trends in AI adoption in the private equity industry?",
        "Which companies have made significant personnel changes recently?",
        "What new products or platforms have been launched in the past week?",
        "What are the main categories of headlines in this dataset?",
        "Which companies are mentioned most frequently in partnership announcements?",
        "What types of acquisitions or deal activities have occurred recently?",
        "Are there any notable trends in client acquisitions?",
        "What research or insights have been published recently in the PE space?"
    ]
    
    console.print(f"\n[bold blue]Running demo with {len(example_questions)} example questions...[/bold blue]\n")
    
    for i, question in enumerate(example_questions, 1):
        console.print(f"\n{'='*100}")
        console.print(f"[bold cyan]Demo Question {i}/{len(example_questions)}:[/bold cyan]")
        console.print(f"[bold white]{question}[/bold white]")
        console.print('='*100)
        
        try:
            answer = qa_system.ask_question(question)
            
            if answer:
                console.print(Panel(
                    Markdown(answer),
                    title="[bold green]Answer[/bold green]",
                    border_style="green"
                ))
            else:
                console.print("[red]Failed to generate answer[/red]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Demo interrupted by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error processing question: {e}[/red]")
            continue
        
        # Add a small pause between questions
        import time
        time.sleep(1)
    
    console.print(f"\n{'='*100}")
    console.print("[bold green]Demo completed![/bold green]")
    console.print("\n[blue]To run the interactive Q&A system, use:[/blue]")
    console.print("  python main.py ask")
    console.print("\n[blue]To ask a single question, use:[/blue]")
    console.print('  python main.py ask --question "Your question here"')
    console.print('='*100)
    
    return True

def run_interactive_mode():
    """Run interactive mode after demo"""
    try:
        qa_system = MarkdownQASystem()
        if qa_system.load_markdown_content("headlines.md"):
            qa_system.interactive_session()
    except Exception as e:
        console.print(f"[red]Error in interactive mode: {e}[/red]")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PE Headlines QA System Demo")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run interactive mode after demo")
    parser.add_argument("--demo-only", action="store_true",
                       help="Run demo only (default)")
    
    args = parser.parse_args()
    
    # Run demo
    demo_success = run_demo()
    
    # Optionally run interactive mode
    if args.interactive and demo_success:
        console.print("\n[blue]Starting interactive mode...[/blue]")
        run_interactive_mode()
