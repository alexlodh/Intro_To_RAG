#!/usr/bin/env python3
"""
Simple example demonstrating the PE Headlines QA System
This shows how to use the system with a single question
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

console = Console()

def simple_example():
    """Simple example of using the QA system"""
    
    console.print(Panel.fit(
        "[bold green]PE Headlines QA System - Simple Example[/bold green]\n"
        "This example shows how to ask a single question about the headlines data.",
        border_style="blue"
    ))
    
    # Check for OpenAI API key
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        console.print(Panel(
            "[red]OpenAI API Key Required[/red]\n\n"
            "To use the QA system, you need to set your OpenAI API key:\n\n"
            "1. Get your API key from: https://platform.openai.com/api-keys\n"
            "2. Create a .env file with: OPENAI_API_KEY=your_key_here\n"
            "3. Or run: python setup_qa.py\n\n"
            "[yellow]For now, showing content statistics only...[/yellow]",
            border_style="yellow"
        ))
        
        # Show content stats without requiring API key
        from main import display_stats
        
        try:
            with open("headlines.md", 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            table_rows = sum(1 for line in lines if line.startswith('|') and not line.startswith('| Date') and not line.startswith('| ---'))
            
            stats = {
                "total_characters": len(content),
                "total_lines": len(lines),
                "estimated_headlines": table_rows,
                "estimated_tokens": len(content) // 4
            }
            
            display_stats(stats)
            
        except FileNotFoundError:
            console.print("[red]headlines.md file not found[/red]")
        
        return
    
    # Try to use the QA system
    try:
        from qa_system import MarkdownQASystem
        
        console.print("[blue]Initializing QA system...[/blue]")
        qa_system = MarkdownQASystem()
        
        if not qa_system.load_markdown_content("headlines.md"):
            return
        
        # Example question
        example_question = "What are the main trends in AI adoption mentioned in the recent headlines?"
        
        console.print(f"\n[bold blue]Example Question:[/bold blue]")
        console.print(f"[white]{example_question}[/white]")
        
        console.print("\n[blue]Getting answer from OpenAI GPT-3.5-turbo...[/blue]")
        
        answer = qa_system.ask_question(example_question)
        
        if answer:
            console.print("\n" + "="*80)
            console.print(Panel(
                Markdown(answer),
                title="[bold green]AI Answer[/bold green]",
                border_style="green"
            ))
            console.print("="*80)
            
            console.print(Panel(
                "[bold green]Success![/bold green]\n\n"
                "The QA system is working correctly. You can now:\n\n"
                "• Run [cyan]python main.py ask[/cyan] for interactive mode\n"
                "• Use [cyan]python main.py ask --question \"Your question\"[/cyan] for single questions\n"
                "• Try [cyan]python qa_demo.py[/cyan] for more examples",
                border_style="green"
            ))
        else:
            console.print("[red]Failed to get answer. Please check your API key and try again.[/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\n[yellow]Try running: python setup_qa.py[/yellow]")

if __name__ == "__main__":
    simple_example()
