#!/usr/bin/env python3
"""
Setup script for the PE Headlines QA System
This script verifies the environment and tests the predefined API key
"""

import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from dotenv import load_dotenv, set_key

console = Console()

def setup_environment():
    """Setup environment variables and configuration"""
    
    console.print(Panel.fit(
        "[bold green]PE Headlines QA System Setup[/bold green]\n"
        "This setup will verify your environment and test the API connection.",
        border_style="blue"
    ))
    
    # Check if .env file exists
    env_file = Path(".env")
    
    # Load existing .env if it exists
    if env_file.exists():
        load_dotenv()
        console.print("[green].env file found[/green]")
    else:
        console.print("[yellow].env file not found, will create one[/yellow]")
    
    # Check for existing API key
    current_api_key = os.getenv("OPENAI_API_KEY")
    
    if current_api_key:
        console.print("[green]OpenAI API key is already configured[/green]")
    else:
        console.print("[yellow]OpenAI API key not found in environment[/yellow]")
        console.print("[blue]Using predefined API key from .env file[/blue]")
        
        # Load the API key from the .env file (it should already be there)
        load_dotenv(override=True)
        current_api_key = os.getenv("OPENAI_API_KEY")
        
        if not current_api_key:
            console.print("[red]No API key found in .env file. Please check your configuration.[/red]")
            return False
    
    # Test the API key
    console.print("\n[blue]Testing OpenAI API connection...[/blue]")
    
    try:
        import openai
        # Reload environment variables
        load_dotenv(override=True)
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        console.print("[green]✓ OpenAI API connection successful![/green]")
        
    except ImportError:
        console.print("[red]OpenAI package not installed. Please install requirements.txt[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ API connection failed: {e}[/red]")
        console.print("[yellow]Please check your API key and try again.[/yellow]")
        return False
    
    # Check for headlines file
    headlines_file = Path("headlines.md")
    if headlines_file.exists():
        console.print("[green]✓ headlines.md file found[/green]")
        
        # Show file stats
        with open(headlines_file, 'r') as f:
            content = f.read()
            lines = len(content.split('\n'))
            chars = len(content)
        
        console.print(f"  File size: {chars:,} characters, {lines:,} lines")
        
    else:
        console.print("[red]✗ headlines.md file not found[/red]")
        console.print("[yellow]Please ensure headlines.md is in the current directory[/yellow]")
        return False
    
    # Final success message
    console.print(Panel(
        "[bold green]Setup completed successfully![/bold green]\n\n"
        "You can now use the QA system with:\n"
        "• [cyan]python main.py ask[/cyan] - Interactive mode\n"
        "• [cyan]python main.py ask --question \"Your question\"[/cyan] - Single question\n"
        "• [cyan]python qa_demo.py[/cyan] - Run demo with example questions\n"
        "• [cyan]python main.py content-stats[/cyan] - Show content statistics",
        title="[bold blue]Ready to Go![/bold blue]",
        border_style="green"
    ))
    
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    console.print("[blue]Checking dependencies...[/blue]")
    
    required_packages = [
        "openai",
        "rich", 
        "click",
        "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            console.print(f"[green]✓ {package}[/green]")
        except ImportError:
            console.print(f"[red]✗ {package}[/red]")
            missing_packages.append(package)
    
    if missing_packages:
        console.print(f"\n[red]Missing packages: {', '.join(missing_packages)}[/red]")
        console.print("[yellow]Please install them with:[/yellow]")
        console.print("pip install -r requirements.txt")
        return False
    
    console.print("[green]All dependencies are installed![/green]")
    return True

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]PE Headlines QA System[/bold cyan]\n"
        "Setup and Configuration Tool",
        border_style="cyan"
    ))
    
    # Check dependencies first
    if not check_dependencies():
        exit(1)
    
    # Run setup
    if setup_environment():
        console.print("\n[green]Setup completed! You're ready to use the QA system.[/green]")
    else:
        console.print("\n[red]Setup failed. Please resolve the issues above and try again.[/red]")
        exit(1)
