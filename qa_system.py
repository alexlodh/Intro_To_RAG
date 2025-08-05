import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import openai
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import click

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)
console = Console()

class MarkdownQASystem:
    """Question-Answering system for markdown content using OpenAI GPT-3.5-turbo"""
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 max_tokens: int = 1500,
                 temperature: float = 0.7):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.markdown_content = ""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.logger.info("OpenAI client initialized successfully")
    
    def load_markdown_content(self, file_path: str) -> bool:
        """Load markdown content from file"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                console.print(f"[red]Error: File not found: {file_path}[/red]")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as file:
                self.markdown_content = file.read()
            
            content_length = len(self.markdown_content)
            self.logger.info(f"Successfully loaded markdown content ({content_length} characters)")
            console.print(f"[green]Loaded markdown content: {content_length:,} characters[/green]")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading markdown content: {e}")
            console.print(f"[red]Error loading content: {e}[/red]")
            return False
    
    def _create_system_prompt(self) -> str:
        """Create system prompt with context about the markdown content"""
        return f"""You are an AI assistant specialized in analyzing and answering questions about PE (Private Equity) industry headlines and data.

You have access to a comprehensive markdown file containing PE industry headlines with the following structure:
- Date: Headlines organized by date (YYYYMMDD format)
- Category: Different categories like Personnel/Office, New Client, Research, etc.
- Headline: Detailed information about industry developments
- Vendor(s): Companies and organizations mentioned

The data covers recent developments in the PE industry including:
- Company personnel changes and new hires
- New client acquisitions and partnerships
- Product launches and updates
- Research publications and insights
- Deal activities and acquisitions
- Technology integrations and partnerships

When answering questions:
1. Be specific and cite relevant headlines when possible
2. Provide context about dates, companies, and categories
3. If you notice trends or patterns, highlight them
4. Be concise but comprehensive in your responses
5. If a question cannot be answered with the available data, clearly state that

Current date for reference: {datetime.now().strftime('%Y-%m-%d')}

Content to analyze:
{self.markdown_content}"""
    
    def _create_user_prompt(self, question: str) -> str:
        """Create user prompt with the specific question"""
        return f"""Based on the PE industry headlines data provided in the system message, please answer the following question:

{question}

Please provide a detailed and informative response based on the available data."""
    
    def ask_question(self, question: str) -> Optional[str]:
        """Ask a question about the markdown content using OpenAI"""
        if not self.markdown_content:
            console.print("[red]Error: No markdown content loaded. Please load content first.[/red]")
            return None
        
        try:
            console.print(f"[blue]Processing question: {question}[/blue]")
            
            # Check if content is too long for the model
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(question)
            
            # Rough token estimation (1 token ≈ 4 characters)
            estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4
            
            if estimated_tokens > 15000:  # Leave room for response
                console.print("[yellow]Warning: Content is very large. Response might be truncated.[/yellow]")
                self.logger.warning(f"Large content detected: ~{estimated_tokens} tokens")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content
            self.logger.info(f"Successfully generated response for question: {question[:50]}...")
            
            return answer
            
        except openai.APIError as e:
            error_msg = f"OpenAI API error: {e}"
            self.logger.error(error_msg)
            console.print(f"[red]{error_msg}[/red]")
            return None
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            self.logger.error(error_msg)
            console.print(f"[red]{error_msg}[/red]")
            return None
    
    def interactive_session(self):
        """Start an interactive Q&A session"""
        console.print(Panel.fit(
            "[bold green]PE Headlines Q&A System[/bold green]\n"
            "Ask questions about the PE industry headlines data.\n"
            "Type 'quit', 'exit', or 'q' to end the session.",
            border_style="blue"
        ))
        
        while True:
            try:
                question = click.prompt("\n[bold blue]Your question[/bold blue]", type=str)
                
                if question.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                if not question.strip():
                    console.print("[yellow]Please enter a valid question.[/yellow]")
                    continue
                
                answer = self.ask_question(question)
                
                if answer:
                    console.print("\n" + "="*80)
                    console.print(Panel(
                        Markdown(answer),
                        title="[bold green]Answer[/bold green]",
                        border_style="green"
                    ))
                    console.print("="*80)
                else:
                    console.print("[red]Sorry, I couldn't generate an answer for that question.[/red]")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
                break
            except EOFError:
                console.print("\n[yellow]Session ended. Goodbye![/yellow]")
                break
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the loaded content"""
        if not self.markdown_content:
            return {}
        
        lines = self.markdown_content.split('\n')
        
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
        
        return {
            "total_characters": len(self.markdown_content),
            "total_lines": len(lines),
            "estimated_headlines": table_rows,
            "estimated_tokens": len(self.markdown_content) // 4
        }

def display_stats(stats: Dict[str, Any]):
    """Display content statistics"""
    if not stats:
        console.print("[yellow]No content loaded[/yellow]")
        return
    
    console.print(Panel(
        f"""[bold]Content Statistics:[/bold]
        
• Total Characters: {stats['total_characters']:,}
• Total Lines: {stats['total_lines']:,}
• Estimated Headlines: {stats['estimated_headlines']:,}
• Estimated Tokens: {stats['estimated_tokens']:,}""",
        title="[bold blue]Markdown Content Info[/bold blue]",
        border_style="blue"
    ))

@click.group()
def qa_cli():
    """PE Headlines Question-Answering System using OpenAI GPT-3.5-turbo"""
    pass

@qa_cli.command()
@click.option("--file-path", default="headlines.md", help="Path to markdown file")
@click.option("--question", help="Single question to ask (optional)")
@click.option("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
@click.option("--max-tokens", default=1500, help="Maximum tokens in response")
@click.option("--temperature", default=0.7, help="Temperature for response generation")
def ask(file_path: str, question: str, model: str, max_tokens: int, temperature: float):
    """Ask questions about the markdown content"""
    
    # Initialize QA system
    try:
        qa_system = MarkdownQASystem(
            model_name=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    except Exception as e:
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
            console.print(Panel(
                Markdown(answer),
                title="[bold green]Answer[/bold green]",
                border_style="green"
            ))
            console.print("="*80)
    else:
        qa_system.interactive_session()

@qa_cli.command()
@click.option("--file-path", default="headlines.md", help="Path to markdown file")
def stats(file_path: str):
    """Show statistics about the markdown content"""
    qa_system = MarkdownQASystem()
    
    if qa_system.load_markdown_content(file_path):
        stats = qa_system.get_summary_stats()
        display_stats(stats)

if __name__ == "__main__":
    qa_cli()
