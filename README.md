# PE Headlines Analysis System

A comprehensive system for analyzing Private Equity industry headlines using both vector search (RAG) and AI-powered question answering. This system combines Qdrant vector database for semantic search with OpenAI GPT-3.5-turbo for intelligent question-answering about the entire dataset.

## Overview

This system implements a complete RAG (Retrieval-Augmented Generation) pipeline with two main analysis modes:

1. **Vector Search & RAG**: Traditional semantic search using vector embeddings stored in Qdrant
2. **AI Question-Answering**: Full content analysis using OpenAI GPT-3.5-turbo for comprehensive insights
3. **LangGraph RAG**: Advanced workflow-based RAG using LangGraph for structured retrieval and generation

### What is RAG?

AI RAG combines Large Language Models (LLMs) with external retrieval to improve relevance, factual accuracy and customizability through:
- **Data Preparation**: Gather, normalize, chunk, and embed documents
- **Query Processing**: Convert queries to embeddings and perform similarity search
- **Generation**: Augment LLM prompts with retrieved context for accurate responses
- **Evaluation**: Assess outputs for correctness and completeness

## Features

### Vector Search & RAG System
- **Parse Headlines**: Automatically extracts structured data from markdown table format
- **Vector Embeddings**: Uses SentenceTransformers to create semantic embeddings
- **Metadata Filtering**: Rich filtering capabilities by date, category, vendor, and more
- **Semantic Search**: Find relevant headlines using natural language queries
- **Duplicate Detection**: Avoid storing duplicate entries
- **Rich CLI**: Beautiful command-line interface with progress indicators and formatted output

### AI Question-Answering System
- **Full Content Analysis**: Ask questions about the entire markdown content using OpenAI GPT-3.5-turbo
- **Interactive Mode**: Real-time Q&A session with the AI
- **Single Question Mode**: Ask one question and get an answer
- **Content Statistics**: View detailed stats about the loaded content
- **Context-Aware Responses**: AI understands PE industry terminology and trends

## Quick Start

### Prerequisites

1. **Python 3.8+**
2. **OpenAI API Key** - Already configured in the project
3. **Docker** (for Qdrant database, optional for QA-only usage)

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd Intro_To_RAG
   pip install -r requirements.txt
   ```

2. **Verify environment** (optional):
   ```bash
   python setup_qa.py
   ```
   This will verify your environment and test the API connection.

3. **The OpenAI API key is already configured** in the `.env` file - no manual setup required!

### Getting Started with Vector Search

1. **Start Qdrant Database**:
   ```bash
   # Using Docker Compose (recommended)
   docker-compose up -d qdrant
   
   # Or using Docker directly
   docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant:latest
   ```

2. **Embed Headlines**:
   ```bash
   # Embed the headlines.md file into Qdrant
   python main.py embed --file-path headlines.md
   
   # Force recreate collection if needed
   python main.py embed --file-path headlines.md --force-recreate
   ```

3. **Search Headlines**:
   ```bash
   # Basic search
   python main.py search --query "AI artificial intelligence"
   
   # Search with filters
   python main.py search --query "private equity" --category "New Client" --year 2025
   ```

### Getting Started with AI Q&A

1. **Check Content Statistics**:
   ```bash
   python main.py content-stats
   ```

2. **Interactive Q&A Session**:
   ```bash
   python main.py ask
   ```

3. **Single Question**:
   ```bash
   python main.py ask --question "What are the recent AI trends in PE?"
   ```

4. **Run Demo**:
   ```bash
   python qa_demo.py
   ```

## Usage Guide

### Question-Answering System

#### Interactive Mode
```bash
python main.py ask
```
Start an interactive session where you can ask multiple questions about the PE headlines.

#### Single Question Mode
```bash
python main.py ask --question "What are the recent trends in AI adoption in the private equity industry?"
```

#### Demo with Example Questions
```bash
python qa_demo.py
```
Runs the system with predefined example questions to showcase capabilities.

#### Content Statistics
```bash
python main.py content-stats
```
Show detailed statistics about the markdown content without requiring OpenAI API.

### Vector Search & RAG System

#### Advanced Search with Filters
```bash
# Complex search with multiple filters
python main.py search \
  --query "fund management" \
  --vendor "BlackRock" \
  --date-from "20250701" \
  --date-to "20250731" \
  --has-vendors true \
  --limit 10

# Find AI-related headlines
python main.py search --query "artificial intelligence machine learning AI"

# Find recent partnerships
python main.py search \
  --query "partnership collaboration" \
  --category "Partnerships & Integrations" \
  --year 2025
```

## Example Questions for AI System

Here are example questions you can ask the AI system:

- "What are the recent trends in AI adoption in the private equity industry?"
- "Which companies have made significant personnel changes recently?"
- "What new products or platforms have been launched in the past week?"
- "What are the main categories of headlines in this dataset?"
- "Which companies are mentioned most frequently in partnership announcements?"
- "What types of acquisitions or deal activities have occurred recently?"
- "Are there any notable trends in client acquisitions?"
- "What research or insights have been published recently in the PE space?"
- "How are companies integrating AI into their investment processes?"

## Available Commands

### Main Commands
```bash
# AI Question-Answering System
python main.py ask                           # Interactive Q&A
python main.py ask --question "Your Q"      # Single question
python main.py content-stats                # Content statistics

# Vector Search & RAG System  
python main.py embed                         # Embed headlines
python main.py search --query "search term" # Search headlines
python main.py info                          # Collection info

# LangGraph RAG System
python langgraph_rag.py --interactive        # Interactive LangGraph RAG
python langgraph_rag.py --question "Your Q" # Single question with LangGraph

# Utilities
python setup_qa.py                          # Verify environment
python qa_demo.py                           # Run demo with examples
python simple_example.py                    # Simple QA example
```

### Available Filters (Vector Search)

- **Category**: Filter by headline category (e.g., "New Client", "Personnel / Office")
- **Vendor**: Filter by vendor name (partial match)
- **Year**: Filter by specific year
- **Date Range**: Filter by date range (YYYYMMDD format)
- **Has Vendors**: Filter by whether headline mentions vendors

### CLI Command Details

#### `embed`
Parses the headlines markdown file and embeds it into Qdrant.

```bash
python main.py embed [OPTIONS]

Options:
  --file-path TEXT     Path to headlines markdown file [default: headlines.md]
  --force-recreate     Force recreate collection
  --skip-duplicates    Skip duplicate entries [default: True]
```

#### `search`
Search headlines with semantic similarity and optional metadata filters.

```bash
python main.py search [OPTIONS]

Options:
  --query TEXT         Search query [required]
  --limit INTEGER      Number of results to return [default: 5]
  --category TEXT      Filter by category
  --vendor TEXT        Filter by vendor (partial match)
  --year INTEGER       Filter by year
  --date-from TEXT     Filter from date (YYYYMMDD)
  --date-to TEXT       Filter to date (YYYYMMDD)
  --has-vendors BOOL   Filter by has vendors (true/false)
```

#### `ask`
Ask questions about the entire content using OpenAI GPT-3.5-turbo.

```bash
python main.py ask [OPTIONS]

Options:
  --file-path TEXT     Path to markdown file [default: headlines.md]
  --question TEXT      Single question to ask (optional)
  --model TEXT         OpenAI model to use [default: gpt-3.5-turbo]
  --max-tokens INT     Maximum tokens in response [default: 1500]
  --temperature FLOAT  Temperature for response generation [default: 0.7]
```

## System Architecture

### Vector Search Components
- **HeadlineProcessor**: Parses markdown table format into structured data
- **QdrantEmbedder**: Handles vector embeddings and Qdrant operations
- **SentenceTransformers**: Generates semantic embeddings using `all-MiniLM-L6-v2` model

### QA System Components
1. **MarkdownQASystem**: Main class handling OpenAI interactions
2. **Content Loading**: Reads and processes markdown files
3. **Prompt Engineering**: Creates context-aware prompts for GPT-3.5-turbo
4. **Response Processing**: Formats and displays AI responses

### LangGraph RAG Components
1. **RAGState**: State management for the workflow (question, context, answer)
2. **retrieve_headlines**: Function to search Qdrant for relevant headlines
3. **generate_answer**: Function to generate AI responses using retrieved context
4. **StateGraph**: LangGraph workflow orchestrating retrieve → generate sequence

### Data Flow

**Vector Search**: `Markdown → Parser → Embedder → Qdrant → Search → Results`

**QA System**: `Markdown File → Content Loader → System Prompt → OpenAI API → Formatted Response`

**LangGraph RAG**: `Question → retrieve_headlines (Qdrant) → generate_answer (OpenAI) → Structured Response`

## Data Structure

Each headline is stored with the following metadata:

```json
{
  "date": "20250725",
  "category": "New Client",
  "headline": "Full headline text...",
  "vendors": ["Vendor1", "Vendor2"],
  "year": 2025,
  "month": 7,
  "day": 25,
  "vendor_count": 2,
  "headline_length": 150,
  "has_vendors": true,
  "document_hash": "abc123..."
}
```

### Content Format

The system expects a markdown file with this table structure:

```markdown
| Date | Category | Headline | Vendor(s) |
| --- | --- | --- | --- |
| 20250725 | Personnel / Office | Company announces new hire... | Company A, Company B |
| 20250724 | New Client | Client selects platform... | Platform Corp |
```

## Configuration

### Default Configuration
- **Collection Name**: `pe_headlines`
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Qdrant URL**: `./qdrant_data` (local file storage)
- **Vector Distance**: Cosine similarity
- **OpenAI Model**: `gpt-3.5-turbo`

### QA System Parameters
- `--model`: OpenAI model (default: gpt-3.5-turbo)
- `--max-tokens`: Maximum response length (default: 1500)
- `--temperature`: Response creativity (default: 0.7)

## VS Code Integration

Available tasks in VS Code (press Ctrl/Cmd+Shift+P → "Tasks: Run Task"):
- **Start Qdrant Database**: Starts the vector database
- **Embed Headlines**: Process and embed headlines
- **Search Headlines**: Interactive search
- **Show Collection Info**: Display database stats
- **Run Demo**: Execute demo script
- **Install Dependencies**: Install required packages

## Troubleshooting

### Common Issues

#### Vector Search System
1. **Qdrant Connection Issues**
   - Ensure Qdrant is running: `docker-compose ps`
   - Check Qdrant logs: `docker-compose logs qdrant`
   - Verify port 6333 is available

2. **Memory Issues**
   - The embedding model requires ~50MB RAM
   - Large datasets may need batch processing

#### QA System
1. **"No module named 'openai'"**
   ```bash
   pip install -r requirements.txt
   ```

2. **"OPENAI_API_KEY environment variable is required"**
   ```bash
   python setup_qa.py
   ```
   The API key should already be configured. If this error persists, check that the `.env` file exists and contains the API key.

3. **"File not found: headlines.md"**
   Ensure the headlines.md file is in the current directory.

4. **Large content warnings**
   The system automatically handles large content but may truncate responses for very large files.

### API Limits & Performance

#### OpenAI API Limits
- GPT-3.5-turbo has token limits (~16k context window)
- Very large markdown files may need content splitting
- Rate limiting may apply based on your OpenAI plan

#### Performance Tips
1. **For Large Files**: The system estimates token usage and warns about potential truncation
2. **Interactive Mode**: More efficient for multiple questions vs. single-question mode
3. **Content Caching**: Content is loaded once per session for better performance
4. **Vector Search**: Use `--limit` to control result count and apply filters to reduce search space

## Development

### Project Structure
```
.
├── main.py              # Main application with all commands
├── qa_system.py         # OpenAI QA system implementation
├── setup_qa.py          # Environment setup utility
├── qa_demo.py           # Demo with example questions
├── simple_example.py    # Simple QA example
├── headlines.md         # Source data file
├── docker-compose.yml   # Qdrant setup
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project configuration
└── README.md           # This file
```

### Adding New Features

#### Vector Search System
1. **New Filters**: Add filter conditions in `search_headlines()` method
2. **New Models**: Change the embedding model in `QdrantEmbedder.__init__()`
3. **New Data Sources**: Extend `HeadlineProcessor` to support other formats
4. **New Metadata**: Add fields to `HeadlineEntry.to_dict()` method

#### QA System
1. Extend `MarkdownQASystem` class in `qa_system.py`
2. Add new CLI commands in `main.py`
3. Update prompt engineering for better responses
4. Add new response formatting options

### Testing
```bash
# Test vector search system
python main.py embed && python main.py search --query "test"

# Test QA system
python qa_demo.py --interactive  # Test with demo + interactive mode
python simple_example.py         # Simple test
```

## Example Workflows

### Research Workflow
1. Get overview: `python main.py content-stats`
2. Ask broad questions: `python main.py ask --question "What are the main trends?"`
3. Deep dive with vector search: `python main.py search --query "specific topic"`
4. Follow up with AI: Interactive session for detailed analysis

### Analysis Workflow
1. Embed data: `python main.py embed`
2. Search specific topics: `python main.py search --query "AI" --category "New Product"`
3. Get AI insights: `python main.py ask --question "How are companies using AI?"`
4. Export findings: Use CLI output redirection for reports
