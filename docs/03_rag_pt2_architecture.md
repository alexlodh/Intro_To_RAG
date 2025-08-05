# Conversational RAG Module Architecture

This document contains Mermaid.js diagrams illustrating the architecture and flow of the `03_rag_pt2.py` conversational RAG module.

## System Overview

```mermaid
graph TB
    subgraph "External Dependencies"
        OpenAI[OpenAI API<br/>GPT-4o-mini + Embeddings]
        QdrantDB[(Qdrant Vector DB<br/>2500 PE Headlines)]
        ConfigFiles[Config Files<br/>utils/config_loader.py<br/>utils/secrets_loader.py]
    end

    subgraph "Core Module: 03_rag_pt2.py"
        Main[main()]
        VectorStore[setup_vector_store()]
        SelfQuery[create_self_query_retriever()]
        RAGChain[create_conversational_rag_chain()]
        
        subgraph "Tools"
            SearchTool[search_headlines]
            CategoresTool[get_categories]
            YearsTool[get_available_years]
        end
        
        subgraph "LangGraph Workflow"
            Agent[call_model_and_execute_tools]
            Memory[MemorySaver<br/>Thread-based Storage]
        end
        
        subgraph "User Interfaces"
            Interactive[interactive_chat_loop]
            Demo[run_demo_with_tools]
            Test[--test mode]
        end
    end

    ConfigFiles --> Main
    Main --> VectorStore
    Main --> SelfQuery
    Main --> RAGChain
    VectorStore --> QdrantDB
    RAGChain --> Agent
    RAGChain --> Memory
    Agent --> SearchTool
    Agent --> CategoresTool
    Agent --> YearsTool
    SearchTool --> QdrantDB
    Agent --> OpenAI
    Main --> Interactive
    Main --> Demo
    Main --> Test

    style OpenAI fill:#e1f5fe
    style QdrantDB fill:#f3e5f5
    style Memory fill:#e8f5e8
    style Agent fill:#fff3e0
```

## Data Flow Architecture

```mermaid
graph LR
    subgraph "Input Layer"
        User[User Question]
        Thread[Thread ID]
    end
    
    subgraph "Processing Layer"
        State[ConversationalState<br/>- question<br/>- messages<br/>- context<br/>- answer]
        
        subgraph "Agent Processing"
            Model[LLM with Tools<br/>GPT-4o-mini]
            ToolExec[Tool Execution]
            FinalResp[Final Response]
        end
    end
    
    subgraph "Memory Layer"
        MemSaver[MemorySaver<br/>Thread-based<br/>Persistence]
    end
    
    subgraph "Data Layer"
        VecDB[(Vector Database<br/>Qdrant)]
        Tools[Tool Functions<br/>- search_headlines<br/>- get_categories<br/>- get_available_years]
    end
    
    subgraph "Output Layer"
        Answer[Formatted Answer]
        Context[Retrieved Context]
        UpdatedState[Updated Messages]
    end

    User --> State
    Thread --> MemSaver
    State --> Model
    Model --> ToolExec
    ToolExec --> Tools
    Tools --> VecDB
    Tools --> ToolExec
    ToolExec --> FinalResp
    FinalResp --> Answer
    FinalResp --> Context
    FinalResp --> UpdatedState
    UpdatedState --> MemSaver
    MemSaver --> State

    style State fill:#e3f2fd
    style MemSaver fill:#e8f5e8
    style VecDB fill:#f3e5f5
    style Tools fill:#fff3e0
```

## Tool Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent as call_model_and_execute_tools
    participant LLM as GPT-4o-mini
    participant Search as search_headlines
    participant VecDB as Qdrant Vector DB
    participant Memory as MemorySaver

    User->>Agent: Question + Thread ID
    Agent->>Memory: Load conversation history
    Memory-->>Agent: Previous messages
    
    Agent->>LLM: System prompt + History + Current question
    LLM-->>Agent: Response with tool_calls
    
    loop For each tool call
        Agent->>Search: Execute tool with args
        Search->>VecDB: Similarity search + filtering
        VecDB-->>Search: Retrieved documents
        Search-->>Agent: Formatted results
    end
    
    Agent->>LLM: Original messages + Tool results
    LLM-->>Agent: Final answer
    
    Agent->>Memory: Save updated conversation
    Agent-->>User: Answer + Context

    Note over Agent,LLM: Tools: search_headlines,<br/>get_categories, get_available_years
    Note over Search,VecDB: Filtering by category<br/>and year if specified
```

## State Management

```mermaid
stateDiagram-v2
    [*] --> Initialization
    
    Initialization --> VectorStoreSetup: Load config & API keys
    VectorStoreSetup --> DocumentLoading: Create/connect to Qdrant
    DocumentLoading --> DeduplicationCheck: Load markdown documents
    DeduplicationCheck --> CollectionReady: Hash-based deduplication
    
    CollectionReady --> ConversationStart: Vector store ready
    
    ConversationStart --> MessageProcessing: User question received
    
    state MessageProcessing {
        [*] --> SystemPromptAdd
        SystemPromptAdd --> ModelInvocation
        ModelInvocation --> ToolCallCheck
        
        state ToolCallCheck <<choice>>
        ToolCallCheck --> ToolExecution: Has tool calls
        ToolCallCheck --> DirectResponse: No tool calls
        
        state ToolExecution {
            [*] --> SearchHeadlines
            SearchHeadlines --> VectorSearch: query + filters
            VectorSearch --> ResultFormatting
            ResultFormatting --> [*]
        }
        
        ToolExecution --> FinalGeneration
        DirectResponse --> FinalGeneration
        FinalGeneration --> [*]
    }
    
    MessageProcessing --> MemoryUpdate: Response generated
    MemoryUpdate --> ConversationContinue: Save to thread
    
    ConversationContinue --> MessageProcessing: Next question
    ConversationContinue --> [*]: Session end

    note right of ToolExecution : Supports category and year filtering
    note right of MemoryUpdate : Thread-based isolation
```

## Component Relationships

```mermaid
classDiagram
    class ConversationalState {
        +str question
        +Sequence[BaseMessage] messages
        +List[Document] context
        +str answer
    }
    
    class SearchTool {
        +search_headlines(query, category?, year?)
        +get_categories()
        +get_available_years()
    }
    
    class VectorStore {
        +QdrantVectorStore vector_store
        +SelfQueryRetriever self_query_retriever
        +similarity_search()
        +add_documents()
    }
    
    class MemoryManager {
        +MemorySaver memory
        +thread_id management
        +conversation persistence
    }
    
    class LangGraphWorkflow {
        +StateGraph workflow
        +call_model_and_execute_tools()
        +tool execution
    }
    
    class DocumentProcessor {
        +load_md_table_as_documents()
        +deduplication logic
        +metadata extraction
    }

    ConversationalState --> LangGraphWorkflow
    LangGraphWorkflow --> SearchTool
    SearchTool --> VectorStore
    LangGraphWorkflow --> MemoryManager
    VectorStore --> DocumentProcessor
    
    ConversationalState : Used for state management
    SearchTool : Provides search capabilities
    VectorStore : Handles document retrieval
    MemoryManager : Manages conversation threads
    LangGraphWorkflow : Orchestrates the flow
    DocumentProcessor : Loads and processes data
```

## System Interaction Flow

```mermaid
journey
    title User Interaction Journey
    section Initialization
      Load Configuration: 5: System
      Setup Vector Store: 4: System
      Create RAG Chain: 5: System
      
    section First Question
      User asks question: 5: User
      Process with tools: 4: Agent
      Search documents: 5: VectorDB
      Generate answer: 5: LLM
      Return response: 5: User
      
    section Follow-up Questions
      User asks follow-up: 5: User
      Load conversation history: 5: Memory
      Context-aware processing: 5: Agent
      Enhanced search: 4: VectorDB
      Contextual answer: 5: User
      
    section Advanced Features
      Filter by category: 4: User
      Filter by year: 4: User
      Multi-turn conversation: 5: User
      Thread management: 5: System
```

## Error Handling & Fallbacks

```mermaid
flowchart TD
    Start[User Question] --> ModelCall[Call Model with Tools]
    ModelCall --> ToolCheck{Tool Calls Present?}
    
    ToolCheck -->|Yes| ToolExec[Execute Tools]
    ToolCheck -->|No| DirectResp[Direct Response]
    
    ToolExec --> ToolError{Tool Error?}
    ToolError -->|Yes| Fallback[Fallback Search]
    ToolError -->|No| ToolSuccess[Tool Results]
    
    Fallback --> FallbackError{Fallback Error?}
    FallbackError -->|Yes| ErrorMsg[Error Message]
    FallbackError -->|No| FallbackSuccess[Fallback Results]
    
    ToolSuccess --> FinalGen[Final Generation]
    FallbackSuccess --> FinalGen
    DirectResp --> FinalGen
    ErrorMsg --> FinalGen
    
    FinalGen --> GenError{Generation Error?}
    GenError -->|Yes| DefaultError[Default Error Response]
    GenError -->|No| Success[Successful Response]
    
    DefaultError --> End[Return to User]
    Success --> End

    style ToolError fill:#ffebee
    style FallbackError fill:#ffebee
    style GenError fill:#ffebee
    style ErrorMsg fill:#ffcdd2
    style DefaultError fill:#ffcdd2
    style Success fill:#e8f5e8
```

## Key Features Summary

### Architecture Highlights

- **Modular Design**: Clean separation between tools, memory, and processing
- **Thread-based Memory**: Isolated conversation threads for multiple users
- **Tool Integration**: Automatic tool selection and execution
- **Error Resilience**: Multiple fallback mechanisms
- **Scalable**: Support for concurrent conversations

### Technical Stack

- **LangChain**: Core RAG framework
- **LangGraph**: Workflow orchestration
- **Qdrant**: Vector database for document storage
- **OpenAI**: LLM and embedding models
- **Custom Tools**: Domain-specific search and filtering

### Performance Characteristics

- **Setup Time**: ~3 seconds (vector store initialization)
- **Query Processing**: ~6 seconds (tool execution + generation)
- **Memory Overhead**: Minimal (thread-based isolation)
- **Document Capacity**: 2500+ documents with deduplication
