# LangChain AI Agent with RAG System

<img width="1839" height="853" alt="image" src="https://github.com/user-attachments/assets/1654475e-7d08-4a8d-a78a-7b0b38937bf7" />
<img width="1840" height="876" alt="image" src="https://github.com/user-attachments/assets/8032b75d-af05-406d-b5e2-264771ba1772" />



A powerful, offline-capable AI chatbot system that combines LangChain, RAG (Retrieval-Augmented Generation), and local LLMs through Ollama to provide intelligent, document-aware responses.

## ✨ Features

###  Core Features
- **Intelligent Mode Selection**: Automatically chooses between RAG, Hybrid, or Direct mode based on query context
- **Document-Aware Responses**: Upload documents (PDF, DOCX, TXT) and ask questions about their content
- **Offline Operation**: Runs completely locally with no internet connection required
- **Multiple LLM Support**: Compatible with any Ollama model (llama2, mistral, phi, etc.)
- **Source Citations**: Transparent source attribution for document-based answers
- **Conversation Memory**: Maintains context across the conversation
- **Real-time Processing**: Live document processing with progress indicators

### Privacy & Security
- **100% Local**: All processing happens on your machine
- **No Data Upload**: Documents never leave your system
- **No API Keys**: No external API dependencies
- **Offline Capable**: Works without internet connection

### Advanced Capabilities
- **Vector Similarity Search**: Fast semantic search using ChromaDB
- **Smart Text Chunking**: Optimized 500-word chunks with 50-word overlap
- **Hybrid Retrieval**: Falls back gracefully when documents lack context
- **Multi-format Support**: Handles TXT, PDF, and DOCX files
- **Session Persistence**: Maintains conversation and document state

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT UI                         │
│  [Chat Interface] [File Upload] [Settings] [History]   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│              DECISION ENGINE (AI Agent)                 │
│  Analyzes Query → Selects Mode → Routes Request        │
└────┬──────────────┬──────────────┬─────────────────────┘
     │              │              │
     ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌──────────┐
│   RAG   │  │  HYBRID  │  │  DIRECT  │
│   MODE  │  │   MODE   │  │   MODE   │
└────┬────┘  └────┬─────┘  └────┬─────┘
     │            │             │
     ▼            ▼             │
┌─────────────────────────┐    │
│   VECTOR STORE          │    │
│   (ChromaDB)            │    │
│   - Embeddings          │    │
│   - HNSW Index          │    │
│   - Metadata            │    │
└─────────┬───────────────┘    │
          │                    │
          ▼                    ▼
     ┌────────────────────────────┐
     │      OLLAMA LLM            │
     │   (Local Model Server)     │
     └────────────┬───────────────┘
                  │
                  ▼
            [RESPONSE]
```

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| **Python** | 3.8+ | Runtime environment |
| **Ollama** | Latest | Local LLM server |
| **Git** | Any | Clone repository (optional) |

### System Requirements

- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 5GB+ for models
- **CPU**: Multi-core recommended (GPU optional)
- **OS**: Windows, macOS, or Linux

---

##  Installation

### Step 1: Install Ollama

#### Windows/Mac:
```bash
# Download from: https://ollama.ai/
# Install and run the installer
```

### Step 2: Install Python Dependencies

#### Option A: Using pip (Recommended)
```bash
# Clone or create project directory
mkdir langchain-rag-agent
cd langchain-rag-agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install streamlit langchain langchain-community chromadb sentence-transformers pypdf2 python-docx ollama
```

#### Option B: Using requirements.txt
```bash
# Create requirements.txt with:
streamlit==1.28.0
langchain==0.1.0
langchain-community==0.0.10
chromadb==0.4.18
sentence-transformers==2.2.2
pypdf2==3.0.1
python-docx==1.1.0
ollama==0.1.6

# Install
pip install -r requirements.txt
```

### Step 3: Download LLM Model

```bash
# Start Ollama server (keep this running)
ollama serve

# In a new terminal, pull a model
ollama pull llama2        # Recommended for beginners (3.8GB)
# OR
ollama pull mistral       # Fast and efficient (4.1GB)
# OR
ollama pull phi          # Lightweight (1.6GB)
```

### Step 4: Verify Installation

```bash
# Check Ollama is running
curl http://localhost:11434

# Check installed models
ollama list

# Check Python packages
pip list | grep -E "streamlit|langchain|chromadb"
```

---

## Quick Start

### Basic Usage (5 Minutes)

1. **Start Ollama Server**
   ```bash
   ollama serve
   ```

2. **Run the Application**
   ```bash
   # In a new terminal with activated venv
   streamlit run langchain_agent_simple.py
   ```

3. **Open Browser**
   - Automatically opens at `http://localhost:8501`
   - If not, manually navigate to the URL

4. **Start Chatting**
   - Type a question: "What is Python?"
   - Watch the system respond!

### With Documents (10 Minutes)

1. **Prepare a Test Document**
   ```bash
   # Create sample_resume.txt
   echo "John Doe
   Skills: Python, Machine Learning, Data Analysis
   Experience: 5 years in AI development
   Education: MS in Computer Science" > sample_resume.txt
   ```

2. **Upload Document**
   - Click "Browse files" in sidebar
   - Select `sample_resume.txt`
   - Click " Process Documents"
   - Wait for " Added 1 documents"

3. **Ask Document-Specific Questions**
   - "What skills does John have?"
   - "How many years of experience?"
   - "What is John's education?"

4. **Watch the Magic**
   - System uses **RAG MODE**
   - Shows sources used
   - Provides accurate answers

---

##  Usage Guide

### Document Upload

#### Supported Formats
- **TXT**: Plain text files (UTF-8)
- **PDF**: Adobe PDF documents
- **DOCX**: Microsoft Word documents

#### Upload Process
```
1. Click "Browse files" → Select file(s)
2. Click "Process Documents" → Wait for processing
3. See "Documents Loaded: X" → Ready to query
```

#### File Size Limits
- Maximum: 200MB per file
- Recommended: < 10MB for best performance
- Large files take longer to process

### Querying the System

#### Example Queries by Mode

**RAG Mode Triggers:**
```
 "What skills are mentioned in my resume?"
 "Based on the document, what are the key points?"
 "According to the uploaded file, what is..."
 "What does my resume say about experience?"
```

**Direct Mode Triggers:**
```
 "What is machine learning?"
 "Explain Python programming"
 "How does AI work?"
 "Tell me about neural networks"
```

**Hybrid Mode Triggers:**
```
 "What should I learn next?" (has docs + question)
 "How can I improve?" (general + personal context)
 "What are my strengths?" (ambiguous)
```

### Configuration Options

#### Model Selection
```python
# In sidebar: Change "Model Name"
# Supported: llama2, mistral, phi, codellama, etc.
# Restart required after change
```

#### Chunk Settings (Advanced)
```python
# Edit in code: langchain_agent_simple.py
chunk_size = 500      # Words per chunk
chunk_overlap = 50    # Overlap between chunks
n_results = 3         # Number of chunks to retrieve
```

---

##  File Structure

```
langchain-rag-agent/
│
├── langchain_agent_rag.py          # Full ReAct agent version
├── langchain_agent_simple.py       # Simplified reliable version
├── chatbox_streamlit.py            # Basic chat (no RAG)
├── chatbox_desktop.py              # Tkinter desktop version
├── app.py                          # Advanced RAG version
├── setup.py                        # Automated setup script
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── venv/                           # Virtual environment (created)
├── chroma_simple_db/              # Vector database (auto-created)
├── chroma_langchain_db/           # Alternative DB (auto-created)
│
└── sample_documents/              # Test documents (optional)
    ├── sample_resume.txt
    ├── test_document.pdf
    └── example.docx
```

---

## Configuration

### Environment Variables

```bash
# Optional: Set Ollama host
export OLLAMA_HOST=http://localhost:11434

# Optional: Enable GPU (if available)
export OLLAMA_GPU=1
```

### Model Configuration

```python
# In the code (langchain_agent_simple.py)
st.session_state.model_name = "llama2"  # Change to your model

# Available models:
# - llama2 (3.8GB) - Balanced performance
# - mistral (4.1GB) - Fast and efficient  
# - phi (1.6GB) - Lightweight
# - codellama (3.8GB) - Best for code
```

### Advanced Settings

```python
# Text Splitter Configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # Adjust chunk size
    chunk_overlap=50,        # Adjust overlap
    length_function=len,
)

# Retrieval Configuration
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}   # Number of chunks to retrieve
)

# LLM Configuration
llm = Ollama(
    model="llama2",
    temperature=0.7,         # Creativity (0-1)
    num_ctx=2048,           # Context window
)
```

---

## Troubleshooting

### Common Issues

#### 1. "Connection refused" Error

**Problem:** Ollama server not running

**Solution:**
```bash
# Start Ollama in a new terminal
ollama serve

# Verify it's running
curl http://localhost:11434
```

#### 2. "Model not found" Error

**Problem:** Model not downloaded

**Solution:**
```bash
# List available models
ollama list

# Pull the required model
ollama pull llama2
```

#### 3. "Module not found" Error

**Problem:** Missing Python packages

**Solution:**
```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 4. "Port already in use" Error

**Problem:** Port 8501 is occupied

**Solution:**
```bash
# Run on different port
streamlit run langchain_agent_simple.py --server.port 8502
```

#### 5. Slow Performance

**Solutions:**
- Use lighter model: `ollama pull phi`
- Reduce chunk size: `chunk_size=300`
- Reduce retrieval: `search_kwargs={"k": 2}`
- Close other applications
- Enable GPU if available

#### 6. "Agent iteration limit" Error

**Problem:** Original agent version hitting limits

**Solution:**
```bash
# Use the simplified version instead
streamlit run langchain_agent_simple.py
```

### Debug Mode

```bash
# Enable verbose logging
streamlit run langchain_agent_simple.py --logger.level=debug

# Check Ollama logs
ollama logs
```

### Getting Help

1. Check the [flowcharts](#) for visual understanding
2. Review the error message carefully
3. Ensure all prerequisites are installed
4. Verify Ollama is running: `ollama list`
5. Check Python environment: `pip list`

---

##  Performance

### Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Document Upload (1MB) | 2-5s | Depends on file type |
| Embedding Generation | 1-3s | Per document |
| Vector Search | <100ms | HNSW index |
| LLM Response (Direct) | 2-5s | Model dependent |
| LLM Response (RAG) | 3-8s | Includes search |

### Optimization Tips

1. **Use Appropriate Models**
   - Small tasks: `phi` (1.6GB)
   - General use: `llama2` (3.8GB)
   - Complex tasks: `mistral` (4.1GB)

2. **Optimize Chunking**
   ```python
   # For long documents
   chunk_size = 800, chunk_overlap = 100
   
   # For short documents
   chunk_size = 300, chunk_overlap = 30
   ```

3. **GPU Acceleration**
   ```bash
   # Enable GPU (if available)
   export OLLAMA_GPU=1
   ```

4. **Reduce Retrieval**
   ```python
   # Faster but less context
   search_kwargs={"k": 2}  # Instead of 3
   ```

---

##  Contributing

Contributions are welcome! Here's how you can help:

### Reporting Bugs

1. Check existing issues
2. Create detailed bug report
3. Include error messages and logs
4. Specify OS and Python version

### Suggesting Features

1. Open an issue with [FEATURE] tag
2. Describe the use case
3. Explain expected behavior
4. Provide examples

### Code Contributions

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone <your-fork-url>
cd langchain-rag-agent

# Create dev environment
python -m venv venv-dev
source venv-dev/bin/activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

##  Acknowledgments

- **LangChain**: For the RAG framework
- **Ollama**: For local LLM infrastructure
- **ChromaDB**: For vector database
- **Streamlit**: For the UI framework
- **HuggingFace**: For embeddings models

---


## Version History

### v1.0.0 (Current)
- Initial release
- Three-mode system (RAG/Hybrid/Direct)
- PDF, DOCX, TXT support
- ChromaDB integration
- Streamlit UI

### v0.9.0 (Beta)
- RAG implementation
- Document processing
- Vector search

### v0.5.0 (Alpha)
- Basic chatbot
- Ollama integration

---




