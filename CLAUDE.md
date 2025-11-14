# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Exam Trainer Agent is a Streamlit-based multimodal RAG (Retrieval-Augmented Generation) application that processes university notes in PDF format. It extracts text, tables, and formulas using Docling, embeds document chunks using nomic-embed-text-v1.5, stores them in Qdrant vector database, and enables chat-based querying through either local Ollama LLMs or Azure OpenAI GPT-5.

The application follows a two-turn conversation pattern:
1. First turn: generates an open-ended exam question based on retrieved context
2. Second turn: evaluates the user's answer and provides feedback with grading

## Setup Commands

### Prerequisites

#### 1. Install Ollama
**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [https://ollama.com/download](https://ollama.com/download)

#### 2. Pull a Model
```bash
# Recommended models (choose one):
ollama pull llama3.2          # Default - balanced performance (3B params)
ollama pull mistral           # Good alternative (7B params)
ollama pull phi3              # Fast, smaller model (3.8B params)
ollama pull qwen2.5           # Strong multilingual support (7B params)
```

#### 3. Start Ollama Service
Ollama usually auto-starts after installation. To manually start:
```bash
ollama serve
```

#### 4. Start Qdrant Vector Database
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run app.py
```

### Environment Variables
Create a `.env` file with:

#### LLM Provider Selection
- `LLM_PROVIDER` - Choose `'ollama'` (local) or `'azure'` (cloud). Default: `ollama`

#### For Ollama (LLM_PROVIDER=ollama)
- `OLLAMA_BASE_URL` - Default: `http://localhost:11434`
- `OLLAMA_MODEL` - Default: `llama3.2` (options: `llama3.2`, `mistral`, `phi3`, `gemma3:1b`, etc.)

#### For Azure OpenAI (LLM_PROVIDER=azure)
- `AZURE_OPENAI_DEPLOYMENT_NAME` - Your Azure deployment name (e.g., `gpt-5`)
- `AZURE_OPENAI_ENDPOINT` - Your Azure endpoint URL
- `AZURE_OPENAI_API_KEY` - Your Azure API key
- `AZURE_OPENAI_API_VERSION` - API version (e.g., `2025-01-01-preview`)

## Architecture

### Document Processing Pipeline
1. **PDF Upload** → [app.py:60-122](app.py#L60-L122) handles file upload and processing orchestration
2. **Docling Conversion** → [src/utils.py:30-62](src/utils.py#L30-L62) converts PDF to markdown with OCR, table extraction, and formula enrichment
3. **Image Replacement** → [src/utils.py:16-27](src/utils.py#L16-L27) replaces base64 images with text summaries from [src/summaries_images.py](src/summaries_images.py)
4. **Chunking** → [src/chunk_embed.py:8-19](src/chunk_embed.py#L8-L19) tokenizes markdown into 1024-token overlapping chunks (stride=100)
5. **Embedding** → [src/chunk_embed.py:28-50](src/chunk_embed.py#L28-L50) generates embeddings in batches using HuggingFace nomic-embed-text-v1.5
6. **Indexing** → [src/index.py:17-35](src/index.py#L17-L35) creates Qdrant collection with DOT similarity metric

### Retrieval and Generation
- **Retriever** → [src/retriever.py:11-31](src/retriever.py#L11-L31) searches Qdrant with quantization and re-scoring (top_k=7)
- **RAG Engine** → [src/rag_engine.py:102-156](src/rag_engine.py#L102-L156) manages conversation state and prompt templates
  - Uses `self.last_question` to track conversation state
  - First query triggers question generation via `qa_prompt_tmpl_str`
  - Second query triggers answer evaluation via `evaluation_prompt`
  - Resets state after evaluation completes

### Key Components
- **QdrantVDB** ([src/index.py](src/index.py)): Vector database wrapper that recreates collection on each ingestion to avoid stale data
- **EmbedData** ([src/chunk_embed.py:28-50](src/chunk_embed.py#L28-L50)): Manages embedding model and batch processing
- **RAG** ([src/rag_engine.py](src/rag_engine.py)): Core RAG logic with flexible LLM backend (Ollama or Azure OpenAI) and conversation management

### Caching Behavior
- Embedding files are saved as `embeddings_{pdf_name}.pkl` in root directory
- [app.py:79-104](app.py#L79-L104) checks for existing embeddings before regenerating
- Session state uses `file_cache` to avoid reprocessing same file within session

### Model Configuration
- Embedding model: `nomic-ai/nomic-embed-text-v1.5` (768 dimensions, cached in `./hf_cache`)
- LLM: Flexible backend configured in [src/rag_engine.py:12-20](src/rag_engine.py#L12-L20)
  - **Ollama (local)**: Default `llama3.2`, configurable via `OLLAMA_MODEL` env var
    - Supports: llama3.2, mistral, phi3, gemma3:1b, qwen2.5, etc.
  - **Azure OpenAI (cloud)**: GPT-5 or other deployments via `AZURE_OPENAI_DEPLOYMENT_NAME`
  - Switch between providers using `LLM_PROVIDER` environment variable
- Vector similarity: DOT product with quantization and 2.0x oversampling

## Important Notes

- **Qdrant** must be running on `localhost:6333` before app startup
- **For Ollama**: Service must be running with a model pulled before app startup (set `LLM_PROVIDER=ollama`)
- **For Azure OpenAI**: Valid API credentials required in `.env` file (set `LLM_PROVIDER=azure`)
- Collection is recreated on each document upload to prevent stale data
- Images are replaced with summaries to reduce context size and improve embedding quality
- The app uses `nest_asyncio.apply()` to handle async operations in Streamlit
- Prompt templates use LaTeX formatting (`$...$` for inline, `$$...$$` for block math)
- Switch LLM providers anytime by changing `LLM_PROVIDER` in `.env` and restarting the app
