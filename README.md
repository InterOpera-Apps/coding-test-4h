# Multimodal Document Chat System - Coding Test

## Project Overview

Build a system that allows users to upload PDF documents, extract text, images, and tables, and engage in multimodal chat based on the extracted content.

### Core Features
1. **Document Processing**: PDF parsing using Docling (extract text, images, tables)
2. **Vector Store**: Store extracted content in vector database
3. **Multimodal Chat**: Provide answers with related images/tables for text questions
4. **Multi-turn Conversation**: Maintain conversation context for continuous questioning

---

## Provided Components (Starting Point)

The following items are **already implemented and provided**:

### Infrastructure Setup
- Docker Compose configuration (PostgreSQL+pgvector, Redis, Backend, Frontend)
- Database schema and models (SQLAlchemy)
- API base structure (FastAPI)
- Frontend base structure (Next.js + TailwindCSS)

### Database Models
- `Document` - Uploaded document information
- `DocumentChunk` - Text chunks (with vector embeddings)
- `DocumentImage` - Extracted images
- `DocumentTable` - Extracted tables
- `Conversation` - Chat sessions
- `Message` - Chat messages

### API Endpoints (Skeleton provided)
- `POST /api/documents/upload` - Upload document
- `GET /api/documents` - List documents
- `GET /api/documents/{id}` - Document details
- `DELETE /api/documents/{id}` - Delete document
- `POST /api/chat` - Send chat message
- `GET /api/conversations` - List conversations
- `GET /api/conversations/{id}` - Get conversation history

### Frontend Pages (Layout only)
- `/` - Home (document list)
- `/upload` - Document upload
- `/chat` - Chat interface
- `/documents/[id]` - Document details

### Development Tools
- FastAPI Swagger UI (`http://localhost:8000/docs`)
- Hot reload (Backend & Frontend)
- Environment configuration

---

## Core Features to Implement (Your Job)

You need to implement the following **3 core features**:

### 1. Document Processing Pipeline (Critical)

**Location**: `backend/app/services/document_processor.py`

**Requirements**:
```python
class DocumentProcessor:
    async def process_document(self, file_path: str, document_id: int) -> Dict[str, Any]:
        """
        Process a PDF document to extract text, images, and tables.
        
        Implementation steps:
        1. Parse PDF using Docling
        2. Extract and chunk text (for vector storage)
        3. Extract and save images (filesystem + DB)
        4. Extract and save tables (structured data + image)
        5. Error handling and status updates
        
        Returns:
            {
                "status": "success",
                "text_chunks": 50,
                "images": 10,
                "tables": 5,
                "processing_time": 12.5
            }
        """
        pass
```

**Evaluation Criteria**:
- Docling integration and PDF parsing accuracy
- Image extraction and storage (filename, path, metadata)
- Table extraction (preserve structure, render as image)
- Text chunking strategy (chunk size, overlap)
- Error handling (invalid PDF, memory overflow, etc.)

---

### 2. Vector Store Integration (Critical)

**Location**: `backend/app/services/vector_store.py`

**Requirements**:
```python
class VectorStore:
    async def store_text_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        document_id: int
    ) -> int:
        """
        Store text chunks with vector embeddings.
        
        Implementation steps:
        1. Generate embeddings using OpenAI/HuggingFace
        2. Store in pgvector (vector + metadata)
        3. Include image/table references in metadata
        
        Returns:
            Number of stored chunks
        """
        pass
    
    async def search_similar(
        self, 
        query: str, 
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query.
        
        Returns:
            [
                {
                    "content": "...",
                    "score": 0.95,
                    "metadata": {...},
                    "related_images": [...],
                    "related_tables": [...]
                }
            ]
        """
        pass
```

**Evaluation Criteria**:
- Embedding model selection and integration
- pgvector utilization (cosine similarity, indexing)
- Metadata management (image/table references)
- Search accuracy and performance

---

### 3. Multimodal Chat Engine (Critical)

**Location**: `backend/app/services/chat_engine.py`

**Requirements**:
```python
class ChatEngine:
    async def process_message(
        self,
        conversation_id: int,
        message: str,
        document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process user message and generate multimodal response.
        
        Implementation steps:
        1. Load conversation history (multi-turn support)
        2. Find relevant context using vector search
        3. Find related images/tables
        4. Generate answer using LLM
        5. Include image/table URLs in response
        
        Returns:
            {
                "answer": "...",
                "sources": [
                    {
                        "type": "text",
                        "content": "...",
                        "score": 0.95
                    },
                    {
                        "type": "image",
                        "url": "/uploads/images/abc123.png",
                        "caption": "Figure 1: ..."
                    },
                    {
                        "type": "table",
                        "url": "/uploads/tables/xyz789.png",
                        "caption": "Table 1: ..."
                    }
                ],
                "processing_time": 2.5
            }
        """
        pass
```

**Evaluation Criteria**:
- RAG implementation quality (relevance, accuracy)
- Multi-turn conversation support (context maintenance)
- Include images/tables in responses
- LLM prompt engineering
- Response speed and user experience

---

## System Architecture

```
┌─────────────┐
│   Frontend  │ (Next.js)
│  Chat UI    │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────┐
│   Backend   │ (FastAPI)
│  API Server │
└──────┬──────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│  Document   │   │    Chat     │
│  Processor  │   │   Engine    │
│  (Docling)  │   │   (RAG)     │
└──────┬──────┘   └──────┬──────┘
       │                 │
       ▼                 ▼
┌─────────────────────────────┐
│      Vector Store           │
│    (PostgreSQL+pgvector)    │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│    File Storage             │
│  (Images, Tables, PDFs)     │
└─────────────────────────────┘
```

---

## Data Models

### Document
```python
class Document:
    id: int
    filename: str
    file_path: str
    upload_date: datetime
    processing_status: str  # 'pending', 'processing', 'completed', 'error'
    total_pages: int
    text_chunks_count: int
    images_count: int
    tables_count: int
```

### DocumentChunk
```python
class DocumentChunk:
    id: int
    document_id: int
    content: str
    embedding: Vector(1536)  # pgvector
    page_number: int
    chunk_index: int
    metadata: JSON  # {related_images: [...], related_tables: [...]}
```

### DocumentImage
```python
class DocumentImage:
    id: int
    document_id: int
    file_path: str
    page_number: int
    caption: str
    width: int
    height: int
```

### DocumentTable
```python
class DocumentTable:
    id: int
    document_id: int
    image_path: str  # Rendered table as image
    data: JSON  # Structured table data
    page_number: int
    caption: str
```

### Conversation & Message
```python
class Conversation:
    id: int
    title: str
    created_at: datetime
    document_id: Optional[int]  # Conversation about specific document

class Message:
    id: int
    conversation_id: int
    role: str  # 'user', 'assistant'
    content: str
    sources: JSON  # Sources used in answer (text, images, tables)
    created_at: datetime
```

---

## Tech Stack

### Backend
- **Framework**: FastAPI
- **PDF Processing**: Docling
- **Vector DB**: PostgreSQL + pgvector
- **Embeddings**: OpenAI API or HuggingFace
- **LLM**: OpenAI GPT-4o-mini or Ollama
- **Task Queue**: Celery + Redis (optional)

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Styling**: TailwindCSS
- **UI Components**: shadcn/ui
- **State Management**: React Hooks
- **API Client**: fetch/axios

### Infrastructure
- **Database**: PostgreSQL 15 + pgvector
- **Cache**: Redis
- **Container**: Docker + Docker Compose

---

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Node.js 18+
- Python 3.11+
- OpenAI API Key (or Ollama for local LLM)

### Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd coding-test-4th

# 2. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Start all services
docker-compose up -d

# 4. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## Evaluation Criteria (100 points)

### 1. Code Quality (25 points)
- **Structure** (10 points): Module separation, responsibility separation, reusability
- **Readability** (8 points): Naming, comments, code style
- **Error Handling** (7 points): Exception handling, error messages, recovery strategy

### 2. Feature Implementation (40 points)
- **Document Processing** (15 points):
  - Docling integration and PDF parsing (5 points)
  - Image extraction and storage (5 points)
  - Table extraction and storage (5 points)

- **Vector Store** (10 points):
  - Embedding generation and storage (5 points)
  - Similarity search accuracy (5 points)

- **Chat Engine** (15 points):
  - RAG implementation quality (5 points)
  - Multimodal responses (images/tables included) (5 points)
  - Multi-turn conversation support (5 points)

### 3. UX/UI (15 points)
- **Chat Interface** (8 points): Intuitiveness, responsiveness, image/table display
- **Document Upload/Management** (4 points): Progress indication, error display
- **Design** (3 points): Consistency, aesthetics

### 4. Documentation (10 points)
- **README** (4 points): Installation, execution, feature explanation
- **Code Comments** (3 points): Complex logic explanation
- **API Documentation** (3 points): Swagger or separate documentation

### 5. Testing (10 points)
- **Unit Tests** (5 points): Core logic testing
- **Integration Tests** (3 points): API endpoint testing
- **Test Coverage** (2 points): 60% or higher

---

## Bonus Points (+20 points)

- **Advanced PDF Processing** (+5 points): OCR, complex layout handling
- **Multi-document Search** (+5 points): Search across multiple documents
- **Real-time Chat** (+5 points): WebSocket-based
- **Deployment** (+5 points): Production deployment setup (Railway, Vercel, etc.)

---

## Submission Requirements

### Required Items
1. **Source Code** (complete project)
2. **README.md** (execution instructions, implementation details)
3. **Screenshots** (minimum 5):
   - Document upload screen
   - Document processing completion screen
   - Chat interface
   - Answer example with images
   - Answer example with tables
4. **Sample PDF** (PDF file used for testing)

### Optional Items
- Test results report
- Performance benchmark results
- Deployment URL (if available)

---

## Test Scenarios

### Scenario 1: Basic Document Processing
1. Upload a technical paper PDF
2. Verify text, images, and tables extraction
3. Check extracted content on document detail page

### Scenario 2: Text-based Question
1. Ask "What is the main conclusion of this paper?"
2. Verify answer is generated with relevant text context

### Scenario 3: Image-related Question
1. Ask "Show me the architecture diagram"
2. Verify related images are displayed in chat

### Scenario 4: Table-related Question
1. Ask "What are the experimental results?"
2. Verify related tables are displayed in chat

### Scenario 5: Multi-turn Conversation
1. First question: "What is the dataset used?"
2. Follow-up: "How many samples does it contain?"
3. Verify previous conversation context is maintained

---

## Sample PDF

A sample PDF file is provided: `1706.03762v7.pdf`

This is a technical paper ("Attention Is All You Need") with:
- Multiple pages with text content
- Diagrams and architecture figures
- Tables with experimental results
- Complex layouts for testing

You should use this PDF to test your implementation.

---

## Implementation Guidelines

Refer to the service skeleton files for detailed implementation guidance:
- `backend/app/services/document_processor.py` - Document processing guidelines
- `backend/app/services/vector_store.py` - Vector store implementation tips
- `backend/app/services/chat_engine.py` - Chat engine implementation tips

Each file contains detailed TODO comments with implementation hints and examples.

---

## FAQ

**Q: Docling won't install.**
A: Try `pip install docling` or use the Docker image.

**Q: I don't have an OpenAI API key.**
A: You can install Ollama locally and use a free LLM.

**Q: Where should I save images?**
A: Save to `backend/uploads/images/` directory and store only the path in DB.

**Q: How should I display tables?**
A: Render tables as images or display JSON data as HTML tables in frontend.

---

## Questions?

If you have any questions, please create an issue or contact us via email.

Good luck!
