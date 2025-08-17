# AI Backend - FastAPI RAG Service

Production-ready backend service for the RAG system built with FastAPI, PostgreSQL + pgvector, and OpenAI.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL with pgvector extension
- Redis
- RabbitMQ
- OpenAI API key

### Installation
```bash
cd ai-backend
pip install uv
uv sync
```

### Environment Setup
```bash
cp .env.example .env
# Edit .env with your credentials
```

### Development
```bash
uv run python main.py
```
Access at: http://localhost:8000

### Production
```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🏗️ Architecture

```mermaid
flowchart TD
    A[FastAPI App] --> B[PDF Processing]
    A --> C[Vector Search]
    A --> D[Chat Service]
    A --> E[Analytics]
    
    B --> F[Text Extraction]
    F --> G[Semantic Chunking]
    G --> H[OpenAI Embeddings]
    H --> I[pgvector Storage]
    
    C --> J[Query Expansion]
    J --> K[Vector Similarity]
    K --> L[Cross-Encoder Reranking]
    
    D --> M[LLM Integration]
    M --> N[Context Assembly]
    N --> O[Response Generation]
```

## 🔧 Core Services

### PDF RAG Service
- **PDF Processing**: Text extraction and semantic chunking
- **Vector Storage**: pgvector integration for embeddings
- **Search API**: Fast similarity search with reranking
- **Chunking Modes**: Sentence-based (semantic) and length-based

### Chat Service
- **Simple Chat**: Direct LLM conversations
- **RAG Chat**: Context-aware responses with document sources
- **Query Expansion**: Multi-perspective search queries
- **Conversation History**: Persistent chat sessions

### Analytics Service
- **Token Tracking**: OpenAI API usage monitoring
- **Cost Calculation**: Real-time cost tracking
- **Usage Statistics**: User activity and performance metrics

## 📊 API Endpoints

### PDF RAG
- `POST /api/v1/pdf-rag/upload` - Upload and process PDFs
- `POST /api/v1/pdf-rag/search` - Vector search with filters
- `DELETE /api/v1/pdf-rag/clear` - Clear database

### Chat
- `POST /api/v1/chat/simple` - Simple chat without RAG
- `POST /api/v1/chat/rag` - RAG-enhanced chat

### Health & Analytics
- `GET /api/v1/health` - Service health check
- `GET /api/v1/analytics/usage/{user_id}` - User analytics

## 🗄️ Database Schema

```mermaid
erDiagram
    PDFChunk {
        id uuid PK
        filename text
        chunk_text text
        chunk_index int
        page_number int
        start_pos int
        end_pos int
        start_line int
        end_line int
        sentence_count int
        embedding vector
        created_at timestamp
    }
    
    TokenUsage {
        id uuid PK
        user_id text
        operation_type text
        model text
        input_tokens int
        output_tokens int
        estimated_cost decimal
        metadata jsonb
        created_at timestamp
    }
```

## 🔑 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/pdf_rag

# OpenAI
OPENAI_API_KEY=your_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
GPT_MODEL=gpt-4o

# Redis & RabbitMQ
REDIS_URL=redis://localhost:6379/0
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
```

### Chunking Configuration
- **Sentence Mode**: AI-powered semantic chunking (default)
- **Length Mode**: Fixed-size chunks for speed
- **Overlap**: Configurable chunk overlap
- **Thresholds**: Statistical breakpoint detection

## 🐳 Docker

```bash
# Build image
docker build -t rag-backend .

# Run container
docker run -p 8000:8000 rag-backend
```

## 📈 Performance Features

- **Async Processing**: Non-blocking PDF processing
- **Vector Indexing**: Optimized pgvector queries
- **Caching**: Redis-based response caching
- **Queue System**: RabbitMQ for background tasks
- **Connection Pooling**: Database connection optimization

## 🔍 Monitoring & Logging

- **Structured Logging**: JSON-formatted logs
- **Performance Metrics**: Request timing and throughput
- **Error Tracking**: Comprehensive error logging
- **Health Checks**: Service availability monitoring
- **Token Usage**: OpenAI API cost tracking

## 🚀 Deployment

### Production Considerations
- **Gunicorn**: WSGI server for production
- **Environment Variables**: Secure credential management
- **Database Migrations**: Schema versioning
- **Health Checks**: Load balancer integration
- **Monitoring**: Application performance monitoring

### Scaling
- **Horizontal Scaling**: Multiple backend instances
- **Database Sharding**: Partitioned vector storage
- **Load Balancing**: Traffic distribution
- **Caching Layers**: Multi-level caching strategy

## 🔒 Security

- **API Key Validation**: Secure endpoint access
- **Input Sanitization**: XSS and injection prevention
- **Rate Limiting**: Request throttling
- **CORS Configuration**: Cross-origin request handling
- **Environment Isolation**: Secure configuration management

## 🧪 Testing

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=modules

# Run specific module
uv run pytest modules/RAG/
```

## 🔍 Troubleshooting

**Common issues:**
- Database connection errors → Check PostgreSQL and pgvector
- OpenAI API errors → Verify API key and quotas
- Vector search failures → Ensure pgvector extension is installed
- Memory issues → Adjust chunk sizes and batch processing

## 📚 Dependencies

### Core Dependencies
- **FastAPI**: Modern web framework
- **SQLAlchemy**: Database ORM
- **pgvector**: Vector similarity search
- **OpenAI**: AI model integration
- **Redis**: Caching and sessions
- **RabbitMQ**: Message queuing

### Development Dependencies
- **uvicorn**: ASGI server
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting

