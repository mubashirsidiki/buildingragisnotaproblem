# PDF RAG + Chat API

## 🚀 Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Set Environment Variables
Create `.env` file:
```env
DATABASE_URL=postgresql+asyncpg://postgres:admin@localhost:5432/pdf_rag
OPENAI_API_KEY=your_openai_api_key_here
GPT_MODEL=gpt-3.5-turbo
DEFAULT_USER_ID=anonymous
REDIS_URL=redis://localhost:6379/0
REDIS_TTL=3600
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
```

### 3. Start Services

**Redis:**
```bash
redis-server
```

**RabbitMQ:**
```bash
docker run -d --name rabbitmq -p 5672:5672 rabbitmq:3
```

**PostgreSQL:**
```bash
docker run -d --name postgres -p 5432:5432 -e POSTGRES_PASSWORD=admin postgres:15
```

### 4. Start System

**Terminal 1 - Worker:**
```bash
uv run run_chat_worker.py
```

**Terminal 2 - Server:**
```bash
uv run main.py
```

### 5. Test

**Simple Chat (No Redis/RabbitMQ):**
```bash
curl -X POST "http://localhost:8000/v1/chat/simple" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Hello!",
       "user_id": "user123",
       "conversation_history": [
         {"role": "user", "content": "Hi there"},
         {"role": "assistant", "content": "Hello! How can I help you today?"}
       ]
     }'
```

**RAG Chat (With Function Calling):**
```bash
curl -X POST "http://localhost:8000/v1/chat/rag" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "I need information from knowledge base about machine learning",
       "user_id": "user123",
       "conversation_history": []
     }'
```

**Full Chat (With Redis/RabbitMQ):**
```bash
curl -X POST "http://localhost:8000/v1/chat/" \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello!", "user_id": "user123"}'
```

**PDF Upload:**
```bash
curl -X POST "http://localhost:8000/v1/pdf-rag/upload" \
     -F "file=@document.pdf"
```

**PDF Search:**
```bash
curl -X POST "http://localhost:8000/v1/pdf-rag/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?"}'
```

## 🛑 Stop
```bash
# Ctrl+C in terminals
docker stop redis rabbitmq postgres
```

