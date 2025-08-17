from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from .models import Base
from config import CONFIG

# Create async engine for database operations
async_engine = create_async_engine(
    CONFIG.database_url,
    echo=False
)

# Create sync engine for setup operations (convert async URL to sync)
sync_database_url = CONFIG.database_url
if "+asyncpg" in sync_database_url:
    sync_database_url = sync_database_url.replace("+asyncpg", "")
elif "+psycopg" in sync_database_url:
    sync_database_url = sync_database_url.replace("+psycopg", "")

sync_engine = create_engine(
    sync_database_url,
    echo=False
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Create sync session factory for setup
SessionLocal = sessionmaker(
    bind=sync_engine,
    expire_on_commit=False
)

async def setup_database():
    """Set up database tables for PDF RAG system."""
    with sync_engine.connect() as conn:
        try:
            # Ensure pgvector extension is enabled
            print("🔧 Ensuring pgvector extension is enabled...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            print("✅ pgvector extension enabled")
            
            # Create tables (this will create them if they don't exist)
            print("🔧 Creating database tables...")
            Base.metadata.create_all(bind=sync_engine)
            
            # Also create analytics tables
            from modules.analytics.models import Base as AnalyticsBase
            AnalyticsBase.metadata.create_all(bind=sync_engine)
            print("✅ Database tables created")
            
            # Create HNSW index for embeddings if it doesn't exist
            print("🔧 Creating vector index...")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS pdf_chunks_embedding_idx 
                ON pdf_chunks USING hnsw (embedding vector_cosine_ops)
            """))
            conn.commit()
            print("✅ Vector index created")
            
        except Exception as e:
            print(f"❌ Error during database setup: {e}")
            print("\nPossible solutions:")
            print("1. Make sure PostgreSQL is running")
            print("2. Check your database credentials")
            print("3. Ensure the 'vector' extension is installed in your PostgreSQL instance")
            print("4. Try running: CREATE EXTENSION vector; manually in psql if it's not installed")
            raise 