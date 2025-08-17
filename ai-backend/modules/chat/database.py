"""
Chat database setup and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from .models import Base
from config import CONFIG
from core.logger.logger import LOG

# Async engine for async operations
async_engine = create_async_engine(CONFIG.database_url)
AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for worker operations
sync_engine = create_engine(CONFIG.database_url.replace("+asyncpg", ""))
SyncSessionLocal = sessionmaker(bind=sync_engine)

async def setup_chat_database():
    """Create chat tables if they don't exist."""
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        LOG.info("✅ Chat database tables created/verified")
    except Exception as e:
        LOG.error(f"❌ Chat database setup failed: {e}")
        raise
