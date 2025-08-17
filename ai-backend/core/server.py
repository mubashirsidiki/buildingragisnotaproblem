from fastapi import FastAPI
from fastapi.middleware import Middleware
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from core.middlewares.middleware import middleware_handler
from core.logger.logger import LOG

from config import CONFIG
from modules.healthcheck.healthcheck_routes import API_ROUTER
from modules.logviewer.log_viewer_routes import API_ROUTER as LOG_VIEWER_ROUTER
from modules.RAG.routes import API_ROUTER as PDF_RAG_ROUTER
from modules.RAG.database import setup_database, async_engine, sync_engine
from modules.chat.routes import API_ROUTER as CHAT_ROUTER
from modules.chat.database import setup_chat_database
from modules.chat.redis_service import redis_service
from modules.chat.rabbitmq_service import rabbitmq_service
from modules.analytics.routes import router as ANALYTICS_ROUTER

def init_routers(app_: FastAPI) -> None:
    app_.include_router(API_ROUTER)
    app_.include_router(LOG_VIEWER_ROUTER)
    app_.include_router(PDF_RAG_ROUTER)
    app_.include_router(CHAT_ROUTER)
    app_.include_router(ANALYTICS_ROUTER)

def make_middleware() -> list[Middleware]:
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ]
    return middleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    
    # Startup
    LOG.info("🚀 Starting PDF RAG application...")
    
    try:
        # Initialize PDF RAG database
        LOG.info("📊 Setting up PDF RAG database...")
        await setup_database()
        LOG.info("✅ PDF RAG database setup completed successfully")
        
        # Initialize Chat database
        LOG.info("💬 Setting up Chat database...")
        await setup_chat_database()
        LOG.info("✅ Chat database setup completed successfully")
        
        # Initialize Redis connection
        # LOG.info("🔴 Connecting to Redis...")
        # await redis_service.connect()
        # LOG.info("✅ Redis connection established")
        
        # Initialize RabbitMQ connection
        # LOG.info("🐰 Connecting to RabbitMQ...")
        # await rabbitmq_service.connect()
        # LOG.info("✅ RabbitMQ connection established")
        
        # Test database connectivity
        LOG.info("🔍 Testing database connectivity...")
        async with async_engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        LOG.info("✅ Database connectivity verified")
        
        LOG.info("🎉 Application startup completed successfully")
        
    except Exception as e:
        LOG.error(f"❌ Failed to initialize application: {e}")
        raise
    
    # Application is running
    yield
    
    # Shutdown
    LOG.info("🛑 Shutting down PDF RAG application...")
    
    try:
        # Disconnect from Redis
        # LOG.info("🔴 Disconnecting from Redis...")
        # await redis_service.disconnect()
        # LOG.info("✅ Redis disconnected")
        
        # Disconnect from RabbitMQ
        # LOG.info("🐰 Disconnecting from RabbitMQ...")
        # await rabbitmq_service.disconnect()
        # LOG.info("✅ RabbitMQ disconnected")
        
        # Close database connections
        LOG.info("🔌 Closing database connections...")
        await async_engine.dispose()
        sync_engine.dispose()
        LOG.info("✅ Database connections closed successfully")
        
        LOG.info("👋 Application shutdown completed")
        
    except Exception as e:
        LOG.error(f"❌ Error during shutdown: {e}")

def create_app() -> FastAPI:
    app_ = FastAPI(
        title=CONFIG.app_name,
        description=f'{CONFIG.description} - PDF RAG System',
        version=CONFIG.version,
        middleware=make_middleware(),
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )
    init_routers(app_=app_)
    middleware_handler(app=app_)
    return app_

app = create_app()
