from sqlalchemy.orm import sessionmaker, Session
from modules.RAG.database import sync_engine

# Database dependency
SessionLocal = sessionmaker(bind=sync_engine)

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
