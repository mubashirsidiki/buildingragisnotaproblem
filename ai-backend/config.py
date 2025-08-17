import tomllib
from os import getenv
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

with open("pyproject.toml", "rb") as f:
    toml_object: dict[str] = tomllib.load(f).get("project", {})
    name = toml_object.get("name")
    description = toml_object.get("description")
    version = toml_object.get("version")

class ConfigClass(BaseModel):
    app_name: str
    description: str
    version: str
    api_key: Optional[str]
    
    # Database configuration
    database_url: str = getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/pdf_rag")
    
    # OpenAI configuration
    openai_api_key: str = getenv("OPENAI_API_KEY", "")
    openai_embedding_model: str = getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Chat configuration
    gpt_model: str = getenv("GPT_MODEL", "gpt-4o")
    default_user_id: str = getenv("DEFAULT_USER_ID", "anonymous")
    
    # Redis configuration
    redis_url: str = getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_ttl: int = int(getenv("REDIS_TTL", "3600"))  # 1 hour
    
    # RabbitMQ configuration
    rabbitmq_url: str = getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    
    # Token cost configuration (per 1M tokens in USD)
    gpt4o_input_tokens_cost: float = float(getenv("GPT4O_INPUT_TOKENS_COST", "2.50"))
    gpt4o_output_tokens_cost: float = float(getenv("GPT4O_OUTPUT_TOKENS_COST", "10.00"))

CONFIG = ConfigClass(
    app_name = name,
    description = description,
    version = version,
    api_key = getenv("API_KEY") if getenv("API_KEY") else None
)