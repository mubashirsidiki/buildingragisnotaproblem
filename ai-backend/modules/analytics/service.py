import json
from typing import Dict, List, Optional
from datetime import datetime
from uuid import uuid4
from sqlalchemy.orm import Session
from .models import TokenUsage
from config import CONFIG

# OpenAI Pricing (per 1M tokens) - Latest GPT-4o pricing from environment variables
def get_pricing_config():
    from os import getenv
    
    return {
        # GPT-4o (Main model) - Latest pricing: $2.50 input, $10.00 output per 1M tokens
        "gpt-4o": {
            "input": float(getenv("GPT4O_INPUT_TOKENS_COST", "2.50")), 
            "output": float(getenv("GPT4O_OUTPUT_TOKENS_COST", "10.00"))
        },
        
        # Embedding Models (input only, no output tokens)
        "text-embedding-3-small": {
            "input": float(getenv("TEXT_EMBEDDING_3_SMALL_COST", "0.02")), 
            "output": 0.00
        },
        "text-embedding-3-large": {
            "input": float(getenv("TEXT_EMBEDDING_3_LARGE_COST", "0.13")), 
            "output": 0.00
        },
        "text-embedding-ada-002": {
            "input": float(getenv("TEXT_EMBEDDING_ADA_002_COST", "0.10")), 
            "output": 0.00
        },
        
        # Aliases for common model variations - all use GPT-4o pricing
        "gpt-4o-mini": {"input": float(getenv("GPT4O_INPUT_TOKENS_COST", "2.50")), "output": float(getenv("GPT4O_OUTPUT_TOKENS_COST", "10.00"))},
        "gpt-3.5-turbo": {"input": float(getenv("GPT4O_INPUT_TOKENS_COST", "2.50")), "output": float(getenv("GPT4O_OUTPUT_TOKENS_COST", "10.00"))},
        "gpt-4": {"input": float(getenv("GPT4O_INPUT_TOKENS_COST", "2.50")), "output": float(getenv("GPT4O_OUTPUT_TOKENS_COST", "10.00"))},
    }

class TokenTrackingService:
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage and model pricing"""
        pricing_config = get_pricing_config()
        if model not in pricing_config:
            model = "gpt-4o"  # fallback to main GPT-4o model
        
        pricing = pricing_config[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)
    
    def track_usage(self, 
                   user_id: str, 
                   operation_type: str, 
                   model: str,
                   input_tokens: int, 
                   output_tokens: int,
                   metadata: Optional[Dict] = None) -> TokenUsage:
        """Track token usage for an operation"""
        total_tokens = input_tokens + output_tokens
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        usage = TokenUsage(
            id=str(uuid4()),
            user_id=user_id,
            operation_type=operation_type,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=cost,
            request_data=json.dumps(metadata) if metadata else None
        )
        
        self.db.add(usage)
        self.db.commit()
        return usage
    

    
    def get_user_analytics(self, user_id: str) -> Dict:
        """Get analytics summary for a user"""
        from core.logger.logger import LOG
        
        # Debug: Check what's in the database
        all_usages = self.db.query(TokenUsage).all()
        LOG.info(f"🔍 Database Debug - Total records: {len(all_usages)}")
        LOG.info(f"   ┌─────────────────────────────────────────────────────────────")
        for usage in all_usages[:5]:  # Show first 5 records
            LOG.info(f"   │ 👤 User ID: {usage.user_id}")
            LOG.info(f"   │ 🏷️ Operation: {usage.operation_type}")
            LOG.info(f"   │ 🤖 Model: {usage.model}")
            LOG.info(f"   │ 🔢 Tokens: {usage.total_tokens}")
            LOG.info(f"   │ 💵 Cost: {usage.estimated_cost}")
            LOG.info(f"   │ ⏰ Time: {usage.timestamp}")
            LOG.info(f"   ├─────────────────────────────────────────────────────────────")
        
        usages = self.db.query(TokenUsage).filter(TokenUsage.user_id == user_id).all()
        LOG.info(f"🔍 Filtered Query - user_id: {user_id}, found: {len(usages)} records")
        
        total_tokens = sum(u.total_tokens for u in usages)
        total_cost = sum(u.estimated_cost for u in usages)
        
        by_operation = {}
        for usage in usages:
            op = usage.operation_type
            if op not in by_operation:
                by_operation[op] = {"count": 0, "tokens": 0, "cost": 0.0}
            by_operation[op]["count"] += 1
            by_operation[op]["tokens"] += usage.total_tokens
            by_operation[op]["cost"] += usage.estimated_cost
        
        return {
            "total_operations": len(usages),
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 4),
            "by_operation": by_operation,
            "recent_operations": [
                {
                    "id": u.id,
                    "operation_type": u.operation_type,
                    "model": u.model,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "total_tokens": u.total_tokens,
                    "estimated_cost": u.estimated_cost,
                    "timestamp": u.timestamp.isoformat()
                }
                for u in sorted(usages, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }
    
    def get_current_pricing(self) -> Dict:
        """Get current pricing configuration"""
        return get_pricing_config()
