from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from core.database import get_db
from .service import TokenTrackingService

router = APIRouter(prefix="/v1/analytics", tags=["Analytics"])

@router.get("/usage/{user_id}")
async def get_user_analytics(user_id: str, db: Session = Depends(get_db)):
    """Get token usage analytics for a user"""
    try:
        service = TokenTrackingService(db)
        analytics = service.get_user_analytics(user_id)
        
        # Debug logging
        from core.logger.logger import LOG
        LOG.info(f"🔍 Analytics Query for user_id: {user_id}")
        LOG.info(f"   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 📊 Total Operations: {analytics.get('total_operations', 0)}")
        LOG.info(f"   │ 🔢 Total Tokens: {analytics.get('total_tokens', 0)}")
        LOG.info(f"   │ 💵 Total Cost: {analytics.get('total_cost', 0)}")
        LOG.info(f"   │ 🏷️ Operations: {list(analytics.get('by_operation', {}).keys())}")
        LOG.info(f"   └─────────────────────────────────────────────────────────────")
        
        return {"success": True, "data": analytics}
    except Exception as e:
        from core.logger.logger import LOG
        LOG.error(f"❌ Analytics Error for user_id {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pricing")
async def get_pricing_config():
    """Get current pricing configuration"""
    from .service import get_pricing_config as pricing_func
    return {"success": True, "data": pricing_func()}
