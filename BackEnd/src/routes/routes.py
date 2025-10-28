from fastapi import APIRouter
from src.controllers.report_controller import router as report_controller
from src.routes.ai_routes import router as ai_router

router = APIRouter()

router.include_router(report_controller, prefix="/reports", tags=["Reports"])
router.include_router(ai_router, prefix="/ai", tags=["AI"])