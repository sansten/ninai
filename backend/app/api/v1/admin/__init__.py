"""Admin API router initialization"""
from fastapi import APIRouter
from app.api.v1.admin.routes import router as admin_router

# Create the main admin router
admin_api = APIRouter()
admin_api.include_router(admin_router)

__all__ = ["admin_api"]
