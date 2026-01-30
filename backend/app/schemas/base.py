"""
Base Schemas
============

Common schema patterns and mixins.
"""

from typing import TypeVar, Generic, List, Optional
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """
    Base Pydantic schema with common configuration.
    
    All API schemas should inherit from this base.
    """
    
    model_config = ConfigDict(
        # Allow ORM mode for SQLAlchemy models
        from_attributes=True,
        # Validate default values
        validate_default=True,
        # Use enum values instead of names
        use_enum_values=True,
        # Strip whitespace from strings
        str_strip_whitespace=True,
    )


class TimestampSchema(BaseSchema):
    """Schema mixin for timestamp fields."""
    
    created_at: datetime
    updated_at: datetime


T = TypeVar("T")


class PaginatedResponse(BaseSchema, Generic[T]):
    """
    Generic paginated response wrapper.
    
    Attributes:
        items: List of items for this page
        total: Total number of items
        page: Current page number (1-indexed)
        page_size: Number of items per page
        pages: Total number of pages
    """
    
    items: List[T]
    total: int
    page: int
    page_size: int
    pages: int
    
    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        page: int,
        page_size: int,
    ) -> "PaginatedResponse[T]":
        """Create a paginated response from items."""
        pages = (total + page_size - 1) // page_size if page_size > 0 else 0
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            pages=pages,
        )


class ErrorResponse(BaseSchema):
    """Standard error response."""
    
    detail: str
    code: Optional[str] = None
    errors: Optional[List[dict]] = None


class SuccessResponse(BaseSchema):
    """Standard success response."""
    
    success: bool = True
    message: Optional[str] = None
