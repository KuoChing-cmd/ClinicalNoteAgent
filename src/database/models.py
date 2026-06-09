"""Base model for database operations."""

from sqlalchemy import Column, Integer, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

BaseModel = declarative_base()


class TimestampMixin:
    """Mixin class to add timestamp fields."""

    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )


class IdMixin:
    """Mixin class to add primary key field."""

    id = Column(Integer, primary_key=True, autoincrement=True)
