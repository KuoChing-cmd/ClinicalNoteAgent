"""CRUD operations for database models."""

import logging
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CRUDOperations(Generic[T]):
    """Generic CRUD operations for database models."""

    def __init__(self, model: Type[T], session: Session):
        """Initialize CRUD operations.

        Args:
            model: SQLAlchemy model class
            session: Database session
        """
        self.model = model
        self.session = session

    def create(self, obj_data: Dict[str, Any]) -> T:
        """Create a new record.

        Args:
            obj_data: Dictionary of model fields

        Returns:
            Created model instance
        """
        try:
            obj = self.model(**obj_data)
            self.session.add(obj)
            self.session.commit()
            logger.info(f"Created record: {obj}")
            return obj
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating record: {str(e)}")
            raise

    def read(self, obj_id: int) -> Optional[T]:
        """Read a record by ID.

        Args:
            obj_id: Record ID

        Returns:
            Model instance or None if not found
        """
        try:
            obj = self.session.query(self.model).filter(self.model.id == obj_id).first()
            if obj:
                logger.info(f"Read record: {obj}")
            return obj
        except Exception as e:
            logger.error(f"Error reading record: {str(e)}")
            raise

    def read_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """Read all records with pagination.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of model instances
        """
        try:
            objs = self.session.query(self.model).offset(skip).limit(limit).all()
            logger.info(f"Read {len(objs)} records")
            return objs
        except Exception as e:
            logger.error(f"Error reading records: {str(e)}")
            raise

    def update(self, obj_id: int, obj_data: Dict[str, Any]) -> Optional[T]:
        """Update a record.

        Args:
            obj_id: Record ID
            obj_data: Dictionary of fields to update

        Returns:
            Updated model instance or None if not found
        """
        try:
            obj = self.session.query(self.model).filter(self.model.id == obj_id).first()
            if obj:
                for key, value in obj_data.items():
                    setattr(obj, key, value)
                self.session.commit()
                logger.info(f"Updated record: {obj}")
                return obj
            return None
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating record: {str(e)}")
            raise

    def delete(self, obj_id: int) -> bool:
        """Delete a record.

        Args:
            obj_id: Record ID

        Returns:
            True if deleted, False if not found
        """
        try:
            obj = self.session.query(self.model).filter(self.model.id == obj_id).first()
            if obj:
                self.session.delete(obj)
                self.session.commit()
                logger.info(f"Deleted record with ID: {obj_id}")
                return True
            return False
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting record: {str(e)}")
            raise

    def query(self, **filters) -> List[T]:
        """Query records by filters.

        Args:
            **filters: Filter conditions

        Returns:
            List of matching model instances
        """
        try:
            query = self.session.query(self.model)
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.filter(getattr(self.model, key) == value)
            objs = query.all()
            logger.info(f"Query returned {len(objs)} records")
            return objs
        except Exception as e:
            logger.error(f"Error querying records: {str(e)}")
            raise

    def count(self) -> int:
        """Count total records.

        Returns:
            Total number of records
        """
        try:
            count = self.session.query(self.model).count()
            logger.info(f"Total records: {count}")
            return count
        except Exception as e:
            logger.error(f"Error counting records: {str(e)}")
            raise
