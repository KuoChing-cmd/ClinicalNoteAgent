"""Batch operations for database models."""

from typing import List, Dict, Any, Type, TypeVar
import logging
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BatchOperations:
    """Batch operations for efficient bulk data handling."""

    def __init__(self, model: Type[T], session: Session):
        """Initialize batch operations.

        Args:
            model: SQLAlchemy model class
            session: Database session
        """
        self.model = model
        self.session = session

    def batch_insert(self, objects_data: List[Dict[str, Any]]) -> List[T]:
        """Insert multiple records at once.

        Args:
            objects_data: List of dictionaries containing model data

        Returns:
            List of created model instances
        """
        try:
            objects = [self.model(**data) for data in objects_data]
            self.session.add_all(objects)
            self.session.commit()
            logger.info(f"Batch inserted {len(objects)} records")
            return objects
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error batch inserting records: {str(e)}")
            raise

    def batch_update(self, updates: List[Dict[str, Any]]) -> int:
        """Update multiple records.

        Args:
            updates: List of dictionaries with 'id' and fields to update

        Returns:
            Number of records updated
        """
        try:
            count = 0
            for update_data in updates:
                obj_id = update_data.pop("id")
                obj = self.session.query(self.model).filter(
                    self.model.id == obj_id
                ).first()
                if obj:
                    for key, value in update_data.items():
                        setattr(obj, key, value)
                    count += 1
            self.session.commit()
            logger.info(f"Batch updated {count} records")
            return count
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error batch updating records: {str(e)}")
            raise

    def batch_delete(self, obj_ids: List[int]) -> int:
        """Delete multiple records by IDs.

        Args:
            obj_ids: List of record IDs to delete

        Returns:
            Number of records deleted
        """
        try:
            count = self.session.query(self.model).filter(
                self.model.id.in_(obj_ids)
            ).delete()
            self.session.commit()
            logger.info(f"Batch deleted {count} records")
            return count
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error batch deleting records: {str(e)}")
            raise

    def batch_upsert(self, objects_data: List[Dict[str, Any]]) -> int:
        """Insert or update multiple records.

        Args:
            objects_data: List of dictionaries containing model data

        Returns:
            Number of records inserted or updated
        """
        try:
            count = 0
            for data in objects_data:
                obj_id = data.get("id")
                if obj_id:
                    # Update existing
                    obj = self.session.query(self.model).filter(
                        self.model.id == obj_id
                    ).first()
                    if obj:
                        for key, value in data.items():
                            if key != "id":
                                setattr(obj, key, value)
                else:
                    # Insert new
                    obj = self.model(**data)
                    self.session.add(obj)
                count += 1
            self.session.commit()
            logger.info(f"Batch upserted {count} records")
            return count
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error batch upserting records: {str(e)}")
            raise

    def export_to_list(self, records: List[T]) -> List[Dict[str, Any]]:
        """Export records to list of dictionaries.

        Args:
            records: List of model instances

        Returns:
            List of dictionaries
        """
        try:
            result = []
            for record in records:
                row = {}
                for column in self.model.__table__.columns:
                    row[column.name] = getattr(record, column.name)
                result.append(row)
            logger.info(f"Exported {len(result)} records")
            return result
        except Exception as e:
            logger.error(f"Error exporting records: {str(e)}")
            raise
