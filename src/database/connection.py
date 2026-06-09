"""Database connection manager."""

from typing import Optional, Any
import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from .config import DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""

    engine: Any
    session_factory: Any

    def __init__(self, config: DatabaseConfig):
        """Initialize database manager.

        Args:
            config: DatabaseConfig instance
        """
        self.config = config
        self.config.validate()
        self.engine = None
        self.session_factory = None
        self._session: Optional[Session] = None

    def connect(self) -> bool:
        """Establish database connection.

        Returns:
            True if connection successful
        """
        try:
            connection_string = self.config.get_connection_string()
            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                connect_args={
                    "connect_timeout": int(self.config.connect_timeout),
                    "read_timeout": int(self.config.read_timeout),
                    "write_timeout": int(self.config.write_timeout),
                },
                echo=False,
            )

            self.session_factory = sessionmaker(bind=self.engine)

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(
                f"Successfully connected to {self.config.host}:{self.config.port}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def disconnect(self) -> None:
        """Close database connection."""
        if self._session:
            self._session.close()
            self._session = None
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")

    def get_session(self) -> Session:
        """Get or create database session.

        Returns:
            SQLAlchemy session
        """
        if self.session_factory is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        if self._session is None:
            self._session = self.session_factory()
        return self._session

    def close_session(self) -> None:
        """Close current session."""
        if self._session:
            self._session.close()
            self._session = None

    def create_tables(self, base) -> None:
        """Create all tables defined in models.

        Args:
            base: SQLAlchemy declarative base
        """
        if self.engine is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        base.metadata.create_all(self.engine)
        logger.info("All tables created successfully")

    def drop_tables(self, base) -> None:
        """Drop all tables defined in models.

        Args:
            base: SQLAlchemy declarative base
        """
        if self.engine is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        base.metadata.drop_all(self.engine)
        logger.info("All tables dropped successfully")

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in database.

        Args:
            table_name: Name of the table

        Returns:
            True if table exists
        """
        if self.engine is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
