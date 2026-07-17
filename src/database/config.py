"""Database configuration management."""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration for remote MySQL database connection."""

    host: str = "localhost"
    port: int = 3307
    user: str = "root"
    password: str = "hanwen123"
    database: str = "mimic4"
    charset: str = "utf8mb4"
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    connect_timeout: int = 10
    read_timeout: int = 120
    write_timeout: int = 120

    def get_connection_string(self) -> str:
        """Generate SQLAlchemy connection string.

        Returns:
            Connection string for SQLAlchemy
        """
        return (
            f"mysql+pymysql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}?"
            f"charset={self.charset}"
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DatabaseConfig":
        """Create config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            DatabaseConfig instance
        """
        return cls(**config_dict)

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if valid, raises ValueError otherwise
        """
        if not self.host:
            raise ValueError("Database host is required")
        if not self.user:
            raise ValueError("Database user is required")
        if not self.database:
            raise ValueError("Database name is required")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Invalid port number")
        return True
