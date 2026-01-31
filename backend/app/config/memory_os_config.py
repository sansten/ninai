"""
Configuration file for Memory OS services for testing and deployment.
"""

import os
from typing import Optional

class Config:
    """Service configuration."""
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "true").lower() == "true"
    
    # Snapshot Storage
    SNAPSHOT_STORAGE_PATH: str = os.getenv(
        "SNAPSHOT_STORAGE_PATH",
        "backend_attachments/snapshots"
    )
    SNAPSHOT_STORAGE_BACKEND: str = os.getenv("SNAPSHOT_STORAGE_BACKEND", "local")  # local or s3
    
    # S3 Configuration
    S3_BUCKET: str = os.getenv("S3_BUCKET", "ninai-snapshots")
    S3_REGION: str = os.getenv("S3_REGION", "us-east-1")
    S3_ACCESS_KEY: Optional[str] = os.getenv("S3_ACCESS_KEY")
    S3_SECRET_KEY: Optional[str] = os.getenv("S3_SECRET_KEY")
    
    # Rate Limiting
    DEFAULT_RATE_LIMIT: int = int(os.getenv("DEFAULT_RATE_LIMIT", "100"))
    DEFAULT_RATE_WINDOW: int = int(os.getenv("DEFAULT_RATE_WINDOW", "60"))
    
    # Monthly Quotas
    DEFAULT_MONTHLY_TOKENS: int = int(os.getenv("DEFAULT_MONTHLY_TOKENS", "100000"))
    DEFAULT_MONTHLY_STORAGE_MB: int = int(os.getenv("DEFAULT_MONTHLY_STORAGE_MB", "1000"))
    DEFAULT_MONTHLY_REQUESTS: int = int(os.getenv("DEFAULT_MONTHLY_REQUESTS", "100000"))
    
    # Policy Versioning
    ENABLE_CANARY_ROLLOUT: bool = os.getenv("ENABLE_CANARY_ROLLOUT", "true").lower() == "true"
    DEFAULT_CANARY_PERCENTAGE: int = int(os.getenv("DEFAULT_CANARY_PERCENTAGE", "50"))
    
    # Backup
    BACKUP_RETENTION_DAYS: int = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
    ENABLE_AUTOMATED_BACKUPS: bool = os.getenv("ENABLE_AUTOMATED_BACKUPS", "true").lower() == "true"
    
    # DLQ
    DLQ_MAX_RETRIES: int = int(os.getenv("DLQ_MAX_RETRIES", "3"))
    DLQ_RETRY_DELAYS: list = [60, 300, 900]  # 1min, 5min, 15min


def get_config() -> Config:
    """Get configuration instance."""
    return Config()
