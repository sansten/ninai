"""
Database Backup Service - Database backup and restore operations
"""

import subprocess
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List
from sqlalchemy.orm import Session
from uuid import UUID

from app.models.backup import BackupTask, BackupSchedule, BackupRestore
from app.models.user import User


logger = logging.getLogger(__name__)


class S3BackupStorage:
    """Handle S3 storage operations for backups"""
    
    def __init__(self, bucket: str, prefix: str = "backups"):
        self.bucket = bucket
        self.prefix = prefix
        # Initialize boto3 client in production
        # self.s3_client = boto3.client('s3')
    
    def upload_backup(self, local_path: Path, s3_key: str) -> Tuple[bool, Optional[str]]:
        """Upload backup file to S3"""
        try:
            # In production: self.s3_client.upload_file(str(local_path), self.bucket, s3_key)
            logger.info(f"Uploading backup to s3://{self.bucket}/{s3_key}")
            return True, s3_key
        except Exception as e:
            logger.error(f"Failed to upload backup: {e}")
            return False, None
    
    def download_backup(self, s3_key: str, local_path: Path) -> bool:
        """Download backup file from S3"""
        try:
            # In production: self.s3_client.download_file(self.bucket, s3_key, str(local_path))
            logger.info(f"Downloading backup from s3://{self.bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to download backup: {e}")
            return False
    
    def delete_backup(self, s3_key: str) -> bool:
        """Delete backup file from S3"""
        try:
            # In production: self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            logger.info(f"Deleting backup from s3://{self.bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return False


class BackupValidator:
    """Validate backup integrity and consistency"""
    
    @staticmethod
    def calculate_checksum(file_path: Path) -> str:
        """Calculate SHA256 checksum of backup file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    @staticmethod
    def verify_backup_integrity(file_path: Path, expected_checksum: str) -> bool:
        """Verify backup file integrity"""
        actual_checksum = BackupValidator.calculate_checksum(file_path)
        return actual_checksum == expected_checksum
    
    @staticmethod
    def validate_backup_size(file_path: Path, max_size_gb: int) -> bool:
        """Check if backup exceeds maximum size"""
        size_gb = file_path.stat().st_size / (1024 ** 3)
        return size_gb <= max_size_gb


class DatabaseBackupService:
    """Main backup service for database backups"""
    
    def __init__(self, db_url: str, backup_dir: Path, s3_storage: Optional[S3BackupStorage] = None):
        self.db_url = db_url
        self.backup_dir = backup_dir
        self.s3_storage = s3_storage
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # read-only filesystem in containers; directory created at runtime
    
    def create_full_backup(self, db: Session, db_name: str = "ninai") -> Optional[BackupTask]:
        """Create full database backup"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"backup_full_{timestamp}.sql.gz"

            # pg_dump | gzip, run without a shell — db_name is passed as an
            # argv element, never interpolated into a shell command string,
            # so it can't be used for command injection regardless of where
            # a future caller sources it from.
            with open(backup_file, "wb") as out:
                dump_proc = subprocess.Popen(
                    ["pg_dump", "-h", "localhost", "-U", "postgres", "-d", db_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                gzip_result = subprocess.run(
                    ["gzip"], stdin=dump_proc.stdout, stdout=out, stderr=subprocess.PIPE
                )
                dump_proc.stdout.close()
                _, dump_stderr = dump_proc.communicate()

            if dump_proc.returncode != 0 or gzip_result.returncode != 0:
                logger.error(
                    f"pg_dump/gzip failed: pg_dump rc={dump_proc.returncode} "
                    f"({dump_stderr.decode(errors='replace')}), gzip rc={gzip_result.returncode}"
                )
                return None
            
            # Calculate checksum
            checksum = BackupValidator.calculate_checksum(backup_file)
            size_bytes = backup_file.stat().st_size
            
            # Create backup task record
            backup_task = BackupTask(
                backup_type="full",
                size_bytes=size_bytes,
                checksum_sha256=checksum,
                status="completed",
                duration_seconds=0  # Would calculate actual duration
            )
            
            # Upload to S3 if configured
            if self.s3_storage:
                s3_key = f"full/{timestamp}.sql.gz"
                success, s3_path = self.s3_storage.upload_backup(backup_file, s3_key)
                if success:
                    backup_task.s3_object_key = s3_key
                    backup_task.s3_path = f"s3://{self.s3_storage.bucket}/{s3_key}"
            
            db.add(backup_task)
            db.commit()
            
            logger.info(f"Full backup created: {backup_file} ({size_bytes} bytes)")
            return backup_task
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def create_incremental_backup(self, db: Session, db_name: str = "ninai") -> Optional[BackupTask]:
        """Create incremental backup (using WAL)"""
        # Implementation would use WAL-based incremental backups
        return self.create_full_backup(db, db_name)
    
    def schedule_backup(self, db: Session, frequency: str = "daily", 
                       retention_days: int = 30, backup_time: str = "02:00") -> BackupSchedule:
        """Create a backup schedule"""
        schedule = BackupSchedule(
            frequency=frequency,
            retention_days=retention_days,
            backup_time=backup_time,
            enabled=True,
            s3_bucket=self.s3_storage.bucket if self.s3_storage else "default-bucket",
            max_backup_size_gb=10,
            enable_incremental=True
        )
        db.add(schedule)
        db.commit()
        logger.info(f"Backup schedule created: {frequency} at {backup_time} UTC")
        return schedule
    
    def restore_backup(self, db: Session, backup_id: UUID, initiated_by: UUID, 
                      db_name: str = "ninai") -> Optional[BackupRestore]:
        """Restore database from backup"""
        try:
            backup_task = db.query(BackupTask).filter(BackupTask.id == backup_id).first()
            if not backup_task:
                raise ValueError("Backup not found")
            
            # Download from S3 if needed
            backup_file = None
            if backup_task.s3_object_key:
                backup_file = self.backup_dir / f"restore_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.sql.gz"
                if self.s3_storage:
                    success = self.s3_storage.download_backup(backup_task.s3_object_key, backup_file)
                    if not success:
                        raise ValueError("Failed to download backup from S3")
            else:
                # Find local backup file
                for f in self.backup_dir.glob("backup_*"):
                    if f.stat().st_size == backup_task.size_bytes:
                        backup_file = f
                        break
            
            if not backup_file:
                raise ValueError("Backup file not found")
            
            # Verify integrity
            if not BackupValidator.verify_backup_integrity(backup_file, backup_task.checksum_sha256):
                raise ValueError("Backup integrity check failed")
            
            # Create restore record
            restore = BackupRestore(
                backup_id=backup_id,
                initiated_by=initiated_by,
                status="in_progress"
            )
            db.add(restore)
            db.commit()
            
            start_time = datetime.utcnow()

            # gunzip | psql, run without a shell — same injection-avoidance
            # rationale as create_full_backup above.
            with open(backup_file, "rb") as inp:
                gunzip_proc = subprocess.Popen(
                    ["gunzip"], stdin=inp, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                result = subprocess.run(
                    ["psql", "-h", "localhost", "-U", "postgres", "-d", db_name],
                    stdin=gunzip_proc.stdout,
                    capture_output=True,
                    text=True,
                )
                gunzip_proc.stdout.close()
                _, gunzip_stderr = gunzip_proc.communicate()

            duration = (datetime.utcnow() - start_time).total_seconds()

            if result.returncode != 0 or gunzip_proc.returncode != 0:
                restore.status = "failed"
                restore.error_message = result.stderr or gunzip_stderr.decode(errors="replace")
                logger.error(
                    f"Restore failed: psql rc={result.returncode} ({result.stderr}), "
                    f"gunzip rc={gunzip_proc.returncode}"
                )
            else:
                restore.status = "completed"
                restore.duration_seconds = int(duration)
                logger.info(f"Restore completed in {duration} seconds")
            
            db.commit()
            return restore
            
        except Exception as e:
            logger.error(f"Restore operation failed: {e}")
            return None
    
    def cleanup_old_backups(self, db: Session, retention_days: int = 30) -> int:
        """Delete backups older than retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        # Mark old backups for deletion
        old_backups = db.query(BackupTask).filter(
            BackupTask.created_at < cutoff_date,
            BackupTask.status == "completed"
        ).all()
        
        deleted_count = 0
        for backup in old_backups:
            # Delete from S3
            if backup.s3_object_key and self.s3_storage:
                self.s3_storage.delete_backup(backup.s3_object_key)
            
            # Delete local file
            for f in self.backup_dir.glob("backup_*"):
                if f.stat().st_size == backup.size_bytes:
                    f.unlink()
            
            db.delete(backup)
            deleted_count += 1
        
        db.commit()
        logger.info(f"Cleaned up {deleted_count} old backups")
        return deleted_count
    
    def get_backup_statistics(self, db: Session) -> dict:
        """Get backup statistics"""
        total_backups = db.query(BackupTask).filter(BackupTask.status == "completed").count()
        total_size_bytes = db.query(BackupTask).filter(
            BackupTask.status == "completed"
        ).with_entities(
            db.func.sum(BackupTask.size_bytes)
        ).scalar() or 0
        
        failed_backups = db.query(BackupTask).filter(BackupTask.status == "failed").count()
        
        last_backup = db.query(BackupTask).filter(
            BackupTask.status == "completed"
        ).order_by(BackupTask.created_at.desc()).first()
        
        return {
            "total_backups": total_backups,
            "total_size_gb": total_size_bytes / (1024 ** 3),
            "failed_backups": failed_backups,
            "last_backup_time": last_backup.created_at if last_backup else None,
            "success_rate": (total_backups - failed_backups) / total_backups * 100 if total_backups > 0 else 0
        }


class BackupScheduler:
    """Handle backup scheduling and execution"""
    
    @staticmethod
    def should_run_backup(schedule: BackupSchedule) -> bool:
        """Check if a scheduled backup should run"""
        if not schedule.enabled:
            return False
        
        now = datetime.utcnow()
        current_time = now.strftime("%H:%M")
        
        # Check if current time matches backup_time
        if current_time != schedule.backup_time:
            return False
        
        # Check frequency
        if schedule.last_run_at is None:
            return True
        
        if schedule.frequency == "daily":
            return (now - schedule.last_run_at).days >= 1
        elif schedule.frequency == "weekly":
            return (now - schedule.last_run_at).days >= 7
        elif schedule.frequency == "monthly":
            return (now - schedule.last_run_at).days >= 30
        
        return False
    
    @staticmethod
    def execute_scheduled_backup(db: Session, schedule: BackupSchedule, 
                                backup_service: DatabaseBackupService) -> bool:
        """Execute a scheduled backup"""
        try:
            backup = backup_service.create_full_backup(db)
            
            if backup:
                schedule.last_run_at = datetime.utcnow()
                schedule.next_run_at = BackupScheduler.calculate_next_run(schedule)
                schedule.last_success_at = datetime.utcnow()
                schedule.consecutive_failures = 0
                db.commit()
                return True
            else:
                schedule.consecutive_failures += 1
                db.commit()
                return False
                
        except Exception as e:
            logger.error(f"Scheduled backup failed: {e}")
            schedule.consecutive_failures += 1
            db.commit()
            return False
    
    @staticmethod
    def calculate_next_run(schedule: BackupSchedule) -> datetime:
        """Calculate next scheduled backup time"""
        now = datetime.utcnow()
        backup_hour, backup_minute = map(int, schedule.backup_time.split(":"))
        
        next_run = now.replace(hour=backup_hour, minute=backup_minute, second=0, microsecond=0)
        
        if next_run <= now:
            # Already passed today, schedule for tomorrow
            next_run += timedelta(days=1)
        
        if schedule.frequency == "weekly":
            # Schedule for same time next week
            next_run += timedelta(days=6)
        elif schedule.frequency == "monthly":
            # Schedule for same time next month
            if next_run.month == 12:
                next_run = next_run.replace(year=next_run.year + 1, month=1)
            else:
                next_run = next_run.replace(month=next_run.month + 1)
        
        return next_run
