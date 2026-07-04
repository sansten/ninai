"""Regression tests: DatabaseBackupService used to build pg_dump/psql commands
as f-string-interpolated shell commands (`subprocess.run(cmd, shell=True)`),
so a dynamic db_name would be shell command injection. It's currently dead
code (backups.py endpoints are TODO stubs that never call it), but the fix
closes the hole before anything wires it up. These tests assert the argv-list
/ no-shell invocation shape rather than actually running pg_dump.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.db_backup_service import DatabaseBackupService


def _make_service(tmp_path: Path) -> DatabaseBackupService:
    return DatabaseBackupService(db_url="postgresql://x", backup_dir=tmp_path)


class TestCreateFullBackupNoShell:
    def test_pg_dump_invoked_with_argv_list_not_shell_string(self, tmp_path):
        svc = _make_service(tmp_path)
        db = MagicMock()

        malicious_db_name = "ninai; rm -rf / #"

        popen_mock = MagicMock()
        popen_mock.returncode = 0
        popen_mock.stdout = MagicMock()
        popen_mock.communicate.return_value = (b"", b"")

        gzip_result = MagicMock()
        gzip_result.returncode = 0

        with patch("subprocess.Popen", return_value=popen_mock) as popen_patch, \
             patch("subprocess.run", return_value=gzip_result) as run_patch, \
             patch("app.services.db_backup_service.BackupValidator.calculate_checksum", return_value="abc"), \
             patch("pathlib.Path.stat") as stat_mock:
            stat_mock.return_value = MagicMock(st_size=123)
            svc.create_full_backup(db, db_name=malicious_db_name)

        # pg_dump must be called as an argv list (no shell), and the db_name
        # must appear as a single argv element — never concatenated into a
        # string that a shell would parse.
        popen_args = popen_patch.call_args
        cmd = popen_args[0][0]
        assert isinstance(cmd, list)
        assert malicious_db_name in cmd
        assert popen_args.kwargs.get("stdout") is not None

        # gzip must also be a plain argv list — no shell=True anywhere.
        run_args = run_patch.call_args
        assert isinstance(run_args[0][0], list)
        assert run_args.kwargs.get("shell", False) is not True


class TestRestoreBackupNoShell:
    def test_psql_invoked_with_argv_list_not_shell_string(self, tmp_path):
        svc = _make_service(tmp_path)
        db = MagicMock()

        backup_file = tmp_path / "backup_full_20260101_000000.sql.gz"
        backup_file.write_bytes(b"fake-gzip-content")

        from app.models.backup import BackupTask
        backup_task = MagicMock(spec=BackupTask)
        backup_task.id = "b1"
        backup_task.s3_object_key = None
        backup_task.size_bytes = backup_file.stat().st_size
        backup_task.checksum_sha256 = "abc"

        db.query.return_value.filter.return_value.first.return_value = backup_task

        malicious_db_name = "ninai; rm -rf / #"

        gunzip_mock = MagicMock()
        gunzip_mock.returncode = 0
        gunzip_mock.stdout = MagicMock()
        gunzip_mock.communicate.return_value = (b"", b"")

        psql_result = MagicMock()
        psql_result.returncode = 0
        psql_result.stderr = ""

        with patch("subprocess.Popen", return_value=gunzip_mock) as popen_patch, \
             patch("subprocess.run", return_value=psql_result) as run_patch, \
             patch(
                 "app.services.db_backup_service.BackupValidator.verify_backup_integrity",
                 return_value=True,
             ):
            svc.restore_backup(db, backup_id="b1", initiated_by="u1", db_name=malicious_db_name)

        gunzip_args = popen_patch.call_args
        assert isinstance(gunzip_args[0][0], list)

        psql_args = run_patch.call_args
        cmd = psql_args[0][0]
        assert isinstance(cmd, list)
        assert malicious_db_name in cmd
        assert psql_args.kwargs.get("shell", False) is not True
