from __future__ import annotations

import re
from pathlib import Path


def _script_path() -> Path:
    return Path(__file__).resolve().parents[2] / "upgrade.sh"


def _script_text() -> str:
    return _script_path().read_text(encoding="utf-8")


def test_script_exists():
    assert _script_path().exists()


def test_script_has_shebang():
    text = _script_text()
    assert text.startswith("#!/usr/bin/env bash")


def test_dry_run_flag_recognized():
    text = _script_text()
    assert '[[ "${1:-}" == "--dry-run" ]]' in text


def test_missing_postgres_url_causes_exit_1_message():
    text = _script_text()
    assert '[[ -z "${POSTGRES_URL:-}" ]] && die "POSTGRES_URL not set"' in text


def test_missing_license_token_causes_exit_1_message():
    text = _script_text()
    assert '[[ -z "${LICENSE_TOKEN:-}" ]] && die "LICENSE_TOKEN not set"' in text


def test_missing_ninai_enterprise_message_present():
    text = _script_text()
    assert "ninai-enterprise package not found" in text


def test_dry_run_skips_alembic_command():
    text = _script_text()
    assert "if $DRY_RUN; then" in text
    assert "alembic upgrade head" in text


def test_dry_run_logs_would_run_message():
    text = _script_text()
    assert "[DRY RUN] Would run: alembic upgrade head" in text


def test_dry_run_skips_health_check():
    text = _script_text()
    assert "if ! $DRY_RUN; then" in text
    assert "http://localhost:8000/api/v1/health" in text


def test_script_checks_current_version():
    text = _script_text()
    assert "Checking current Ninai version" in text
    assert "getattr(app, '__version__', 'unknown')" in text


def test_script_logs_enterprise_package_version():
    text = _script_text()
    assert "Enterprise package version" in text
    assert "import ninai_enterprise; print(ninai_enterprise.__version__)" in text


def test_alembic_upgrade_head_called_without_dry_run():
    text = _script_text()
    pattern = r"if \$DRY_RUN; then[\s\S]*else[\s\S]*alembic upgrade head"
    assert re.search(pattern, text)


def test_health_check_url_is_localhost_8000():
    text = _script_text()
    assert "http://localhost:8000/api/v1/health" in text


def test_script_logs_upgrade_complete_message():
    text = _script_text()
    assert "Upgrade complete. Enterprise features are now active." in text


def test_dry_run_defaults_false():
    text = _script_text()
    assert "DRY_RUN=false" in text
