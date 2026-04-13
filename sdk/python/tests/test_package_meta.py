from __future__ import annotations

import re
import tomllib
from pathlib import Path

import ninai


def _sdk_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    path = _sdk_root() / "pyproject.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_version_is_pep440() -> None:
    # Accepts semver (1.2.3) and pre-release forms (1.2.3a1, 1.2.3b1, 1.2.3rc1)
    version = ninai.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", version), version


def test_dunder_all_exports_ninai_client() -> None:
    assert "NinaiClient" in ninai.__all__


def test_package_name_is_ninai() -> None:
    data = _pyproject()
    assert data["project"]["name"] == "ninai"


def test_required_dependencies_present() -> None:
    deps = _pyproject()["project"]["dependencies"]
    assert any(dep.startswith("httpx") for dep in deps)
    assert any(dep.startswith("pydantic") for dep in deps)


def test_changelog_exists() -> None:
    assert (_sdk_root() / "CHANGELOG.md").exists()


def test_description_non_empty() -> None:
    description = _pyproject()["project"]["description"]
    assert isinstance(description, str)
    assert description.strip()


def test_requires_python_is_at_least_3_10() -> None:
    assert _pyproject()["project"]["requires-python"] == ">=3.10"


def test_keywords_include_ninai() -> None:
    keywords = _pyproject()["project"]["keywords"]
    assert "ninai" in keywords


def test_authors_include_email() -> None:
    authors = _pyproject()["project"]["authors"]
    assert authors
    assert "email" in authors[0]
    assert "@" in authors[0]["email"]


def test_urls_has_homepage() -> None:
    urls = _pyproject()["project"]["urls"]
    assert "Homepage" in urls
    assert urls["Homepage"].startswith("https://")
