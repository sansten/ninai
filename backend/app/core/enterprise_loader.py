"""Optional Enterprise plugin loader.

The OSS build must run without any enterprise dependencies.
If the enterprise package is installed, it can register additional routes,
Celery tasks, and a license-backed FeatureGate implementation.

Expected enterprise package API:
  - module name: ninai_enterprise
  - callable: register(app: FastAPI) -> None

This loader is intentionally best-effort and never blocks app startup.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI


logger = logging.getLogger(__name__)


def try_register_enterprise(app: FastAPI) -> None:
    try:
        import ninai_enterprise  # type: ignore

        register = getattr(ninai_enterprise, "register", None)
        if callable(register):
            register(app)
            logger.info("Enterprise package detected; enterprise features registered")
        else:
            logger.warning("Enterprise package detected but no register(app) found")
    except ImportError:
        # Community edition: enterprise package not installed.
        return
    except Exception:
        logger.exception("Failed to register enterprise package; continuing in community mode")
        return
