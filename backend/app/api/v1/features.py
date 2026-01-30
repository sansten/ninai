"""
Feature detection endpoint.

Returns available features based on license configuration.
Allows frontend to gracefully handle enterprise vs community builds.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.v1.endpoints.auth import get_current_user
from app.models.user import User

router = APIRouter(prefix="/features", tags=["features"])


class FeatureFlags(BaseModel):
    """Available feature flags."""

    admin_operations: bool
    drift_detection: bool
    auto_eval_benchmarks: bool
    memory_observability: bool


@router.get("", response_model=FeatureFlags)
async def get_features(
    current_user: User = Depends(get_current_user),
) -> FeatureFlags:
    """
    Get available features based on license.

    Community builds: All enterprise features False
    Enterprise builds: Features determined by license claims
    """
    # Check if enterprise plugin is available
    try:
        from ninai_enterprise.licensing import get_license_info

        license_info = get_license_info()
        if license_info and license_info.get("valid"):
            # Enterprise license active - all features available
            return FeatureFlags(
                admin_operations=True,
                drift_detection=True,
                auto_eval_benchmarks=True,
                memory_observability=True,
            )
    except ImportError:
        # Enterprise plugin not installed - community build
        pass

    # Community build or invalid license - no enterprise features
    return FeatureFlags(
        admin_operations=False,
        drift_detection=False,
        auto_eval_benchmarks=False,
        memory_observability=False,
    )
