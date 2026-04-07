"""SCIM 2.0 provisioning endpoints (Feature 22)."""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.core.feature_gate import EnterpriseFeatures, require_feature
from app.core.security import get_password_hash
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.models.user import Role, User, UserRole

router = APIRouter()

SCIM_LIST_RESPONSE_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:ListResponse"
SCIM_USER_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:User"
SCIM_GROUP_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:Group"


def _user_to_scim(user: User) -> dict:
    return {
        "schemas": [SCIM_USER_SCHEMA],
        "id": str(user.id),
        "userName": str(user.email),
        "name": {"formatted": str(user.full_name or user.email)},
        "active": bool(user.is_active),
    }


def _group_to_scim(role: Role, members: list[dict]) -> dict:
    return {
        "schemas": [SCIM_GROUP_SCHEMA],
        "id": str(role.id),
        "displayName": str(role.display_name or role.name),
        "members": members,
    }


def _etag_for_user(user: User) -> str:
    ts = getattr(user, "updated_at", None)
    if ts is None:
        return 'W/"0"'
    return f'W/"{int(ts.timestamp())}"'


def _extract_display_name(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    name_obj = payload.get("name") if isinstance(payload.get("name"), dict) else {}
    formatted = name_obj.get("formatted")
    if isinstance(formatted, str) and formatted.strip():
        return formatted.strip()
    display = payload.get("displayName")
    if isinstance(display, str):
        return display.strip()
    return ""


def _parse_scim_user_filter(filter_expr: str | None) -> tuple[str, str] | None:
    if not filter_expr:
        return None
    expr = filter_expr.strip()
    if expr.startswith("userName eq "):
        value = expr[len("userName eq "):].strip().strip('"').strip("'")
        return ("userName", value.lower())
    if expr.startswith("active eq "):
        value = expr[len("active eq "):].strip().lower()
        if value in {"true", "false"}:
            return ("active", value)
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid SCIM filter expression")


def _apply_scim_patch_op(user: User, op: dict) -> None:
    path = str(op.get("path", "")).lower()
    value = op.get("value")
    action = str(op.get("op", "replace")).lower()

    if path == "active":
        user.is_active = bool(value) if action != "remove" else False
    elif path == "displayname":
        user.full_name = str(value or "")
    elif path == "username":
        user.email = str(value or "").lower()


@router.get("/Users")
async def list_scim_users(
    filter: str | None = Query(default=None),
    startIndex: int = Query(default=1),
    count: int = Query(default=100),
    tenant: TenantContext = Depends(require_org_admin()),
    _: None = Depends(require_feature(EnterpriseFeatures.SCIM)),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    parsed_filter = _parse_scim_user_filter(filter)

    stmt = (
        select(User)
        .join(UserRole, UserRole.user_id == User.id)
        .where(UserRole.organization_id == tenant.org_id)
        .order_by(User.email.asc())
    )
    if parsed_filter:
        kind, value = parsed_filter
        if kind == "userName":
            stmt = stmt.where(func.lower(User.email) == value)
        elif kind == "active":
            stmt = stmt.where(User.is_active.is_(value == "true"))

    rows = await db.execute(stmt)
    users = rows.scalars().all()

    start = max(startIndex, 1)
    per_page = max(count, 0)
    sliced = users[start - 1:start - 1 + per_page] if per_page else []

    resources = [_user_to_scim(user) for user in sliced]
    payload = {
        "schemas": [SCIM_LIST_RESPONSE_SCHEMA],
        "totalResults": len(users),
        "startIndex": start,
        "itemsPerPage": len(resources),
        "Resources": resources,
    }

    return Response(
        content=json.dumps(payload),
        media_type="application/scim+json",
        headers={"ETag": f'W/"{len(users)}"'},
    )


@router.post("/Users", status_code=status.HTTP_201_CREATED)
async def create_scim_user(
    payload: dict = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    _: None = Depends(require_feature(EnterpriseFeatures.SCIM)),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    user_name = str(payload.get("userName") or "").strip().lower()
    if not user_name:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="userName is required")

    existing = await db.execute(select(User).where(User.email == user_name))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already exists")

    full_name = str((payload.get("name") or {}).get("formatted") or user_name).strip() or user_name
    active = bool(payload.get("active", True))

    user = User(
        id=str(uuid4()),
        email=user_name,
        hashed_password=get_password_hash(uuid4().hex),
        full_name=full_name,
        is_active=active,
    )
    db.add(user)
    await db.flush()

    role_stmt = select(Role).where(
        Role.organization_id == tenant.org_id,
        Role.name == "member",
    )
    role_res = await db.execute(role_stmt)
    role = role_res.scalar_one_or_none()
    if role is None:
        role = Role(
            id=str(uuid4()),
            name="member",
            display_name="Member",
            description="Default member role",
            permissions=[],
            organization_id=tenant.org_id,
            is_system=False,
            is_default=True,
        )
        db.add(role)
        await db.flush()

    db.add(
        UserRole(
            id=str(uuid4()),
            user_id=user.id,
            role_id=role.id,
            organization_id=tenant.org_id,
        )
    )

    await db.commit()
    return _user_to_scim(user)


@router.get("/Users/{user_id}")
async def get_scim_user(
    user_id: str,
    tenant: TenantContext = Depends(require_org_admin()),
    _: None = Depends(require_feature(EnterpriseFeatures.SCIM)),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    stmt = (
        select(User)
        .join(UserRole, UserRole.user_id == User.id)
        .where(User.id == user_id, UserRole.organization_id == tenant.org_id)
    )
    res = await db.execute(stmt)
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    response = _user_to_scim(user)
    return Response(
        content=json.dumps(response),
        media_type="application/scim+json",
        headers={"ETag": _etag_for_user(user)},
    )


@router.patch("/Users/{user_id}")
async def patch_scim_user(
    user_id: str,
    payload: dict = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    _: None = Depends(require_feature(EnterpriseFeatures.SCIM)),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    stmt = (
        select(User)
        .join(UserRole, UserRole.user_id == User.id)
        .where(User.id == user_id, UserRole.organization_id == tenant.org_id)
    )
    res = await db.execute(stmt)
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    for op in payload.get("Operations", []):
        _apply_scim_patch_op(user, op)

    await db.commit()
    response = _user_to_scim(user)
    return Response(
        content=json.dumps(response),
        media_type="application/scim+json",
        headers={"ETag": _etag_for_user(user)},
    )


@router.put("/Users/{user_id}")
async def replace_scim_user(
    user_id: str,
    payload: dict = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    _: None = Depends(require_feature(EnterpriseFeatures.SCIM)),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    stmt = (
        select(User)
        .join(UserRole, UserRole.user_id == User.id)
        .where(User.id == user_id, UserRole.organization_id == tenant.org_id)
    )
    res = await db.execute(stmt)
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user_name = payload.get("userName")
    if isinstance(user_name, str) and user_name.strip():
        user.email = user_name.strip().lower()

    display_name = _extract_display_name(payload)
    if display_name:
        user.full_name = display_name

    user.is_active = bool(payload.get("active", True))

    await db.commit()
    response = _user_to_scim(user)
    return Response(
        content=json.dumps(response),
        media_type="application/scim+json",
        headers={"ETag": _etag_for_user(user)},
    )


@router.delete("/Users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scim_user(
    user_id: str,
    tenant: TenantContext = Depends(require_org_admin()),
    _: None = Depends(require_feature(EnterpriseFeatures.SCIM)),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    stmt = (
        select(User)
        .join(UserRole, UserRole.user_id == User.id)
        .where(User.id == user_id, UserRole.organization_id == tenant.org_id)
    )
    res = await db.execute(stmt)
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Deprovision in-place for auditability and GDPR workflow triggers.
    user.is_active = False
    await db.commit()
    return None


@router.patch("/Groups/{group_id}")
async def patch_scim_group(
    group_id: str,
    payload: dict = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    _: None = Depends(require_feature(EnterpriseFeatures.SCIM)),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    role_res = await db.execute(
        select(Role).where(Role.id == group_id, Role.organization_id == tenant.org_id)
    )
    role = role_res.scalar_one_or_none()
    if role is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found")

    for op in payload.get("Operations", []):
        action = str(op.get("op", "")).lower()
        value = op.get("value")
        members = value if isinstance(value, list) else []
        for member in members:
            user_id = str(member.get("value", "")).strip()
            if not user_id:
                continue
            if action == "add":
                existing = await db.execute(
                    select(UserRole).where(
                        UserRole.organization_id == tenant.org_id,
                        UserRole.user_id == user_id,
                        UserRole.role_id == role.id,
                    )
                )
                if existing.scalar_one_or_none() is None:
                    db.add(
                        UserRole(
                            id=str(uuid4()),
                            user_id=user_id,
                            role_id=role.id,
                            organization_id=tenant.org_id,
                        )
                    )
            elif action == "remove":
                existing = await db.execute(
                    select(UserRole).where(
                        UserRole.organization_id == tenant.org_id,
                        UserRole.user_id == user_id,
                        UserRole.role_id == role.id,
                    )
                )
                ur = existing.scalar_one_or_none()
                if ur is not None:
                    await db.delete(ur)

    await db.commit()
    members_res = await db.execute(
        select(User.id, User.email)
        .join(UserRole, UserRole.user_id == User.id)
        .where(
            UserRole.organization_id == tenant.org_id,
            UserRole.role_id == role.id,
        )
    )
    members = [{"value": str(uid), "display": str(email)} for uid, email in members_res.all()]
    response = _group_to_scim(role, members)
    return Response(content=json.dumps(response), media_type="application/scim+json")


@router.delete("/Groups/{group_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scim_group(
    group_id: str,
    tenant: TenantContext = Depends(require_org_admin()),
    _: None = Depends(require_feature(EnterpriseFeatures.SCIM)),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    role_res = await db.execute(
        select(Role).where(Role.id == group_id, Role.organization_id == tenant.org_id)
    )
    role = role_res.scalar_one_or_none()
    if role is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found")

    await db.delete(role)
    await db.commit()
    return None


@router.get("/Groups")
async def list_scim_groups(
    tenant: TenantContext = Depends(require_org_admin()),
    _: None = Depends(require_feature(EnterpriseFeatures.SCIM)),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    roles_res = await db.execute(
        select(Role).where(Role.organization_id == tenant.org_id).order_by(Role.name.asc())
    )
    roles = roles_res.scalars().all()

    resources: list[dict] = []
    for role in roles:
        members_res = await db.execute(
            select(User.id, User.email)
            .join(UserRole, UserRole.user_id == User.id)
            .where(
                UserRole.organization_id == tenant.org_id,
                UserRole.role_id == role.id,
            )
        )
        members = [
            {
                "value": str(user_id),
                "display": str(email),
            }
            for user_id, email in members_res.all()
        ]
        resources.append(_group_to_scim(role, members))

    return {
        "schemas": [SCIM_LIST_RESPONSE_SCHEMA],
        "totalResults": len(resources),
        "startIndex": 1,
        "itemsPerPage": len(resources),
        "Resources": resources,
    }


@router.post("/Groups", status_code=status.HTTP_201_CREATED)
async def create_scim_group(
    payload: dict = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    _: None = Depends(require_feature(EnterpriseFeatures.SCIM)),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    display_name = str(payload.get("displayName") or "").strip()
    if not display_name:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="displayName is required")

    role_name = display_name.lower().replace(" ", "_")
    existing_stmt = select(func.count()).select_from(Role).where(
        Role.organization_id == tenant.org_id,
        Role.name == role_name,
    )
    existing = await db.execute(existing_stmt)
    if int(existing.scalar_one() or 0) > 0:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Group already exists")

    role = Role(
        id=str(uuid4()),
        name=role_name,
        display_name=display_name,
        description="SCIM provisioned group",
        permissions=[],
        organization_id=tenant.org_id,
        is_system=False,
        is_default=False,
    )
    db.add(role)
    await db.commit()

    return _group_to_scim(role, members=[])
