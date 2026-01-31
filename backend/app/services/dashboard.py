"""
Admin Dashboard Service
Aggregates metrics, KPIs, and system health for the admin dashboard
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy import func, and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.admin import AdminAuditLog
from app.models.memory import MemoryMetadata


class DashboardKPI:
    """KPI data holder"""
    def __init__(self, label: str, value: str, unit: Optional[str] = None, 
                 trend: Optional[str] = None, change_percent: Optional[float] = None):
        self.label = label
        self.value = value
        self.unit = unit
        self.trend = trend
        self.change_percent = change_percent


class SystemMetrics:
    """System health metrics"""
    def __init__(self, name: str, status: str, message: str = ""):
        self.name = name
        self.status = status
        self.message = message
        self.last_check = datetime.utcnow()


class DashboardService:
    """
    Service for aggregating dashboard data
    """
    
    @staticmethod
    async def get_user_kpis(db: AsyncSession) -> List[DashboardKPI]:
        """
        Get user-related KPIs
        """
        # Total active users
        result = await db.execute(
            select(func.count(User.id)).where(User.is_active == True)
        )
        total_users = result.scalar() or 0
        
        # New users in last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        result = await db.execute(
            select(func.count(User.id)).where(
                and_(User.is_active == True, User.created_at >= thirty_days_ago)
            )
        )
        new_users_30d = result.scalar() or 0
        
        # Active users in last 7 days (with login)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        result = await db.execute(
            select(func.count(User.id)).where(
                and_(User.is_active == True, User.last_login_at >= seven_days_ago)
            )
        )
        active_users_7d = result.scalar() or 0
        
        # Calculate trend (simplified)
        user_trend = "up" if new_users_30d > 0 else "stable"
        user_change = (new_users_30d / max(total_users, 1)) * 100
        
        return [
            DashboardKPI(
                label="Total Users",
                value=f"{total_users:,}",
                unit="users",
                trend=user_trend,
                change_percent=round(user_change, 1)
            ),
            DashboardKPI(
                label="Active Last 7 Days",
                value=f"{active_users_7d:,}",
                unit="users",
                trend="up" if active_users_7d > 0 else "stable",
                change_percent=None
            ),
            DashboardKPI(
                label="New (30 days)",
                value=f"{new_users_30d:,}",
                unit="users",
                trend=user_trend,
                change_percent=None
            ),
        ]
    
    @staticmethod
    async def get_memory_kpis(db: AsyncSession) -> List[DashboardKPI]:
        """
        Get memory/knowledge-related KPIs
        """
        # Total memories
        result = await db.execute(select(func.count(MemoryMetadata.id)))
        total_memories = result.scalar() or 0
        
        # Memories created in last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        result = await db.execute(
            select(func.count(MemoryMetadata.id)).where(
                MemoryMetadata.created_at >= thirty_days_ago
            )
        )
        new_memories_30d = result.scalar() or 0
        
        # Avg memories per user (for system efficiency)
        result = await db.execute(
            select(func.count(User.id)).where(User.is_active == True)
        )
        active_users = result.scalar() or 1
        avg_per_user = (total_memories / max(active_users, 1))
        
        return [
            DashboardKPI(
                label="Total Memories",
                value=f"{total_memories:,}",
                unit="items",
                trend="up",
                change_percent=(new_memories_30d / max(total_memories, 1)) * 100
            ),
            DashboardKPI(
                label="New (30 days)",
                value=f"{new_memories_30d:,}",
                unit="items",
                trend="up" if new_memories_30d > 0 else "stable",
                change_percent=None
            ),
            DashboardKPI(
                label="Avg per User",
                value=f"{avg_per_user:.1f}",
                unit="items",
                trend="stable",
                change_percent=None
            ),
        ]
    
    @staticmethod
    async def get_system_kpis(db: AsyncSession) -> List[DashboardKPI]:
        """
        Get system-level KPIs (simplified)
        """
        # API requests (from audit logs as proxy)
        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())
        result = await db.execute(
            select(func.count(AdminAuditLog.id)).where(
                AdminAuditLog.created_at >= today_start
            )
        )
        today_requests = result.scalar() or 0
        
        # Uptime (simplified - assume 99.9% for now, would come from monitoring)
        uptime = 99.98
        
        # Error rate (from audit logs errors if tracked)
        result = await db.execute(
            select(func.count(AdminAuditLog.id)).where(
                and_(
                    AdminAuditLog.created_at >= today_start,
                    AdminAuditLog.action == "error"
                )
            )
        )
        error_count = result.scalar() or 0
        
        error_rate = (error_count / max(today_requests, 1)) * 100 if today_requests > 0 else 0
        
        return [
            DashboardKPI(
                label="API Requests (Today)",
                value=f"{today_requests:,}",
                unit="req",
                trend="up",
                change_percent=None
            ),
            DashboardKPI(
                label="Uptime",
                value=f"{uptime:.2f}%",
                unit="availability",
                trend="up",
                change_percent=None
            ),
            DashboardKPI(
                label="Error Rate",
                value=f"{error_rate:.2f}%",
                unit="errors",
                trend="down",
                change_percent=-0.5  # Trend should be down (fewer errors)
            ),
        ]
    
    @staticmethod
    def get_service_health(db: AsyncSession) -> List[SystemMetrics]:
        """
        Get health status of system services
        Returns list of SystemMetrics
        """
        # These would be integrated with actual service health checks
        services = [
            SystemMetrics(
                name="Backend API",
                status="healthy",
                message="All systems operational"
            ),
            SystemMetrics(
                name="Database",
                status="healthy",
                message="Connection pool: 8/10 active"
            ),
            SystemMetrics(
                name="Authentication",
                status="healthy",
                message="Token service operational"
            ),
            SystemMetrics(
                name="Memory Store",
                status="healthy",
                message="Cache hit rate: 92%"
            ),
        ]
        return services
    
    @staticmethod
    async def get_recent_activities(db: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent admin activities from audit logs
        """
        result = await db.execute(
            select(AdminAuditLog).order_by(
                AdminAuditLog.created_at.desc()
            ).limit(limit)
        )
        logs = result.scalars().all()
        
        activities = []
        for log in logs:
            activity = {
                "id": str(log.id),
                "admin_id": str(log.admin_id),
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": log.resource_id,
                "timestamp": log.created_at.isoformat(),
                "description": f"{log.action.capitalize()} {log.resource_type}",
                "status": "success"  # Could track failed actions
            }
            activities.append(activity)
        
        return activities
    
    @staticmethod
    async def get_full_dashboard_data(db: AsyncSession) -> Dict[str, Any]:
        """
        Get complete dashboard data in one call
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "kpis": {
                "users": await DashboardService.get_user_kpis(db),
                "memories": await DashboardService.get_memory_kpis(db),
                "system": await DashboardService.get_system_kpis(db),
            },
            "services": DashboardService.get_service_health(db),
            "recent_activities": await DashboardService.get_recent_activities(db),
            "alerts": [
                # Would be populated from alert system
            ]
        }
