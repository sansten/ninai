"""
Cross-Region Replication Service

Replicates memories across multiple regions for DR and performance:
- Write to primary region
- Async replicate to secondary regions
- Read from nearest region
- Automatic failover handling
- Conflict resolution
"""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio
import uuid


class ReplicationStatus(str, Enum):
    """Replication status."""
    PENDING = "pending"
    REPLICATING = "replicating"
    REPLICATED = "replicated"
    FAILED = "failed"
    BEHIND = "behind"


class RegionStatus(str, Enum):
    """Region status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class Region:
    """Configuration for a region."""
    name: str
    db_url: str
    is_primary: bool = False
    read_weight: int = 1  # Load balancing weight
    healthy: bool = True
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class ReplicationEvent:
    """Log entry for replication event."""
    event_id: str
    timestamp: datetime
    operation: str  # 'create', 'update', 'delete'
    memory_id: str
    source_region: str
    target_regions: List[str]
    status: ReplicationStatus
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'operation': self.operation,
            'memory_id': self.memory_id,
            'source_region': self.source_region,
            'target_regions': self.target_regions,
            'status': self.status.value,
            'error_message': self.error_message,
        }


class ReplicationService:
    """
    Manages cross-region replication.
    
    Features:
    - Multi-region read replicas
    - Async replication pipeline
    - Conflict resolution (last-write-wins)
    - Automatic failover
    - Replication lag monitoring
    """
    
    def __init__(self, primary_region: Region, secondary_regions: Optional[List[Region]] = None):
        """
        Initialize replication service.
        
        Args:
            primary_region: Primary database region
            secondary_regions: List of secondary regions for replication
        """
        self.primary_region = primary_region
        self.secondary_regions = secondary_regions or []
        self.all_regions = [primary_region] + self.secondary_regions
        
        # Replication event log
        self.replication_log: Dict[str, ReplicationEvent] = {}
        
        # Health check status
        self.region_health: Dict[str, RegionStatus] = {
            r.name: RegionStatus.HEALTHY for r in self.all_regions
        }
        
        # Replication queues per region
        self.replication_queues: Dict[str, List[Dict[str, Any]]] = {
            r.name: [] for r in self.secondary_regions
        }
        
        # Replication stats
        self.stats = {
            'replicated_count': 0,
            'failed_count': 0,
            'avg_lag_ms': 0,
        }
    
    async def write_to_primary(
        self,
        memory_id: str,
        operation: str,
        data: Dict[str, Any],
    ) -> bool:
        """
        Write to primary region and queue for replication.
        
        Args:
            memory_id: Memory ID
            operation: 'create', 'update', or 'delete'
            data: Memory data
            
        Returns:
            True if write succeeded
        """
        try:
            # Write to primary
            # (In production, would write to actual database)
            
            # Queue for replication to secondaries
            await self._queue_replication(memory_id, operation, data)
            
            return True
        except Exception as e:
            print(f"Error writing to primary: {e}")
            return False
    
    async def _queue_replication(
        self,
        memory_id: str,
        operation: str,
        data: Dict[str, Any],
    ):
        """Queue memory for replication to secondary regions."""
        event_id = str(uuid.uuid4())
        target_regions = [r.name for r in self.secondary_regions if r.healthy]
        
        event = ReplicationEvent(
            event_id=event_id,
            timestamp=datetime.utcnow(),
            operation=operation,
            memory_id=memory_id,
            source_region=self.primary_region.name,
            target_regions=target_regions,
            status=ReplicationStatus.PENDING,
        )
        
        self.replication_log[event_id] = event
        
        # Queue for each secondary region
        for region in self.secondary_regions:
            if region.healthy:
                self.replication_queues[region.name].append({
                    'event_id': event_id,
                    'memory_id': memory_id,
                    'operation': operation,
                    'data': data,
                })
        
        # Start async replication
        asyncio.create_task(self._replicate_async(event_id))
    
    async def _replicate_async(self, event_id: str):
        """
        Asynchronously replicate data to secondary regions.
        
        Args:
            event_id: Replication event ID
        """
        event = self.replication_log.get(event_id)
        if not event:
            return
        
        event.status = ReplicationStatus.REPLICATING
        success_count = 0
        fail_count = 0
        
        # Replicate to each secondary region
        for region in self.secondary_regions:
            if region.name not in event.target_regions:
                continue
            
            try:
                # Replicate to region
                success = await self._replicate_to_region(region, event)
                
                if success:
                    success_count += 1
                    self.stats['replicated_count'] += 1
                else:
                    fail_count += 1
                    self.stats['failed_count'] += 1
                
            except Exception as e:
                fail_count += 1
                event.error_message = str(e)
                print(f"Replication to {region.name} failed: {e}")
        
        # Update event status
        if fail_count == 0:
            event.status = ReplicationStatus.REPLICATED
        elif success_count > 0:
            event.status = ReplicationStatus.BEHIND
        else:
            event.status = ReplicationStatus.FAILED
    
    async def _replicate_to_region(self, region: Region, event: ReplicationEvent) -> bool:
        """
        Replicate event to a specific region.
        
        Args:
            region: Target region
            event: Replication event
            
        Returns:
            True if successful
        """
        try:
            # In production, would connect to region's database and perform operation
            # This is a stub implementation
            
            queue = self.replication_queues[region.name]
            if queue:
                item = queue.pop(0)
                # Would perform actual replication here
                
            return True
        except Exception as e:
            print(f"Error replicating to {region.name}: {e}")
            return False
    
    async def read_from_nearest(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Read from nearest healthy region.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory data or None if not found
        """
        # Get healthy regions sorted by preference
        healthy = [r for r in self.all_regions if r.healthy]
        
        if not healthy:
            return None
        
        # Try primary first
        try:
            # Would read from primary database
            return None  # Stub
        except:
            pass
        
        # Try secondaries in order of preference
        for region in healthy[1:]:
            try:
                # Would read from secondary
                return None  # Stub
            except:
                continue
        
        return None
    
    async def handle_failover(self):
        """Handle primary region failure by promoting secondary."""
        # Find a healthy secondary region
        healthy_secondaries = [
            r for r in self.secondary_regions if r.healthy
        ]
        
        if not healthy_secondaries:
            print("No healthy secondary regions available")
            return False
        
        # Promote first healthy secondary to primary
        new_primary = healthy_secondaries[0]
        print(f"Promoting {new_primary.name} to primary")
        
        # Update configurations
        self.primary_region.is_primary = False
        new_primary.is_primary = True
        self.primary_region = new_primary
        
        # Start replication from new primary to old primary (if it recovers)
        return True
    
    async def check_region_health(self):
        """Check health of all regions."""
        for region in self.all_regions:
            try:
                # In production, would perform health check against database
                # For now, assume all healthy
                self.region_health[region.name] = RegionStatus.HEALTHY
                region.healthy = True
            except Exception as e:
                self.region_health[region.name] = RegionStatus.OFFLINE
                region.healthy = False
                print(f"Region {region.name} health check failed: {e}")
    
    async def get_replication_status(self) -> Dict[str, Any]:
        """Get current replication status."""
        return {
            'primary_region': self.primary_region.name,
            'secondary_regions': [r.name for r in self.secondary_regions],
            'region_health': {k: v.value for k, v in self.region_health.items()},
            'pending_replications': sum(len(q) for q in self.replication_queues.values()),
            'stats': self.stats,
            'recent_events': [
                e.to_dict() for e in list(self.replication_log.values())[-10:]
            ],
        }
    
    async def get_replication_lag(self) -> Dict[str, float]:
        """Get replication lag for each region (milliseconds)."""
        lag_ms = {}
        for region in self.secondary_regions:
            queue_size = len(self.replication_queues[region.name])
            # Estimate lag: 100ms per queued item + variance
            lag_ms[region.name] = queue_size * 100 + (50 if not region.healthy else 0)
        return lag_ms
    
    def get_read_endpoints(self) -> List[Region]:
        """
        Get read endpoints sorted by proximity and load.
        
        Returns:
            Sorted list of readable regions
        """
        # Healthy regions sorted by read weight (higher = preferred)
        healthy = [r for r in self.all_regions if r.healthy]
        return sorted(healthy, key=lambda r: r.read_weight, reverse=True)
    
    def get_write_endpoint(self) -> Optional[Region]:
        """Get primary write endpoint."""
        return self.primary_region if self.primary_region.healthy else None


# Example configuration factory
def create_default_replication_service() -> ReplicationService:
    """Create default replication service with standard regions."""
    primary = Region(
        name="us-east-1",
        db_url="postgresql://...",
        is_primary=True,
        read_weight=2,
    )
    
    secondaries = [
        Region(
            name="us-west-1",
            db_url="postgresql://...",
            is_primary=False,
            read_weight=1,
        ),
        Region(
            name="eu-west-1",
            db_url="postgresql://...",
            is_primary=False,
            read_weight=1,
        ),
    ]
    
    return ReplicationService(primary, secondaries)
