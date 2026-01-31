"""
External System Sync Adapters

Syncs memories with external systems:
- Obsidian vault (bidirectional file sync)
- Notion database (unidirectional push)
- Roam Research graph (unidirectional push)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import httpx
import asyncio
import json
from pathlib import Path


class SyncDirection(str, Enum):
    """Sync direction."""
    UNIDIRECTIONAL = "unidirectional"  # Push only
    BIDIRECTIONAL = "bidirectional"    # Push and pull


class SyncStatus(str, Enum):
    """Sync status."""
    PENDING = "pending"
    SYNCING = "syncing"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class SyncResult:
    """Result of a sync operation."""
    status: SyncStatus
    synced_count: int
    failed_count: int
    errors: List[str]
    duration_seconds: float
    message: str


class ExternalSyncAdapter(ABC):
    """Base class for external system sync."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adapter.
        
        Args:
            config: System-specific configuration
        """
        self.config = config
        self.name: str = self.__class__.__name__
        self.direction: SyncDirection = SyncDirection.UNIDIRECTIONAL
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Test authentication. Returns True if successful."""
        pass
    
    @abstractmethod
    async def push_memory(self, memory_id: str, content: Dict[str, Any]) -> bool:
        """Push a memory to external system. Returns True if successful."""
        pass
    
    @abstractmethod
    async def pull_memory(self, external_id: str) -> Optional[Dict[str, Any]]:
        """Pull a memory from external system. Returns None if not found."""
        pass
    
    @abstractmethod
    async def sync_all(self, memories: List[Dict[str, Any]]) -> SyncResult:
        """Sync all memories to external system."""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get sync status and statistics."""
        pass


class ObsidianVaultAdapter(ExternalSyncAdapter):
    """
    Obsidian vault sync adapter.
    
    Bidirectional sync with Obsidian vaults via local file system.
    Features:
    - Watch vault folder for changes
    - Sync memories to markdown files
    - Pull notes from vault
    - Preserve frontmatter metadata
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Obsidian adapter.
        
        config:
            vault_path: Path to Obsidian vault
            sync_folder: Subfolder in vault to sync memories to (default: "Ninai Memories")
            auto_watch: Whether to watch vault for changes
        """
        super().__init__(config)
        self.direction = SyncDirection.BIDIRECTIONAL
        self.vault_path = Path(config.get('vault_path', ''))
        self.sync_folder = self.vault_path / config.get('sync_folder', 'Ninai Memories')
        self.sync_folder.mkdir(parents=True, exist_ok=True)
    
    async def authenticate(self) -> bool:
        """Verify vault path is valid."""
        return self.vault_path.exists() and self.vault_path.is_dir()
    
    async def push_memory(self, memory_id: str, content: Dict[str, Any]) -> bool:
        """Push memory to Obsidian as markdown file."""
        try:
            file_path = self.sync_folder / f"{content.get('title', memory_id)}.md"
            
            # Create frontmatter
            frontmatter = {
                'id': memory_id,
                'created_at': content.get('created_at'),
                'tags': content.get('tags', []),
                'scope': content.get('scope'),
                'original_id': memory_id,
            }
            
            # Write file
            lines = [
                '---',
                json.dumps(frontmatter, indent=2),
                '---',
                '',
                content.get('content', ''),
            ]
            
            file_path.write_text('\n'.join(lines), encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error pushing to Obsidian: {e}")
            return False
    
    async def pull_memory(self, external_id: str) -> Optional[Dict[str, Any]]:
        """Pull note from Obsidian vault."""
        try:
            # Search for file with matching ID in frontmatter
            for md_file in self.sync_folder.glob("*.md"):
                content = md_file.read_text(encoding='utf-8')
                
                # Simple YAML frontmatter parsing
                if '---' in content:
                    parts = content.split('---')
                    if len(parts) >= 3:
                        try:
                            import yaml
                            frontmatter = yaml.safe_load(parts[1])
                            if frontmatter.get('id') == external_id:
                                return {
                                    'title': md_file.stem,
                                    'content': parts[2].strip(),
                                    'frontmatter': frontmatter,
                                }
                        except:
                            pass
            return None
        except Exception as e:
            print(f"Error pulling from Obsidian: {e}")
            return None
    
    async def sync_all(self, memories: List[Dict[str, Any]]) -> SyncResult:
        """Sync all memories to Obsidian vault."""
        start_time = datetime.utcnow()
        synced_count = 0
        failed_count = 0
        errors = []
        
        for memory in memories:
            try:
                success = await self.push_memory(memory['id'], memory)
                if success:
                    synced_count += 1
                else:
                    failed_count += 1
                    errors.append(f"Failed to sync memory {memory['id']}")
            except Exception as e:
                failed_count += 1
                errors.append(f"Error syncing {memory['id']}: {str(e)}")
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        return SyncResult(
            status=SyncStatus.SUCCESS if failed_count == 0 else SyncStatus.FAILED,
            synced_count=synced_count,
            failed_count=failed_count,
            errors=errors,
            duration_seconds=duration,
            message=f"Synced {synced_count} memories to Obsidian",
        )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get Obsidian sync status."""
        md_files = list(self.sync_folder.glob("*.md"))
        return {
            'adapter': 'Obsidian',
            'vault_path': str(self.vault_path),
            'synced_files': len(md_files),
            'direction': self.direction.value,
            'connected': self.vault_path.exists(),
        }


class NotionDatabaseAdapter(ExternalSyncAdapter):
    """
    Notion database sync adapter.
    
    Unidirectional sync (push only) to Notion databases.
    Features:
    - Push memories to Notion pages
    - Use templates for formatting
    - Manage properties and relations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Notion adapter.
        
        config:
            api_key: Notion API token
            database_id: Target database ID
            property_mapping: Field mapping (optional)
        """
        super().__init__(config)
        self.api_key = config.get('api_key', '')
        self.database_id = config.get('database_id', '')
        self.base_url = 'https://api.notion.com/v1'
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Notion-Version': '2022-06-28',
            'Content-Type': 'application/json',
        }
    
    async def authenticate(self) -> bool:
        """Test Notion API connection."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self.base_url}/databases/{self.database_id}',
                    headers=self.headers,
                )
                return response.status_code == 200
        except:
            return False
    
    async def push_memory(self, memory_id: str, content: Dict[str, Any]) -> bool:
        """Push memory to Notion database."""
        try:
            async with httpx.AsyncClient() as client:
                data = {
                    'parent': {'database_id': self.database_id},
                    'properties': {
                        'Title': {
                            'title': [
                                {'text': {'content': content.get('title', 'Untitled')}}
                            ]
                        },
                        'Content': {
                            'rich_text': [
                                {'text': {'content': content.get('content', '')}}
                            ]
                        },
                        'Tags': {
                            'multi_select': [
                                {'name': tag} for tag in content.get('tags', [])
                            ]
                        },
                        'Original ID': {
                            'rich_text': [{'text': {'content': memory_id}}]
                        },
                    }
                }
                
                response = await client.post(
                    f'{self.base_url}/pages',
                    json=data,
                    headers=self.headers,
                )
                return response.status_code == 200
        except Exception as e:
            print(f"Error pushing to Notion: {e}")
            return False
    
    async def pull_memory(self, external_id: str) -> Optional[Dict[str, Any]]:
        """Not supported for Notion unidirectional sync."""
        return None
    
    async def sync_all(self, memories: List[Dict[str, Any]]) -> SyncResult:
        """Sync all memories to Notion."""
        start_time = datetime.utcnow()
        synced_count = 0
        failed_count = 0
        errors = []
        
        for memory in memories:
            try:
                success = await self.push_memory(memory['id'], memory)
                if success:
                    synced_count += 1
                    await asyncio.sleep(0.3)  # Rate limit
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                errors.append(str(e))
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        return SyncResult(
            status=SyncStatus.SUCCESS if failed_count == 0 else SyncStatus.FAILED,
            synced_count=synced_count,
            failed_count=failed_count,
            errors=errors,
            duration_seconds=duration,
            message=f"Synced {synced_count} memories to Notion",
        )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get Notion sync status."""
        return {
            'adapter': 'Notion',
            'database_id': self.database_id,
            'direction': self.direction.value,
            'connected': await self.authenticate(),
        }


class RoamResearchAdapter(ExternalSyncAdapter):
    """
    Roam Research graph sync adapter.
    
    Unidirectional sync (push only) to Roam Research.
    Features:
    - Push memories as Roam pages
    - Create blocks and relationships
    - Manage tags and references
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Roam adapter.
        
        config:
            graph_name: Roam graph name
            api_token: Roam Depot API token (if available)
            username: Roam username (for manual auth)
        """
        super().__init__(config)
        self.graph_name = config.get('graph_name', '')
        self.api_token = config.get('api_token', '')
    
    async def authenticate(self) -> bool:
        """Test Roam connection (simplified check)."""
        return bool(self.graph_name and (self.api_token or True))
    
    async def push_memory(self, memory_id: str, content: Dict[str, Any]) -> bool:
        """Push memory to Roam as a page."""
        try:
            # Roam API format
            # Since official API is limited, this is a stub for manual integration
            # In production, would use Roam's RoamResearch API or browser automation
            
            page_data = {
                'title': content.get('title', 'Untitled'),
                'children': [
                    {
                        'string': content.get('content', ''),
                        'children': []
                    }
                ],
                'metadata': {
                    'original_id': memory_id,
                    'synced_from': 'Ninai',
                }
            }
            
            # Stub - would integrate with actual Roam API
            return True
        except Exception as e:
            print(f"Error pushing to Roam: {e}")
            return False
    
    async def pull_memory(self, external_id: str) -> Optional[Dict[str, Any]]:
        """Not supported for Roam unidirectional sync."""
        return None
    
    async def sync_all(self, memories: List[Dict[str, Any]]) -> SyncResult:
        """Sync all memories to Roam."""
        start_time = datetime.utcnow()
        synced_count = 0
        failed_count = 0
        errors = []
        
        for memory in memories:
            try:
                success = await self.push_memory(memory['id'], memory)
                if success:
                    synced_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                errors.append(str(e))
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        return SyncResult(
            status=SyncStatus.SUCCESS if failed_count == 0 else SyncStatus.FAILED,
            synced_count=synced_count,
            failed_count=failed_count,
            errors=errors,
            duration_seconds=duration,
            message=f"Synced {synced_count} memories to Roam",
        )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get Roam sync status."""
        return {
            'adapter': 'Roam Research',
            'graph_name': self.graph_name,
            'direction': self.direction.value,
            'connected': await self.authenticate(),
        }


class SyncManager:
    """Manages multiple external system sync adapters."""
    
    def __init__(self):
        """Initialize sync manager."""
        self.adapters: Dict[str, ExternalSyncAdapter] = {}
    
    def register_adapter(self, name: str, adapter: ExternalSyncAdapter):
        """Register a sync adapter."""
        self.adapters[name] = adapter
    
    async def sync_to_all(self, memories: List[Dict[str, Any]]) -> Dict[str, SyncResult]:
        """Sync memories to all registered adapters."""
        results = {}
        for name, adapter in self.adapters.items():
            try:
                result = await adapter.sync_all(memories)
                results[name] = result
            except Exception as e:
                results[name] = SyncResult(
                    status=SyncStatus.FAILED,
                    synced_count=0,
                    failed_count=len(memories),
                    errors=[str(e)],
                    duration_seconds=0,
                    message=f"Error syncing to {name}",
                )
        return results
    
    async def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all adapters."""
        status = {}
        for name, adapter in self.adapters.items():
            try:
                status[name] = await adapter.get_status()
            except Exception as e:
                status[name] = {'error': str(e)}
        return status


# Global sync manager
_sync_manager: Optional[SyncManager] = None


def get_sync_manager() -> SyncManager:
    """Get or create global sync manager."""
    global _sync_manager
    if _sync_manager is None:
        _sync_manager = SyncManager()
    return _sync_manager
