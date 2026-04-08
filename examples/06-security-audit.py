"""
Example 6: Security & Audit

Demonstrates security and compliance features:
- Querying audit logs
- Data export (GDPR Article 20)
- Right-to-erasure (GDPR Article 17)
- User offboarding
- API key management
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Optional


class SecureAuditManager:
    """Manage security, audit, and compliance operations."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
    
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}
    
    async def query_audit_logs(self, start_date: Optional[str] = None, end_date: Optional[str] = None, action: Optional[str] = None) -> dict:
        """Query audit log for compliance review."""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if action:
            params["action"] = action
        
        async with self.session.get(
            f"{self.base_url}/api/v1/audit/logs",
            params=params,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Audit query failed: {resp.status}")
            return await resp.json()
    
    async def export_org_data(self, format: str = "json", include_audit: bool = True) -> dict:
        """Request organization data export (GDPR Article 20)."""
        payload = {
            "request_type": "data_export",
            "format": format,  # json, csv, xml
            "include_audit_logs": include_audit
        }
        async with self.session.post(
            f"{self.base_url}/api/v1/compliance/export/request",
            json=payload,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Export request failed: {resp.status}")
            return await resp.json()
    
    async def request_full_deletion(self, reason: str = "user_requested") -> dict:
        """Request full organization deletion (GDPR Article 17)."""
        payload = {
            "request_type": "full_deletion",
            "reason": reason
        }
        async with self.session.post(
            f"{self.base_url}/api/v1/compliance/export/request",
            json=payload,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Deletion request failed: {resp.status}")
            return await resp.json()
    
    async def list_api_keys(self) -> dict:
        """List API keys for the organization."""
        async with self.session.get(
            f"{self.base_url}/api/v1/api-keys",
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"API key listing failed: {resp.status}")
            return await resp.json()
    
    async def revoke_api_key(self, key_id: str) -> None:
        """Revoke an API key immediately."""
        async with self.session.delete(
            f"{self.base_url}/api/v1/api-keys/{key_id}",
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Revocation failed: {resp.status}")


async def main():
    """Run security and audit example."""
    
    base_url = "http://localhost:8000"
    api_key = "test-api-key"
    
    print("Ninai2 Security & Audit Example")
    print("=" * 50)
    print()
    
    try:
        async with SecureAuditManager(base_url, api_key) as audit_mgr:
            
            # 1. Query audit logs
            print("1. Querying audit logs (last 7 days)...")
            seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
            logs = await audit_mgr.query_audit_logs(start_date=seven_days_ago)
            
            log_count = len(logs.get("logs", []))
            print(f"✓ Retrieved {log_count} audit entries")
            
            if logs.get("logs"):
                sample = logs["logs"][0]
                print(f"  Sample log entry:")
                print(f"    User: {sample.get('user_id')}")
                print(f"    Action: {sample.get('action')}")
                print(f"    Resource: {sample.get('resource_type')}")
                print(f"    Timestamp: {sample.get('timestamp')}")
                print(f"    Status: {sample.get('status')}")
            print()
            
            # 2. Analyze audit events by action
            print("2. Audit log summary by action...")
            if logs.get("logs"):
                actions = {}
                for log in logs["logs"]:
                    action = log.get("action", "unknown")
                    actions[action] = actions.get(action, 0) + 1
                
                for action, count in sorted(actions.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {action}: {count}")
            print()
            
            # 3. Data export request (GDPR Article 20)
            print("3. Requesting organization data export (GDPR Article 20)...")
            try:
                export_req = await audit_mgr.export_org_data(format="json", include_audit=True)
                print(f"✓ Export request submitted")
                print(f"  Request ID: {export_req.get('request_id')}")
                print(f"  Status: {export_req.get('status')}")
                print(f"  Expected completion: {export_req.get('estimated_completion_time')}")
            except RuntimeError as e:
                print(f"⚠ Export not available: {e}")
            print()
            
            # 4. Full deletion request (GDPR Article 17)
            print("4. Full deletion request demo (GDPR Article 17)...")
            print("  ⚠ WARNING: This is a demonstration of the deletion API")
            print("  ⚠ In production, this would trigger a 30-day grace period")
            print("     before permanent deletion, with confirmation required")
            print()
            
            # 5. API key management
            print("5. API Key Management...")
            try:
                keys = await audit_mgr.list_api_keys()
                key_list = keys.get("api_keys", [])
                print(f"✓ Organization has {len(key_list)} active API keys:")
                
                for i, key in enumerate(key_list[:3], 1):  # Show first 3
                    print(f"  {i}. Key ID: {key.get('id')[:16]}...")
                    print(f"     Name: {key.get('name')}")
                    print(f"     Created: {key.get('created_at')}")
                    print(f"     Last used: {key.get('last_used_at', 'Never')}")
                    print()
                
                if len(key_list) > 3:
                    print(f"  ... and {len(key_list) - 3} more")
            except RuntimeError as e:
                print(f"⚠ API key listing skipped: {e}")
            print()
            
            # 6. Security checklist
            print("6. Security Checklist for Production:")
            checklist = [
                ("Encryption in transit (TLS)", "enabled"),
                ("API key rotation (quarterly)", "recommended"),
                ("Audit log review (monthly)", "recommended"),
                ("Access control lists (ACLs)", "enabled"),
                ("Multi-factor authentication (MFA)", "optional"),
                ("Data backup (daily)", "configured"),
                ("Incident response plan", "required"),
                ("Vulnerability scanning", "recommended")
            ]
            
            for item, status in checklist:
                icon = "✓" if status == "enabled" else "→"
                print(f"  {icon} {item}: {status}")
            print()
            
            print("✓ Security & audit example completed successfully!")
            print("\nKey Reminders:")
            print("  - Review audit logs regularly (compliance requirement)")
            print("  - Rotate API keys quarterly")
            print("  - Implement incident response procedures")
            print("  - Honor GDPR deletion requests within 30 days")
            print("  - Keep backup credentials in secure vault")
    
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check API_KEY has audit/compliance permissions (org_admin)")
        print("  2. Verify backend audit logging is enabled")
        print("  3. Check database has audit_events table")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
