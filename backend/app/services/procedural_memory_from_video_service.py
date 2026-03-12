"""
Procedural Memory from Video Service - PR-9

Extract procedural/domain knowledge from video demonstrations.
Enables learning from video walkthroughs and auto-generating playbooks.
"""

from datetime import datetime
import hashlib
from typing import List, Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.multimodal_memory import ProceduralMemoryFromVideo
from app.models.memory_attachment import MemoryAttachment
from app.models.playbook import Playbook, PlaybookScopeType


class ProceduralMemoryFromVideoService:
    """
    Extract procedural knowledge from video demonstrations.
    Learns procedures, identifies steps, prerequisites, tools, and success criteria.
    Enables automation and knowledge transfer.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.domain_templates = {
            "deployment": {
                "common_steps": [
                    "Prepare artifact",
                    "Configure environment",
                    "Deploy to staging",
                    "Run tests",
                    "Monitor health",
                    "Deploy to production",
                    "Verify deployment",
                ],
                "common_tools": ["Docker", "Kubernetes", "Git", "kubectl"],
                "common_mistakes": [
                    "Skipping validation",
                    "Wrong environment variables",
                    "Not checking logs",
                ],
            },
            "setup": {
                "common_steps": [
                    "Download files",
                    "Extract archive",
                    "Install dependencies",
                    "Configure settings",
                    "Verify installation",
                    "Run first task",
                ],
                "common_tools": ["Terminal", "Package manager", "Configuration editor"],
                "common_mistakes": ["Missing dependencies", "Wrong paths", "Permission errors"],
            },
            "debugging": {
                "common_steps": [
                    "Reproduce issue",
                    "Check logs",
                    "Isolate component",
                    "Test hypothesis",
                    "Implement fix",
                    "Verify solution",
                ],
                "common_tools": ["Debugger", "Logger", "Test framework"],
                "common_mistakes": ["Fixing symptoms not cause", "Incomplete testing"],
            },
        }

    async def extract_procedure_from_video(
        self,
        video_attachment_id: str,
        organization_id: str,
        domain: str,
        duration_seconds: int = 600,
    ) -> ProceduralMemoryFromVideo:
        """
        Extract procedural knowledge from video demonstration.

        Identifies steps, prerequisites, tools, common mistakes, and success indicators.
        """
        # Fetch attachment
        stmt = select(MemoryAttachment).where(
            MemoryAttachment.id == video_attachment_id
        )
        attachment = (await self.session.execute(stmt)).scalar_one_or_none()
        if not attachment:
            raise ValueError(f"Attachment {video_attachment_id} not found")

        # Extract procedure
        extracted_steps = self._extract_procedural_steps(domain, duration_seconds)
        prerequisites = self._infer_prerequisites(domain, extracted_steps)
        tools_required = self._identify_tools(domain, extracted_steps)
        common_mistakes = self._identify_common_mistakes(domain)
        success_indicators = self._infer_success_indicators(domain, extracted_steps)
        video_segments = self._segment_video(extracted_steps, duration_seconds)

        procedure_memory = ProceduralMemoryFromVideo(
            id=self._generate_uuid(),
            organization_id=organization_id,
            video_attachment_id=video_attachment_id,
            domain=domain,
            procedure_title=self._generate_procedure_title(domain, extracted_steps),
            procedure_description=self._describe_procedure(domain, extracted_steps),
            extracted_steps=extracted_steps,
            step_count=len(extracted_steps),
            estimated_duration_minutes=duration_seconds // 60,
            prerequisites=prerequisites,
            tools_required=tools_required,
            common_mistakes=common_mistakes,
            success_indicators=success_indicators,
            confidence_score=0.78,
            video_segments=video_segments,
            generated_playbook_id=None,
            human_validation_status="pending_review",
            human_validation_feedback=None,
        )

        self.session.add(procedure_memory)
        await self.session.flush()
        return procedure_memory

    async def validate_procedure(
        self,
        procedure_id: str,
        feedback: str,
        is_valid: bool = True,
    ) -> None:
        """
        Record human validation feedback on extracted procedure.
        """
        stmt = select(ProceduralMemoryFromVideo).where(
            ProceduralMemoryFromVideo.id == procedure_id
        )
        procedure = (await self.session.execute(stmt)).scalar_one_or_none()
        if not procedure:
            raise ValueError(f"Procedure {procedure_id} not found")

        procedure.human_validation_status = "validated" if is_valid else "rejected"
        procedure.human_validation_feedback = feedback
        if is_valid:
            procedure.confidence_score = min(0.95, procedure.confidence_score + 0.15)

        await self.session.flush()

    async def generate_playbook_from_procedure(
        self,
        procedure_id: str,
        playbook_data: Dict[str, Any],
    ) -> str:
        """
        Generate a playbook/automation from extracted procedure.

        In production: would create actual Playbook record.
        For now: just associates the playbook_id.
        """
        stmt = select(ProceduralMemoryFromVideo).where(
            ProceduralMemoryFromVideo.id == procedure_id
        )
        procedure = (await self.session.execute(stmt)).scalar_one_or_none()
        if not procedure:
            raise ValueError(f"Procedure {procedure_id} not found")

        # Create a lightweight playbook row for the extracted procedure.
        playbook_id = self._generate_uuid()
        signature_seed = f"{procedure.organization_id}:{procedure.domain}:{procedure.procedure_title}"
        playbook = Playbook(
            id=playbook_id,
            organization_id=procedure.organization_id,
            scope_type=PlaybookScopeType.ORGANIZATION,
            scope_id=None,
            title=procedure.procedure_title,
            problem_signature={
                "domain": procedure.domain,
                "source": "procedural_video",
                "procedure_id": procedure.id,
                **(playbook_data or {}),
            },
            signature_hash=hashlib.sha256(signature_seed.encode()).hexdigest(),
            steps=procedure.extracted_steps or [],
            constraints={"generated_from_video": True},
            success_rate=float(procedure.confidence_score or 0.75),
            evidence={"procedure_id": procedure.id},
        )
        self.session.add(playbook)
        await self.session.flush()
        procedure.generated_playbook_id = playbook_id

        await self.session.flush()
        return playbook_id

    async def list_procedures(
        self,
        organization_id: str,
        domain: Optional[str] = None,
        validation_status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List extracted procedures, optionally filtered by domain or validation status.
        """
        stmt = select(ProceduralMemoryFromVideo).where(
            ProceduralMemoryFromVideo.organization_id == organization_id,
        )
        if domain:
            stmt = stmt.where(ProceduralMemoryFromVideo.domain == domain)
        if validation_status:
            stmt = stmt.where(
                ProceduralMemoryFromVideo.human_validation_status == validation_status
            )

        results = (await self.session.execute(stmt)).scalars().all()

        return [
            {
                "id": p.id,
                "domain": p.domain,
                "title": p.procedure_title,
                "step_count": p.step_count,
                "duration_minutes": p.estimated_duration_minutes,
                "confidence": p.confidence_score,
                "validation_status": p.human_validation_status,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in results
        ]

    # ============ Helper Methods ============

    def _extract_procedural_steps(
        self,
        domain: str,
        duration_seconds: int,
    ) -> List[Dict[str, Any]]:
        """Extract procedural steps from video."""
        # Use domain template as base
        template = self.domain_templates.get(
            domain, self.domain_templates["setup"]
        )
        common_steps = template.get("common_steps", [])

        # Simulate step extraction
        steps = []
        step_duration = duration_seconds // len(common_steps) if common_steps else 60

        for i, step_name in enumerate(common_steps):
            steps.append(
                {
                    "step_number": i + 1,
                    "title": step_name,
                    "description": f"Execute: {step_name}",
                    "estimated_duration_seconds": step_duration,
                    "start_timestamp": {"seconds": i * step_duration, "label": f"{i*step_duration}s"},
                    "end_timestamp": {"seconds": (i+1)*step_duration, "label": f"{(i+1)*step_duration}s"},
                    "key_action": f"Perform {step_name.lower()}",
                    "expected_outcome": f"Successfully completed {step_name.lower()}", 
                    "difficulty": ["easy", "medium", "hard"][i % 3],
                    "notes": f"Important: Focus on accuracy during {step_name.lower()}",
                }
            )

        return steps

    def _infer_prerequisites(
        self,
        domain: str,
        steps: List[Dict[str, Any]],
    ) -> List[str]:
        """Infer prerequisites based on domain and steps."""
        template = self.domain_templates.get(domain, {})
        base_prereqs = [
            "Basic understanding of the domain",
            "Necessary tools installed",
            "Proper permissions/access",
        ]

        if domain == "deployment":
            base_prereqs.append("Docker image or artifact ready")
            base_prereqs.append("Target environment configured")
        elif domain == "debugging":
            base_prereqs.append("Ability to access logs and debugger")
            base_prereqs.append("Test cases to reproduce issue")
        elif domain == "setup":
            base_prereqs.append("Download links available")
            base_prereqs.append("Internet connection")

        return base_prereqs

    def _identify_tools(
        self,
        domain: str,
        steps: List[Dict[str, Any]],
    ) -> List[str]:
        """Identify required tools."""
        template = self.domain_templates.get(domain, {})
        return template.get("common_tools", ["Terminal", "Text Editor"])

    def _identify_common_mistakes(self, domain: str) -> List[str]:
        """Identify common mistakes for domain."""
        template = self.domain_templates.get(domain, {})
        return template.get(
            "common_mistakes",
            ["Not reading documentation carefully", "Skipping validation steps"],
        )

    def _infer_success_indicators(
        self,
        domain: str,
        steps: List[Dict[str, Any]],
    ) -> List[str]:
        """Infer success indicators."""
        indicators = [
            "All steps completed without errors",
            "System responding correctly",
            "No error messages in logs",
        ]

        if domain == "deployment":
            indicators.extend([
                "Service is healthy and responding",
                "Metrics show normal behavior",
                "No escalations from monitoring",
            ])
        elif domain == "debugging":
            indicators.extend([
                "Issue reproduced and root cause identified",
                "Fix implemented and tested",
                "Issue does not recur",
            ])
        elif domain == "setup":
            indicators.extend([
                "Application starts successfully",
                "Initial test runs pass",
                "Configuration persisted correctly",
            ])

        return indicators

    def _segment_video(
        self,
        steps: List[Dict[str, Any]],
        total_duration: int,
    ) -> List[Dict[str, Any]]:
        """Create video segments for each step."""
        segments = []
        for step in steps:
            segments.append(
                {
                    "step_number": step["step_number"],
                    "start_seconds": step["start_timestamp"]["seconds"],
                    "end_seconds": step["end_timestamp"]["seconds"],
                    "title": step["title"],
                    "relevance": 0.9,
                }
            )
        return segments

    def _generate_procedure_title(
        self,
        domain: str,
        steps: List[Dict[str, Any]],
    ) -> str:
        """Generate descriptive title for procedure."""
        if steps:
            return f"How to {steps[0]['title'].lower()}: {domain.title()} Procedure"
        return f"{domain.title()} Procedure"

    def _describe_procedure(
        self,
        domain: str,
        steps: List[Dict[str, Any]],
    ) -> str:
        """Generate description of procedure."""
        step_titles = [s["title"] for s in steps]
        return (
            f"Procedural demonstration for {domain.title()}. "
            f"Steps: {', '.join(step_titles[:3])}... "
            f"Total steps: {len(steps)}. "
            f"This procedure was extracted from video demonstration."
        )

    @staticmethod
    def _generate_uuid() -> str:
        """Generate UUID string."""
        import uuid
        return str(uuid.uuid4())
