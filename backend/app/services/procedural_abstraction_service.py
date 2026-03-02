"""
PR-7: Compositional Generalization service layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4


class ProceduralAbstractionService:
    """Extract, instantiate, and compose abstract procedures."""

    async def extract_abstract_procedure(
        self,
        playbook_id: str,
        title: str,
        description: str,
        steps: List[str],
        target_abstraction_level: int = 1,
    ) -> Dict[str, Any]:
        abstraction_level = max(0, min(3, target_abstraction_level))

        abstract_title = title
        abstract_description = description
        abstract_steps = list(steps)
        parameters: Dict[str, str] = {}

        if abstraction_level >= 1:
            replacements = {
                "aws": "[infrastructure]",
                "eks": "[platform]",
                "kubernetes": "[platform]",
                "docker image": "[artifact]",
                "image": "[artifact]",
                "cluster": "[platform]",
            }

            lowered_title = abstract_title.lower()
            for source, target in replacements.items():
                if source in lowered_title:
                    parameters[target.strip("[]")] = "string"

            def generalize(text: str) -> str:
                out = text
                for source, target in replacements.items():
                    out = out.replace(source, target).replace(source.upper(), target)
                    out = out.replace(source.title(), target)
                return out

            abstract_title = generalize(abstract_title)
            abstract_description = generalize(abstract_description)
            abstract_steps = [generalize(s) for s in abstract_steps]

        if abstraction_level >= 2:
            abstract_steps = [
                s.replace("deploy", "apply").replace("Deploy", "Apply") for s in abstract_steps
            ]

        if abstraction_level >= 3:
            abstract_steps = [
                "Prepare prerequisites",
                "Execute core operation",
                "Validate outcomes",
                "Rollback or iterate if validation fails",
            ]
            abstract_description = "Principle-level procedure: preparation, execution, validation, adaptation."

        return {
            "id": str(uuid4()),
            "concrete_playbook_id": playbook_id,
            "abstraction_level": abstraction_level,
            "title": abstract_title,
            "description": abstract_description,
            "parameters": parameters,
            "prerequisites": ["required tools installed", "target environment available"],
            "postconditions": ["operation applied", "health validated"],
            "invariants": ["preserve data integrity", "maintain rollback path"],
            "instances": [playbook_id],
            "steps": abstract_steps,
            "created_at": datetime.utcnow().isoformat(),
        }

    async def instantiate_abstract(
        self,
        abstract_procedure: Dict[str, Any],
        parameters: Dict[str, str],
    ) -> Dict[str, Any]:
        def materialize(text: str) -> str:
            output = text
            for key, value in parameters.items():
                output = output.replace(f"[{key}]", str(value))
            return output

        steps = [materialize(s) for s in abstract_procedure.get("steps", [])]
        title = materialize(abstract_procedure.get("title", "Instantiated Procedure"))
        description = materialize(abstract_procedure.get("description", ""))

        return {
            "id": str(uuid4()),
            "source_abstract_id": abstract_procedure.get("id"),
            "title": title,
            "description": description,
            "steps": steps,
            "parameters_used": parameters,
            "created_at": datetime.utcnow().isoformat(),
        }

    async def compose_procedures(
        self,
        abstract_procedures: List[Dict[str, Any]],
        glue_logic: str,
    ) -> Dict[str, Any]:
        composed_steps: List[str] = []
        source_ids: List[str] = []

        for proc in abstract_procedures:
            source_ids.append(proc.get("id", "unknown"))
            composed_steps.extend(proc.get("steps", []))

        if glue_logic:
            composed_steps.append(f"Glue logic: {glue_logic}")

        return {
            "id": str(uuid4()),
            "title": "Composed Procedure",
            "description": "Composite built from multiple abstract procedures.",
            "steps": composed_steps,
            "source_procedures": source_ids,
            "glue_logic": glue_logic,
            "created_at": datetime.utcnow().isoformat(),
        }


class AnalogyService:
    """Discover and use analogies for cross-domain transfer."""

    async def find_analogies(
        self,
        problem_domain: str,
        target_domain: str,
    ) -> List[Dict[str, Any]]:
        base_mappings = {
            "deployment": {
                "customer_onboarding": {
                    "artifact": "onboarding_package",
                    "platform": "customer_environment",
                    "health_check": "activation_check",
                },
                "incident_response": {
                    "artifact": "fix_plan",
                    "platform": "production_system",
                    "health_check": "stability_check",
                },
            }
        }

        mapped = base_mappings.get(problem_domain, {}).get(target_domain, {})
        if not mapped:
            mapped = {
                "prepare": "prepare",
                "execute": "execute",
                "validate": "validate",
            }

        similarity = 0.82 if mapped else 0.55
        applicability = "full" if similarity >= 0.8 else "partial"

        return [
            {
                "id": str(uuid4()),
                "source_domain": problem_domain,
                "target_domain": target_domain,
                "structural_similarity": similarity,
                "mapped_concepts": mapped,
                "constraints": [
                    "verify domain-specific compliance requirements",
                    "adjust terminology for target stakeholders",
                ],
                "applicability": applicability,
                "discovered_at": datetime.utcnow().isoformat(),
            }
        ]

    async def transfer_solution(
        self,
        source_playbook: Dict[str, Any],
        source_domain: str,
        target_domain: str,
        problem_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        analogies = await self.find_analogies(source_domain, target_domain)
        selected = analogies[0]
        mapping = selected.get("mapped_concepts", {})

        transferred_steps = []
        for step in source_playbook.get("steps", []):
            updated = step
            for src, tgt in mapping.items():
                updated = updated.replace(src, tgt)
            transferred_steps.append(updated)

        if problem_context:
            transferred_steps.append(f"Context adaptation: {problem_context}")

        return {
            "id": str(uuid4()),
            "source_domain": source_domain,
            "target_domain": target_domain,
            "analogy_id": selected.get("id"),
            "transferred_title": f"{source_playbook.get('title', 'Playbook')} ({target_domain} adaptation)",
            "transferred_steps": transferred_steps,
            "mapping_used": mapping,
            "confidence": selected.get("structural_similarity", 0.6),
            "created_at": datetime.utcnow().isoformat(),
        }
