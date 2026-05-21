"""
V2 Graph Schema — Chronological Knowledge Graph node and edge definitions.

Node types (label : key properties):
  Entity    — concept/user/task/object nodes; the long-term semantic vocabulary
  Utterance — a single raw input or output turn in an interaction
  Action    — an action taken by the system linked to an outcome

Edge types (relationship : key properties):
  ASSOCIATED_WITH  — general semantic co-occurrence link
  CAUSED           — causal link (source led to target)
  FOLLOWED_BY      — chronological sequence (utterance chain)
  RESPONDED_TO     — utterance → entity it references
  OUTCOME_OF       — action → entity whose state changed

Association weights range [0.0, 1.0].  Decay shrinks them over time.
Nodes/edges below WEIGHT_PRUNE_THRESHOLD are archived (label set to Archived*).
"""

from dataclasses import dataclass, field
from typing import Any


WEIGHT_INITIAL: float = 0.5
WEIGHT_REINFORCE_DELTA: float = 0.1   # added per access
WEIGHT_MAX: float = 1.0
WEIGHT_PRUNE_THRESHOLD: float = 0.05  # archived when weight drops below this
DECAY_FACTOR: float = 0.95            # multiplied per decay tick


@dataclass
class EntityNode:
    id: str
    name: str
    entity_type: str          # concept | user | task | object
    tenant_id: str
    weight: float = WEIGHT_INITIAL
    access_count: int = 0
    created_at: int = 0       # unix ms
    updated_at: int = 0


@dataclass
class UtteranceNode:
    id: str
    text: str
    role: str                 # user | assistant | system
    tenant_id: str
    session_id: str
    weight: float = WEIGHT_INITIAL
    created_at: int = 0


@dataclass
class ActionNode:
    id: str
    action_type: str
    outcome: str
    tenant_id: str
    weight: float = WEIGHT_INITIAL
    created_at: int = 0


@dataclass
class GraphEdge:
    from_id: str
    to_id: str
    rel_type: str             # ASSOCIATED_WITH | CAUSED | FOLLOWED_BY | etc.
    weight: float = WEIGHT_INITIAL
    created_at: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
