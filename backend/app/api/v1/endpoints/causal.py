"""
Causal Reasoning API Endpoints (PR-1)
=====================================

FastAPI endpoints for causal memory graph operations.

These endpoints enable reasoning about causation, going beyond correlation:
- discover: Find causal relationships in episodes
- predict: Forecast outcomes given causes
- explain: Find root causes of effects
- validate: Learn from validated predictions
- counterfactual: Explore hypothetical scenarios
- do_calculus: Estimate interventional effects
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.endpoints.auth import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.causal import (
    CounterfactualRequest,
    CounterfactualResponse,
    DiscoverCausalEdgesRequest,
    DiscoverCausalEdgesResponse,
    DoCalculusRequest,
    DoCalculusResponse,
    ExplainRequest,
    ExplainResponse,
    PredictionRequest,
    PredictionResponse,
    ValidateEdgeRequest,
    ValidateEdgeResponse,
)
from app.services.causal_reasoning_service import CausalReasoningService

router = APIRouter(prefix="/v1/causal", tags=["causal"])


@router.post("/discover/{episode_id}", response_model=DiscoverCausalEdgesResponse)
async def discover_causal_edges(
    episode_id: str,
    request: DiscoverCausalEdgesRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Discover causal edges within an episode.

    This endpoint analyzes an episode to identify potential causal relationships
    between facts. It uses three discovery signals:

    1. **Temporal precedence**: Earlier facts might cause later facts
    2. **Correlational**: Co-occurrence suggests causation
    3. **LLM inference**: "What causes what in this episode?"

    The discovered edges have strength estimates based on discovery signal confidence.

    ## Why This Works

    General intelligence requires moving from Observation → Intervention → Counterfactual.
    This endpoint discovers edges so agents can later predict (intervention) and
    explore hypotheticals (counterfactual).

    Args:
        episode_id: The episode to analyze
        request: Discovery request with auto_save flag
        current_user: Authenticated user
        db: Database session

    Returns:
        Discovered edges with strengths and mechanisms

    Example:
        ```json
        {
            "episode_id": "ep_123",
            "edges_discovered": 3,
            "edges": [
                {
                    "id": "edge_1",
                    "cause_entity_id": "fact_1",
                    "effect_entity_id": "fact_2",
                    "mechanism": "temporal",
                    "strength": 0.85
                }
            ]
        }
        ```
    """
    try:
        service = CausalReasoningService(db, current_user.organization_id)

        # Discover edges in the episode
        edges = await service.discover_causal_edges_from_episode(episode_id)

        return DiscoverCausalEdgesResponse(
            episode_id=episode_id,
            edges_discovered=len(edges),
            edges=edges,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Causal discovery failed: {str(e)}")


@router.get("/edges", response_model=List[dict])
async def get_causal_edges(
    from_entity: Optional[str] = Query(None, description="Source entity ID"),
    to_entity: Optional[str] = Query(None, description="Target entity ID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve causal edges matching criteria.

    Filters edges by cause/effect entities. Can retrieve:
    - All outgoing edges from an entity (from_entity specified)
    - All incoming edges to an entity (to_entity specified)
    - All edges between two entities (both specified)

    Args:
        from_entity: Filter by cause entity
        to_entity: Filter by effect entity
        current_user: Authenticated user
        db: Database session

    Returns:
        List of matching edges with full details
    """
    try:
        if not from_entity and not to_entity:
            raise HTTPException(
                status_code=400,
                detail="Must specify from_entity or to_entity",
            )

        service = CausalReasoningService(db, current_user.organization_id)
        edges = await service.get_edges(
            cause_entity_id=from_entity,
            effect_entity_id=to_entity,
        )

        return [edge.to_dict() for edge in edges]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictionResponse)
async def predict_outcome(
    request: PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Predict outcomes given causes (Interventional reasoning).

    This implements the second level of Pearl's Causal Ladder:
    "What if I DO X?" (vs. just observing)

    Given a set of causes, predicts the most likely effect by:
    1. Finding all causal paths from causes to target
    2. Weighting paths by edge strength
    3. Estimating probability of target outcome

    ## Why This Matters

    Planning requires interventional reasoning. We can:
    - Ask "If I take action X, what will happen?"
    - Trace causal chains to predict downstream effects
    - Identify necessary conditions for goals

    Args:
        request: Cause IDs and target to predict
        current_user: Authenticated user
        db: Database session

    Returns:
        Predicted outcome with probability and causal chain

    Example:
        ```
        Causes: ["memory_user_said_hello", "memory_i_responded"]
        Target: "user_satisfaction"
        
        Returns:
        {
            "outcome_id": "user_satisfaction",
            "probability": 0.78,
            "confidence": 0.85,
            "causal_chain": ["said_hello", "responded", "user_satisfaction"]
        }
        ```
    """
    try:
        service = CausalReasoningService(db, current_user.organization_id)

        result = await service.predict(
            cause_fact_ids=request.cause_ids,
            target_fact_id=request.target_id,
            max_depth=request.max_depth,
        )

        return PredictionResponse(
            outcome_id=result.outcome_id,
            probability=result.probability,
            confidence=result.confidence,
            causal_chain=result.causal_chain or [],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/explain/{effect_id}", response_model=ExplainResponse)
async def explain_effect(
    effect_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Explain an effect by finding root causes (Causal explanation).

    Answers "Why did X happen?" by:
    1. Finding all causal edges leading to the effect
    2. Recursively finding causes of those causes
    3. Returning root causes ranked by total strength

    ## Why This Matters

    Debugging and learning require causal explanation:
    - "Why did the prediction fail?"
    - "What caused user satisfaction?"
    - "What leads to memory consolidation?"

    Enables agents to trace failures to decisions and improve.

    Args:
        effect_id: Effect/outcome to explain
        current_user: Authenticated user
        db: Database session

    Returns:
        Root causes ranked by strength

    Example:
        ```
        Effect: "agent_decision_good"
        
        Returns:
        {
            "effect_id": "agent_decision_good",
            "root_causes": [
                {
                    "cause_id": "memory_past_success",
                    "strength": 0.92,
                    "chain_depth": 2
                },
                {
                    "cause_id": "user_clear_intent",
                    "strength": 0.78,
                    "chain_depth": 1
                }
            ]
        }
        ```
    """
    try:
        service = CausalReasoningService(db, current_user.organization_id)

        explanation = await service.explain(effect_fact_id=effect_id)

        return ExplainResponse(
            effect_id=explanation.effect_id,
            root_causes=explanation.root_causes or [],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.post("/counterfactual", response_model=CounterfactualResponse)
async def counterfactual_query(
    request: CounterfactualRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Explore counterfactual scenarios (Hypothetical reasoning).

    This implements the third level of Pearl's Causal Ladder:
    "What if I HAD done X?" (exploring alternate histories)

    Creates a counterfactual scenario (hypothetical) and tracks it:
    - When the condition manifests in reality, record actual outcomes
    - Calculate accuracy of predictions
    - Learn from mismatches to improve causal understanding

    ## Why This Matters

    Learning requires counterfactual reasoning:
    - "I predicted X would happen if I did Y. Did it actually happen?"
    - "What would've happened if I chose differently?"
    - Improves causal edge strength based on prediction accuracy

    This is how agents improve via experience: predict → observe → validate.

    Args:
        request: Condition and query
        current_user: Authenticated user
        db: Database session

    Returns:
        Created counterfactual scenario ID

    Example:
        ```
        Condition: "If I had responded earlier"
        Query: {"outcome": "user_satisfaction"}
        
        Returns:
        {
            "scenario_id": "cf_456",
            "condition": "If I had responded earlier",
            "status": "pending"
        }
        
        When reality manifests:
        {
            "scenario_id": "cf_456",
            "condition": "If I had responded earlier",
            "status": "resolved",
            "predicted": {"user_satisfaction": 0.85},
            "actual": {"user_satisfaction": 0.88},
            "accuracy": 0.97
        }
        ```
    """
    try:
        service = CausalReasoningService(db, current_user.organization_id)

        scenario = await service.counterfactual_query(
            condition=request.condition,
            query=request.query,
        )

        return CounterfactualResponse(
            scenario_id=scenario.get("scenario_id"),
            condition=scenario.get("condition"),
            status=scenario.get("status"),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Counterfactual query failed: {str(e)}")


@router.get("/do-calculus", response_model=DoCalculusResponse)
async def estimate_causal_effect(
    treatment: str = Query(..., description="Intervention variable"),
    outcome: str = Query(..., description="Outcome to measure"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Estimate causal effects using do-calculus (Pearl's do-operator).

    Computes E[Y | do(X=x)] - "the expected Y if we intervene to set X"

    Differs from observational inference P(Y|X) because interventions break
    confounding. Eliminates backdoor paths and estimates true causal effect.

    ## Why This Matters

    Optimal decision-making requires causal effects:
    - "If I change X, how much will Y improve?"
    - "What's the causal impact (not just correlation)?"
    - Avoids Simpson's paradox and confounding errors

    Classical example: Treatment A correlates with worse outcomes, but has
    positive causal effect. Do-calculus distinguishes these.

    Args:
        treatment: Intervention variable
        outcome: Outcome to measure
        current_user: Authenticated user
        db: Database session

    Returns:
        Causal effect estimate and number of paths considered

    Example:
        ```
        Treatment: "increased_context_window"
        Outcome: "task_success_rate"
        
        Returns:
        {
            "treatment": "increased_context_window",
            "outcome": "task_success_rate",
            "causal_effect": 0.23,
            "num_paths": 5
        }
        
        Interpretation: Interventional increase in context window causally
        increases task success by 23 percentage points.
        ```
    """
    try:
        service = CausalReasoningService(db, current_user.organization_id)

        result = await service.do_calculus(treatment=treatment, outcome_var=outcome)

        return DoCalculusResponse(
            treatment=result["treatment"],
            outcome=result["outcome"],
            causal_effect=result["causal_effect"],
            num_paths=result["num_paths"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Do-calculus failed: {str(e)}")


@router.post("/validate", response_model=ValidateEdgeResponse)
async def validate_edge(
    request: ValidateEdgeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Validate a causal edge based on observed outcomes.

    When a prediction is made and later verified/falsified, update edge strength:
    - Success: Increase strength (edge was correct)
    - Failure: Decrease strength (edge was wrong)

    Uses Bayesian averaging to accumulate evidence:
    ```
    new_strength = (old_strength * count + adjustment) / (count + 1)
    ```

    Also tracks validation_count and invalidation_count for meta-analysis:
    - Edges with many validations are reliable
    - Edges with mixed validation are uncertain
    - Completely invalidated edges can be removed

    ## Why This Matters

    Causal edges are hypotheses. This endpoint turns them into tested knowledge:
    - Observing outcomes validates predictions
    - Strength gradually approaches ground truth
    - Enables continuous improvement via experience

    This is how Ninai learns from the world: discover → predict → observe → validate.

    Args:
        request: Edge ID, success flag, and strength adjustment
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated edge with new strength and validation counts

    Example:
        ```
        Edge: "user_context → model_accuracy" (strength 0.6)
        Observed: Accuracy improved 23% when context increased
        
        Request:
        {
            "edge_id": "edge_123",
            "success": true,
            "strength_adjustment": 0.25
        }
        
        Returns:
        {
            "id": "edge_123",
            "strength": 0.725,
            "validation_count": 5,
            "invalidation_count": 1,
            "last_validated_at": "2026-03-02T10:30:00Z"
        }
        ```
    """
    try:
        service = CausalReasoningService(db, current_user.organization_id)

        edge = await service.validate_edge(
            edge_id=request.edge_id,
            success=request.success,
            strength_adjustment=request.strength_adjustment,
        )

        return ValidateEdgeResponse(
            id=edge.id,
            strength=edge.strength,
            validation_count=edge.validation_count,
            invalidation_count=edge.invalidation_count,
            last_validated_at=edge.updated_at,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
