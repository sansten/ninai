# Backward-compatibility shim — import from inference_engine instead.
from app.v2.llm.inference_engine import (  # noqa: F401
    InferenceEngine,
    InferenceEngine as OllamaReasoningEngine,
    InferenceResult,
)
