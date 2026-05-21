"""
NINAI v2 — Graph-RAG + DNC Cognitive Architecture

Three-component design:
  A. Reasoning Engine  — stateless Ollama LLM (zero-shot, structured JSON output)
  B. Knowledge Graph   — FalkorDB chronological graph (hippocampus)
  C. DNC Memory Router — read/write/purge weighting layer

Three-phase execution loop per interaction:
  Phase 1 (Read)   — dual-path retrieval: Qdrant dense + FalkorDB subgraph
  Phase 2 (Infer)  — LLM generation with injected subgraph context + node citations
  Phase 3 (Learn)  — graph write-back, edge weight strengthening, decay, pruning

Active only when NINAI_ENGINE_VERSION=v2.
All v1 agents and routes remain untouched and callable when version=v1.
"""
