# Self-RAG — Adaptive Retrieval with Hallucination Detection

Self-RAG extends [Corrective RAG (CRAG)](../corrective_rag/) with three additional decision nodes and HyDE query expansion, creating a more autonomous, fault-tolerant RAG pipeline with recovery loops.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  State: { query, retrieve_needed, documents, answer, grades, attempts } │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │  retrieve_or_not    │  → NO → direct_generate → END
                     └──────────┬──────────┘
                                │ YES
                     ┌──────────▼──────────┐
                     │  retrieve           │  → vector store top-k
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │  grade_relevance    │  → FAIL → rewrite → web_search
                     └──────────┬──────────┘
                                │ PASS
                     ┌──────────▼──────────┐
                     │  generate           │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │  grade_grounding    │  → FAIL → hyde_expand → retrieve (max 2x)
                     └──────────┬──────────┘
                                │ PASS
                     ┌──────────▼──────────┐
                     │  grade_answer       │  → FAIL → rewrite → retrieve (max 2x)
                     └──────────┬──────────┘
                                │ PASS
                             ┌──▼──┐
                             │ END │
                             └─────┘
```

## Decision Nodes

| Node | Purpose | Model |
|------|---------|-------|
| **RetrieveOrNot** | Skip retrieval for simple queries (math, greetings) | claude-haiku-4-5 |
| **RelevanceGrader** | Grade document relevance (RELEVANT/AMBIGUOUS/IRRELEVANT) | claude-haiku-4-5 |
| **HallucinationGrader** | Check if answer is grounded in retrieved docs | claude-haiku-4-5 |
| **AnswerGrader** | Check if answer addresses the original question | claude-haiku-4-5 |
| **HyDEQueryExpander** | Generate hypothetical document for better retrieval | claude-haiku-4-5 |

## Quick Start

### Mock mode (no API keys needed)
```bash
cd pipelines/self_rag
streamlit run demo.py
```

### With real APIs
```bash
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
streamlit run demo.py
# Uncheck "Use mock mode" in sidebar
```

### Docker
```bash
cd pipelines/self_rag
docker compose -f docker-compose.override.yml up
# Open http://localhost:8502
```

## Evaluation

Compare Naive RAG vs CRAG vs Self-RAG on the test dataset:
```bash
python pipelines/self_rag/evaluate.py
```

### Expected RAGAS Results

| Metric | Naive RAG | CRAG | Self-RAG | Delta |
|--------|-----------|------|----------|-------|
| Faithfulness | ~0.72 | ~0.82 | ~0.88 | +0.16 |
| Answer Relevancy | ~0.70 | ~0.78 | ~0.82 | +0.12 |
| Hallucination Rate | ~18% | ~12% | ~7% | -11% |

## Testing

```bash
# Unit tests
pytest pipelines/self_rag/tests/unit/ -v

# Integration tests
pytest pipelines/self_rag/tests/integration/ -v

# All tests with coverage
pytest pipelines/self_rag/tests/ -v --cov=pipelines/self_rag/src
```

## Tech Stack

- **Orchestration**: LangGraph with conditional edges + state persistence
- **LLM (generation)**: claude-sonnet-4-6
- **LLM (all grading)**: claude-haiku-4-5
- **Query expansion**: HyDE (Hypothetical Document Embeddings)
- **Observability**: LangFuse tracing
- **Evaluation**: RAGAS metrics
