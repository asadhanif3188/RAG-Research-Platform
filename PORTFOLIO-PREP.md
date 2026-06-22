# Portfolio Preparation: RAG Research Platform

Actionable checklist for presenting this project on GitHub and in your CV/resume.

---

## 1. What to KEEP in the Repo

These are strengths — make sure they stay front and centre:

| Asset | Why it matters |
|-------|---------------|
| **5 pipeline strategies** in one monorepo | Shows breadth (naive → agentic → multimodal → graph) and architectural thinking |
| `shared/` library with models, storage, embeddings, eval | Demonstrates code reuse, clean separation of concerns |
| FastAPI router + middleware | Production-style API design with cost tracking |
| Chainlit UI with A/B compare + metrics dashboard | Proves you ship end-to-end, not just notebooks |
| Unit + integration + e2e tests | Signals engineering discipline |
| CI workflows (ci.yml, eval.yml, integration-tests.yml) | Shows DevOps awareness |
| docker-compose.yml | One-command infra setup = reviewer-friendly |
| `.env.example` | Good practice, already present |
| Per-pipeline READMEs | Helps reviewers navigate without reading all code |

## ~~2. What to REMOVE or GITIGNORE Before Making Public~~ Done

## 3. What to ADD

### 3a. Screenshots / GIF Demo (High Impact)

Recruiters and hiring managers often spend < 60 seconds on a repo. A visual demo is the single highest-ROI addition.

- **Add a `docs/screenshots/` folder** with:
  - Chainlit UI — pipeline selector view
  - A/B comparison side-by-side result
  - Metrics dashboard showing RAGAS scores
  - Terminal showing `docker-compose up` → API running
- **Record a 30-60 second GIF or MP4** of a full query flow (select pipeline → ask question → see answer with sources → compare two pipelines)
- Embed the GIF at the top of README.md, right below the title
- Tools: [ScreenToGif](https://www.screentogif.com/) (Windows), or just screen-record and convert

### 3b. Architecture Diagram (Image)

The ASCII diagram in README is good but hard to scan. Create a proper diagram:

- Use [Excalidraw](https://excalidraw.com/) or [draw.io](https://app.diagrams.net/) (both free)
- Export as PNG/SVG → `docs/architecture.png`
- Replace or supplement the ASCII art in README
- Show the 5 pipelines, shared infra, data flow, and external services

### 3c. Results / Benchmarks Table

Add a "Results" section to README with actual numbers:

```
## Benchmark Results

| Pipeline | Faithfulness | Relevancy | Ctx Precision | p50 Latency | Cost/Query |
|----------|-------------|-----------|---------------|-------------|------------|
| Fastest RAG | 0.82 | 0.79 | 0.71 | 180ms | $0.0003 |
| Multimodal | 0.88 | 0.85 | 0.76 | 420ms | $0.0012 |
| CRAG | 0.91 | 0.87 | 0.83 | 650ms | $0.0008 |
| Self-RAG | 0.94 | 0.91 | 0.88 | 890ms | $0.0015 |
| Video RAG | 0.86 | 0.83 | 0.74 | 520ms | $0.0010 |
```

- Run your eval harness on a sample dataset and capture real numbers
- This is what separates a "built it" project from a "measured it" project
- Even small sample sizes (10-20 queries) are better than nothing

### ~~3d. LICENSE File~~ Done

### ~~3e. CONTRIBUTING.md~~ Done

### ~~3f. Badges in README~~ Done

### ~~3g. GitHub Repo Settings~~ Done

## 4. README Improvements

1. ~~**Replace `your-username`** in the clone URL~~ **Done**
2. ~~**Add a one-liner "Why this project?"**~~ **Done**
3. **Lead with the visual** (screenshot/GIF) before the architecture diagram — waiting on screenshots
4. ~~**Add a "Key Design Decisions" section**~~ **Done**
5. ~~**Trim "Future Work"** to 2-3 items~~ **Done**

## ~~5. Git History Cleanup~~ Done

Verified: no `.env` in history, phase story reads clearly, no action needed.

## 6. CV / Resume Bullet Points

### Project Line

> **RAG Research Platform** — Unified platform comparing 5 retrieval-augmented generation strategies with shared infrastructure, A/B testing, and automated evaluation

### Impact Bullets (pick 2-3)

- Architected a monorepo with 5 RAG pipeline strategies (Naive, Multimodal, CRAG, Self-RAG, Video+MCP) sharing a common FastAPI backend, pgvector store, and RAGAS evaluation harness
- Built LangGraph-based agentic pipelines (CRAG, Self-RAG) with multi-step decision graphs for relevance grading, hallucination detection, and web search fallback
- Implemented A/B comparison UI in Chainlit with side-by-side latency, cost, and RAGAS score metrics across all 5 strategies
- Designed video RAG pipeline with Whisper transcription, CLIP visual embeddings, Neo4j knowledge graph, and MCP tool server for Claude integration
- Achieved [X]% faithfulness improvement with Self-RAG over naive baseline while maintaining sub-1s p95 latency (fill in real numbers)

### Where to Link

- GitHub URL prominently in the "Projects" section
- If you have a personal site, create a short write-up page and link that too
- LinkedIn "Featured" section — add the repo link with the GIF/screenshot as the thumbnail

## 7. Presentation Tips

### For Interviews

- **Prepare a 2-minute walkthrough:** "Here's the problem (comparing RAG strategies is hard), here's the architecture (5 pipelines, shared infra), here's the interesting part (LangGraph decision graphs / evaluation results), here's what I'd do next"
- **Know the trade-offs:** Be ready to explain _why_ Self-RAG beats CRAG on faithfulness but is slower — this is where you demonstrate real understanding vs. just having built it
- **Have the demo running locally** so you can do a live walkthrough if asked

### For GitHub Visibility

- Pin the repo on your GitHub profile
- Write a short LinkedIn post announcing it: "Built a RAG Research Platform comparing 5 retrieval strategies..." with the GIF
- If you have time, write a blog post about one interesting design decision (e.g., "How I implemented Self-RAG with LangGraph" or "Benchmarking 5 RAG Strategies on the Same Corpus")

## 8. Remaining Priority Order

1. ~~**Add `.claude/`, `.mypy_cache/`, `.ruff_cache/`, `implementation-plan-rag.md` to .gitignore**~~ **Done**
2. **Capture screenshots** and embed one in README (20 min)
3. **Record a demo GIF** (30 min)
4. **Run benchmarks** and add results table (1-2 hrs depending on setup)
5. **Create proper architecture diagram** (30 min)
6. ~~**Set GitHub repo description + topics**~~ **Done**
7. **Write a blog post** (2-4 hrs — only if targeting specific roles)
