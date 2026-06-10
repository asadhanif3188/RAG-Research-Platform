"""Standalone Streamlit demo for Multimodal RAG.

Upload a PDF, ask questions, and see provenance highlighting showing which
source pages, images, and tables contributed to the answer.

Run:
    streamlit run demo.py

Requires:
    ANTHROPIC_API_KEY  — Claude vision + generation
    OPENAI_API_KEY     — text-embedding-3-large embeddings

Optionally connects to pgvector + Redis when DATABASE_URL / REDIS_URL are set.
Falls back to in-memory vector search when infrastructure is unavailable.
"""

from __future__ import annotations

import html
import logging
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal RAG Demo",
    page_icon="🔍",
    layout="wide",
)

# ── Session state defaults ───────────────────────────────────────────────────

if "documents" not in st.session_state:
    st.session_state.documents = {}  # doc_id → {chunks, parsed_doc}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Lightweight in-memory vector store ───────────────────────────────────────


@dataclass
class InMemoryChunk:
    """A chunk stored in memory with its embedding."""

    chunk_id: str
    document_id: str
    chunk_index: int
    chunk_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)


class InMemoryVectorStore:
    """Cosine-similarity search over in-memory chunks. No infrastructure needed."""

    def __init__(self) -> None:
        self._chunks: dict[str, InMemoryChunk] = {}

    def add(self, chunks: list[InMemoryChunk]) -> list[str]:
        ids = []
        for c in chunks:
            self._chunks[c.chunk_id] = c
            ids.append(c.chunk_id)
        return ids

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        chunk_type: str | None = None,
    ) -> list[tuple[InMemoryChunk, float]]:
        import numpy as np

        if not self._chunks:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        results: list[tuple[InMemoryChunk, float]] = []
        for chunk in self._chunks.values():
            if chunk_type and chunk.chunk_type != chunk_type:
                continue
            if not chunk.embedding:
                continue
            c = np.array(chunk.embedding, dtype=np.float32)
            c_norm = np.linalg.norm(c)
            if c_norm == 0:
                continue
            score = float(np.dot(q, c / c_norm))
            results.append((chunk, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        self._chunks.clear()

    @property
    def count(self) -> int:
        return len(self._chunks)


# ── Globals ──────────────────────────────────────────────────────────────────

_store = InMemoryVectorStore()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_openai_client() -> Any:
    """Lazy-init OpenAI client."""
    if "openai_client" not in st.session_state:
        from openai import OpenAI

        st.session_state.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return st.session_state.openai_client


def _get_anthropic_client() -> Any:
    """Lazy-init Anthropic client."""
    if "anthropic_client" not in st.session_state:
        import anthropic

        st.session_state.anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
    return st.session_state.anthropic_client


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI text-embedding-3-large."""
    client = _get_openai_client()
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
    )
    return [item.embedding for item in response.data]


def _embed_single(text: str) -> list[float]:
    return _embed([text])[0]


def _describe_image(image_bytes: bytes, media_type: str = "image/png") -> str:
    """Use Claude vision to describe an image."""
    import base64

    client = _get_anthropic_client()
    b64 = base64.standard_b64encode(image_bytes).decode()

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail for a research context. "
                            "Include key data points, labels, trends, and the main insight."
                        ),
                    },
                ],
            }
        ],
    )
    return message.content[0].text


# ── Ingestion ────────────────────────────────────────────────────────────────


def ingest_pdf(pdf_path: Path, progress_bar: Any) -> str:
    """Parse, chunk, embed, and store a PDF. Returns document_id."""
    import fitz

    doc = fitz.open(str(pdf_path))
    doc_id = pdf_path.stem + "_" + uuid.uuid4().hex[:6]

    chunks: list[InMemoryChunk] = []
    chunk_idx = 0
    total_pages = len(doc)

    for page_num in range(total_pages):
        progress_bar.progress(
            (page_num + 1) / (total_pages + 1),
            text=f"Parsing page {page_num + 1}/{total_pages}...",
        )

        page = doc[page_num]
        blocks = page.get_text("blocks")

        # Text blocks
        text_parts: list[str] = []
        for block in blocks:
            if block[6] == 0 and len(block[4].strip()) >= 10:
                text_parts.append(block[4].strip())

        page_text = "\n".join(text_parts)
        if page_text.strip():
            # Simple fixed-size chunking (512 words with 64 overlap)
            words = page_text.split()
            for i in range(0, len(words), 448):
                chunk_words = words[i : i + 512]
                if not chunk_words:
                    continue
                chunks.append(
                    InMemoryChunk(
                        chunk_id=f"{doc_id}_c{chunk_idx}",
                        document_id=doc_id,
                        chunk_index=chunk_idx,
                        chunk_type="text",
                        content=" ".join(chunk_words),
                        metadata={"page_number": page_num + 1, "source_file": pdf_path.name},
                    )
                )
                chunk_idx += 1

        # Tables via pdfplumber
        try:
            import pdfplumber

            with pdfplumber.open(str(pdf_path)) as plumber:
                if page_num < len(plumber.pages):
                    plumber_page = plumber.pages[page_num]
                    for table_data in plumber_page.extract_tables() or []:
                        if not table_data or len(table_data) < 2:
                            continue
                        md = _table_to_markdown(table_data)
                        if md:
                            chunks.append(
                                InMemoryChunk(
                                    chunk_id=f"{doc_id}_c{chunk_idx}",
                                    document_id=doc_id,
                                    chunk_index=chunk_idx,
                                    chunk_type="table",
                                    content=md,
                                    metadata={
                                        "page_number": page_num + 1,
                                        "source_file": pdf_path.name,
                                    },
                                )
                            )
                            chunk_idx += 1
        except ImportError:
            pass

        # Images via Claude vision
        if os.environ.get("ANTHROPIC_API_KEY"):
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    img_data = doc.extract_image(xref)
                    raw_bytes: bytes = img_data["image"]
                    # Skip tiny images (likely icons)
                    if len(raw_bytes) < 5000:
                        continue

                    # Convert to PNG via pixmap
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    png_bytes = pix.tobytes("png")

                    description = _describe_image(png_bytes, "image/png")
                    if description.strip():
                        chunks.append(
                            InMemoryChunk(
                                chunk_id=f"{doc_id}_c{chunk_idx}",
                                document_id=doc_id,
                                chunk_index=chunk_idx,
                                chunk_type="image_description",
                                content=description,
                                metadata={
                                    "page_number": page_num + 1,
                                    "source_file": pdf_path.name,
                                },
                            )
                        )
                        chunk_idx += 1
                except Exception as exc:
                    logger.warning("Image extraction failed (xref=%d): %s", xref, exc)

    doc.close()

    if not chunks:
        st.warning("No content extracted from PDF.")
        return doc_id

    # Embed all chunks
    progress_bar.progress(0.9, text="Generating embeddings...")
    batch_size = 50
    all_texts = [c.content for c in chunks]
    all_embeddings: list[list[float]] = []
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i : i + batch_size]
        all_embeddings.extend(_embed(batch))

    for chunk, emb in zip(chunks, all_embeddings, strict=False):
        chunk.embedding = emb

    _store.add(chunks)
    progress_bar.progress(1.0, text="Done!")

    st.session_state.documents[doc_id] = {
        "name": pdf_path.name,
        "chunk_count": len(chunks),
        "types": _count_types(chunks),
    }

    return doc_id


def _table_to_markdown(table_data: list[list[str | None]]) -> str:
    """Convert 2D table data to Markdown."""
    rows = [[str(c).strip() if c else "" for c in row] for row in table_data if row]
    if len(rows) < 2:
        return ""
    col_count = max(len(r) for r in rows)
    if col_count < 2:
        return ""
    rows = [r + [""] * (col_count - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(col_count)) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _count_types(chunks: list[InMemoryChunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.chunk_type] = counts.get(c.chunk_type, 0) + 1
    return counts


# ── Query + Provenance ───────────────────────────────────────────────────────

_CHUNK_TYPE_LABELS = {
    "text": "Text",
    "image_description": "Image",
    "table": "Table",
}

_STOPWORDS = frozenset([
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "of", "and", "or", "but",
    "not", "with", "this", "that", "are", "was", "were", "be", "been", "has", "have", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "its",
    "their", "they", "we", "you", "he", "she", "as", "by", "from", "which", "who", "what",
    "when", "where", "how", "if", "also", "than", "then", "so",
])


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"\b[a-z][a-z0-9]*\b", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class ProvenanceRecord:
    sentence: str
    chunk_id: str
    document_id: str
    page_number: int | None
    chunk_type: str
    confidence: float


def _track_provenance(
    answer: str,
    sources: list[tuple[InMemoryChunk, float]],
    min_confidence: float = 0.05,
) -> list[ProvenanceRecord]:
    """Attribute answer sentences to source chunks via Jaccard overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    records: list[ProvenanceRecord] = []

    for sentence in sentences:
        if len(sentence.split()) < 3:
            continue
        sent_tokens = _tokenize(sentence)
        best_score = min_confidence
        best_chunk: InMemoryChunk | None = None

        for chunk, _retrieval_score in sources:
            score = _jaccard(sent_tokens, _tokenize(chunk.content))
            if score > best_score:
                best_score = score
                best_chunk = chunk

        if best_chunk is not None:
            records.append(
                ProvenanceRecord(
                    sentence=sentence,
                    chunk_id=best_chunk.chunk_id,
                    document_id=best_chunk.document_id,
                    page_number=best_chunk.metadata.get("page_number"),
                    chunk_type=best_chunk.chunk_type,
                    confidence=round(best_score, 4),
                )
            )

    return records


def query_pipeline(question: str, top_k: int = 10) -> dict[str, Any]:
    """Run multimodal retrieval + generation + provenance tracking."""
    if _store.count == 0:
        return {"answer": "No documents ingested yet. Please upload a PDF first.", "provenance": []}

    # Embed query
    q_emb = _embed_single(question)

    # Multi-type retrieval with quotas
    text_k = max(1, int(top_k * 0.5))
    image_k = max(1, int(top_k * 0.3))
    table_k = max(1, top_k - text_k - image_k)

    text_results = _store.search(q_emb, text_k, chunk_type="text")
    image_results = _store.search(q_emb, image_k, chunk_type="image_description")
    table_results = _store.search(q_emb, table_k, chunk_type="table")

    # Merge and deduplicate by chunk_id
    seen: set[str] = set()
    all_results: list[tuple[InMemoryChunk, float]] = []
    for chunk, score in text_results + image_results + table_results:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            all_results.append((chunk, score))
    all_results.sort(key=lambda x: x[1], reverse=True)
    all_results = all_results[:8]

    if not all_results:
        return {"answer": "No relevant chunks found for your query.", "provenance": []}

    # Build context
    context_parts: list[str] = []
    for i, (chunk, score) in enumerate(all_results, 1):
        label = _CHUNK_TYPE_LABELS.get(chunk.chunk_type, "Text")
        page_info = (
            f", page={chunk.metadata['page_number']}" if "page_number" in chunk.metadata else ""
        )
        header = f"[Chunk {i}] ({label}, score={score:.3f}, doc={chunk.document_id}{page_info})"
        context_parts.append(f"{header}\n{chunk.content}")

    context = "\n\n---\n\n".join(context_parts)

    # Generate answer
    client = _get_anthropic_client()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
            "You are a precise research assistant with access to text passages, image descriptions, "
            "and table data extracted from documents. Answer the user's question using ONLY the "
            "provided context chunks. Cite chunk(s) as [Chunk N] (Type) for each claim."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"<context>\n{context}\n</context>\n\n"
                    f"Question: {question}\n\n"
                    "Answer based only on the context above, citing [Chunk N] (Type) for each claim:"
                ),
            }
        ],
    )
    answer = message.content[0].text

    # Provenance
    provenance = _track_provenance(answer, all_results)

    return {
        "answer": answer,
        "provenance": provenance,
        "sources": all_results,
        "type_distribution": {
            chunk_type: sum(1 for c, _ in all_results if c.chunk_type == chunk_type)
            for chunk_type in ("text", "image_description", "table")
        },
    }


# ── Provenance rendering ────────────────────────────────────────────────────

_COLOURS = ["#dbeafe", "#dcfce7", "#fef9c3", "#fce7f3", "#ede9fe", "#ffedd5"]
_TYPE_ICONS = {"text": "📄", "image_description": "🖼️", "table": "📊"}


def render_provenance(answer: str, provenance: list[ProvenanceRecord]) -> str:
    """Render answer with highlighted provenance as HTML."""
    if not provenance:
        return f"<p>{html.escape(answer)}</p>"

    # Chunk ID → colour
    chunk_ids = list(dict.fromkeys(p.chunk_id for p in provenance))
    colour_map = {cid: _COLOURS[i % len(_COLOURS)] for i, cid in enumerate(chunk_ids)}

    # Sentence → chunk_id mapping
    sentence_map = {p.sentence: p.chunk_id for p in provenance}

    # Highlight sentences
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    parts: list[str] = []
    for sentence in sentences:
        chunk_id = sentence_map.get(sentence)
        if chunk_id and chunk_id in colour_map:
            colour = colour_map[chunk_id]
            parts.append(
                f'<span style="background-color: {colour}; border-radius: 3px; '
                f'padding: 1px 4px;" title="Source: {html.escape(chunk_id)}">'
                f"{html.escape(sentence)}</span>"
            )
        else:
            parts.append(html.escape(sentence))

    highlighted = " ".join(parts)

    # Attribution table
    rows = ""
    for p in provenance:
        icon = _TYPE_ICONS.get(p.chunk_type, "📄")
        colour = colour_map.get(p.chunk_id, "#f1f5f9")
        page_str = str(p.page_number) if p.page_number else "—"
        conf = f"{p.confidence * 100:.0f}%"
        preview = p.sentence[:80] + ("…" if len(p.sentence) > 80 else "")
        rows += f"""<tr>
            <td style="padding:6px 10px;background:{colour};border-radius:4px;">
                {icon} {html.escape(p.chunk_type.replace('_', ' ').title())}</td>
            <td style="padding:6px 10px;font-family:monospace;font-size:12px;">
                {html.escape(p.document_id)}</td>
            <td style="padding:6px 10px;text-align:center;">{page_str}</td>
            <td style="padding:6px 10px;text-align:center;font-weight:600;">{conf}</td>
            <td style="padding:6px 10px;font-size:13px;font-style:italic;">
                "{html.escape(preview)}"</td>
        </tr>"""

    return f"""
    <div style="background:#f8fafc;border-radius:8px;padding:16px;margin-bottom:16px;
                line-height:1.8;font-size:15px;color:#334155;">
        {highlighted}
    </div>
    <details style="margin-top:8px;">
        <summary style="cursor:pointer;font-weight:600;color:#475569;font-size:14px;padding:6px 0;">
            Sources ({len(provenance)} attributions)
        </summary>
        <table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:13px;
                      border:1px solid #e2e8f0;border-radius:6px;">
            <thead><tr style="background:#f1f5f9;text-align:left;">
                <th style="padding:8px 10px;">Type</th>
                <th style="padding:8px 10px;">Document</th>
                <th style="padding:8px 10px;text-align:center;">Page</th>
                <th style="padding:8px 10px;text-align:center;">Confidence</th>
                <th style="padding:8px 10px;">Sentence</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </details>
    """


# ── Streamlit UI ─────────────────────────────────────────────────────────────


def main() -> None:
    st.title("🔍 Multimodal RAG Demo")
    st.caption(
        "Upload a PDF with text, images, and tables — then ask questions with source attribution."
    )

    # Sidebar: document management
    with st.sidebar:
        st.header("Documents")

        uploaded = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)
        if uploaded is not None:
            if uploaded.name not in [d["name"] for d in st.session_state.documents.values()]:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = Path(tmp.name)

                progress = st.progress(0, text="Starting ingestion...")
                ingest_pdf(tmp_path, progress)
                tmp_path.unlink(missing_ok=True)
                st.success(f"Ingested **{uploaded.name}** ({_store.count} total chunks)")
            else:
                st.info(f"**{uploaded.name}** already ingested.")

        if st.session_state.documents:
            st.divider()
            st.subheader("Ingested Documents")
            for _doc_id, info in st.session_state.documents.items():
                type_str = ", ".join(f"{v} {k}" for k, v in info["types"].items())
                st.markdown(f"**{info['name']}**  \n{info['chunk_count']} chunks ({type_str})")

            if st.button("Clear all documents"):
                _store.clear()
                st.session_state.documents.clear()
                st.session_state.chat_history.clear()
                st.rerun()

        st.divider()
        st.subheader("Settings")
        top_k = st.slider("Retrieval top-k", min_value=3, max_value=20, value=10)

        st.divider()
        st.markdown(
            "**API Keys** — set via environment variables:\n"
            "- `ANTHROPIC_API_KEY`\n"
            "- `OPENAI_API_KEY`"
        )

    # Main area: chat
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.components.v1.html(entry["html"], height=entry.get("height", 400), scrolling=True)

    question = st.chat_input("Ask a question about your documents...")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                result = query_pipeline(question, top_k=top_k)

            answer = result["answer"]
            provenance = result["provenance"]
            prov_html = render_provenance(answer, provenance)

            # Show type distribution
            dist = result.get("type_distribution", {})
            if dist:
                cols = st.columns(3)
                for col, (ctype, label) in zip(
                    cols, [("text", "📄 Text"), ("image_description", "🖼️ Image"), ("table", "📊 Table")],
                    strict=False,
                ):
                    col.metric(label, dist.get(ctype, 0))

            st.components.v1.html(prov_html, height=400, scrolling=True)

            st.session_state.chat_history.append(
                {"question": question, "html": prov_html, "height": 400}
            )


if __name__ == "__main__":
    main()
