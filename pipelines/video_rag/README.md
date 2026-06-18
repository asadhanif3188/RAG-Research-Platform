# Video RAG — MCP-powered RAG over Videos

Hybrid text+visual retrieval pipeline for video content with timestamp-level segments, Neo4j knowledge graph for multi-hop queries, and FastMCP tool server for Claude integration.

## Architecture

```
YouTube URL / Video File
        │
        ▼
┌─────────────────┐     ┌─────────────────┐
│  VideoIndexer   │     │  SceneDetector   │
│  (Whisper ASR)  │     │  (OpenCV)        │
│                 │     │                  │
│  Transcripts    │     │  Keyframes       │
│  + timestamps   │     │  + timestamps    │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ OpenAI Embed    │     │  CLIPEmbedder   │
│ (text-embed-3)  │     │  (ViT-B-32)     │
│                 │     │                  │
│ Text embeddings │     │ Visual embeddings│
└────────┬────────┘     └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌────────────────┐
            │  pgvector      │
            │  (hybrid index)│
            └────────┬───────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Segment  │  │ Knowledge│  │   MCP    │
│ Retriever│  │  Graph   │  │  Server  │
│ (fused)  │  │  (Neo4j) │  │ (FastMCP)│
└──────────┘  └──────────┘  └──────────┘
```

## Components

| Module | Description |
|--------|-------------|
| `video_indexer.py` | Download (yt-dlp) + transcribe (Whisper) + segment by sentences |
| `scene_detector.py` | Keyframe extraction at scene boundaries via histogram difference |
| `clip_embedder.py` | CLIP (ViT-B-32) visual embeddings for keyframes and text queries |
| `timestamp_chunker.py` | Convert segments to DocumentChunks with start_ts/end_ts metadata |
| `segment_retriever.py` | Dual retrieval (text + visual) with fused ranking (0.6/0.4) |
| `knowledge_graph.py` | Neo4j topic graph for multi-hop queries |
| `mcp_server.py` | FastMCP server exposing `search_video`, `get_segment`, `list_videos`, `get_transcript` |
| `pipeline.py` | API-compatible wrapper with Claude answer generation |

## Quick Start

```bash
# Install
uv sync --package video-rag

# Demo: ingest a video
uv run python pipelines/video_rag/demo.py --video "https://youtube.com/watch?v=..."

# Run tests
uv run pytest pipelines/video_rag/tests/unit/ -v

# Start MCP server (for Claude integration)
uv run fastmcp run pipelines/video_rag/src/video_rag/mcp_server.py
```

## MCP Integration

Add to your Claude MCP config:

```json
{
  "mcpServers": {
    "video-rag": {
      "command": "uv",
      "args": ["run", "fastmcp", "run", "pipelines/video_rag/src/video_rag/mcp_server.py"]
    }
  }
}
```

Available tools:
- `search_video(query, top_k, video_id)` — hybrid text+visual search
- `get_segment(segment_id, video_id)` — get segment details
- `list_videos()` — list all indexed videos
- `get_transcript(video_id)` — get full transcript with timestamps
