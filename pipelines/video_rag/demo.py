"""Demo script for Video RAG pipeline.

Usage:
    uv run python pipelines/video_rag/demo.py --video <path-or-url> --query "your question"
    uv run python pipelines/video_rag/demo.py --mcp  # start MCP server
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


async def run_demo(video_source: str, query: str) -> None:
    """Run the full video RAG demo: ingest → retrieve → answer."""
    from video_rag.clip_embedder import CLIPEmbedder
    from video_rag.scene_detector import SceneDetector
    from video_rag.timestamp_chunker import TimestampChunker
    from video_rag.video_indexer import VideoIndexer

    # 1. Transcribe video
    logger.info("Step 1: Transcribing video...")
    indexer = VideoIndexer(whisper_model="base")
    indexer.connect()
    video_path, segments = indexer.index_video(video_source)
    logger.info("Transcribed %d segments", len(segments))

    # 2. Detect scenes
    logger.info("Step 2: Detecting scenes...")
    detector = SceneDetector()
    keyframes = detector.detect_scenes(video_path)
    logger.info("Detected %d keyframes", len(keyframes))

    # 3. Compute CLIP embeddings
    logger.info("Step 3: Computing CLIP embeddings...")
    clip = CLIPEmbedder()
    clip.connect()
    kf_embeddings = clip.embed_images_batch([kf.frame for kf in keyframes])
    logger.info("Computed %d CLIP embeddings", len(kf_embeddings))

    # 4. Create timestamped chunks
    logger.info("Step 4: Creating timestamped chunks...")
    chunker = TimestampChunker()
    frame_mapping = chunker.assign_frame_embeddings(
        segments=segments,
        keyframe_timestamps=[kf.timestamp_s for kf in keyframes],
        keyframe_embeddings=kf_embeddings,
    )
    video_id = str(uuid.uuid4())[:8]
    chunks = chunker.chunks_from_segments(
        video_id=video_id,
        segments=segments,
        frame_embeddings=frame_mapping,
    )
    logger.info("Created %d chunks with timestamps", len(chunks))

    # 5. Print sample chunks
    print("\n--- Sample Chunks ---")
    for chunk in chunks[:5]:
        ts = f"[{chunk.metadata['start_ts']:.1f}s - {chunk.metadata['end_ts']:.1f}s]"
        has_frame = "frame_embedding" in chunk.metadata
        print(f"  {ts} {chunk.content[:80]}... (has_frame={has_frame})")

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Total keyframes: {len(keyframes)}")
    print(f"Video ID: {video_id}")
    print("\nTo query this video, integrate with the full pipeline (requires pgvector + Neo4j).")


def run_mcp_server() -> None:
    """Start the MCP server for Claude integration."""
    logger.info("Starting MCP server...")
    print("MCP server requires full pipeline setup (pgvector, Neo4j, CLIP).")
    print("Configure in your Claude MCP settings:")
    print(
        '  "video-rag": {"command": "uv", "args": ["run", "fastmcp", "run", '
        '"pipelines/video_rag/src/video_rag/mcp_server.py"]}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Video RAG Demo")
    parser.add_argument("--video", type=str, help="Video file path or YouTube URL")
    parser.add_argument("--query", type=str, default="What topics are covered?")
    parser.add_argument("--mcp", action="store_true", help="Start MCP server")
    args = parser.parse_args()

    if args.mcp:
        run_mcp_server()
    elif args.video:
        asyncio.run(run_demo(args.video, args.query))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
