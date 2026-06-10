"""VisionDescriber — sends images and tables to Claude vision for text descriptions.

Converts visual content (PNG/JPEG images, raw table text) into searchable text descriptions
that can be embedded and stored as IMAGE_DESCRIPTION / TABLE chunks.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)

_IMAGE_DESCRIBE_PROMPT = """\
Describe this image in detail for a research/academic context. Include:
- What type of visualization/figure this is (chart, diagram, photograph, etc.)
- Key data points, labels, axis titles, or values shown
- The main insight or finding this image conveys
- Any relevant statistics, measurements, or trends visible
Be specific and factual. Your description will be used for text-based semantic retrieval."""

_TABLE_TEXT_PROMPT = """\
You are given raw table text extracted from a PDF. Convert it to clean Markdown format:
- Use | for column separators
- Include a header separator row (|---|---|...)
- Preserve all values exactly as shown
- Add a brief one-sentence summary at the top describing what the table contains
Return the summary line followed by the Markdown table."""

_TABLE_IMAGE_PROMPT = """\
Extract the data from this table image as clean Markdown format:
- Use | for column separators
- Include a header separator row (|---|---|...)
- Preserve all values exactly as shown
- Add a brief one-sentence summary at the top describing what the table contains
Return the summary line followed by the Markdown table."""


class VisionDescriber:
    """Describe images and tables using Claude's vision capability.

    Args:
        api_key: Anthropic API key.
        model: Claude model with vision capability.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client: Any = None
        self._total_tokens: int = 0

    def connect(self) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        logger.info("VisionDescriber ready (model=%s)", self._model)

    async def describe_image(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
        context: str = "",
    ) -> str:
        """Send image bytes to Claude vision and return a text description.

        Args:
            image_bytes: Raw image data (PNG, JPEG, etc.).
            media_type: MIME type of the image.
            context: Optional document context to improve description quality.

        Returns:
            Text description suitable for embedding and retrieval.
        """
        if self._client is None:
            self.connect()

        image_b64 = base64.standard_b64encode(image_bytes).decode()
        prompt = (
            f"Document context: {context}\n\n{_IMAGE_DESCRIBE_PROMPT}"
            if context
            else _IMAGE_DESCRIBE_PROMPT
        )

        message = await self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        description = message.content[0].text
        self._total_tokens += message.usage.input_tokens + message.usage.output_tokens
        logger.debug("Image described (%d input tokens)", message.usage.input_tokens)
        return description

    async def extract_table(
        self,
        table_image_bytes: bytes | None = None,
        table_text: str | None = None,
        media_type: str = "image/png",
    ) -> str:
        """Extract table content as Markdown from either an image or raw text.

        Args:
            table_image_bytes: Raw image bytes containing the table (optional).
            table_text: Raw text extracted from the table region (optional).
            media_type: MIME type if image is provided.

        Returns:
            Markdown-formatted table with a summary header line.

        Raises:
            ValueError: If neither argument is provided.
        """
        if self._client is None:
            self.connect()

        if table_image_bytes is not None:
            image_b64 = base64.standard_b64encode(table_image_bytes).decode()
            content: list[dict[str, Any]] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": _TABLE_IMAGE_PROMPT},
            ]
        elif table_text is not None:
            content = [
                {
                    "type": "text",
                    "text": f"Raw table text:\n\n{table_text}\n\n{_TABLE_TEXT_PROMPT}",
                }
            ]
        else:
            raise ValueError("Must provide either table_image_bytes or table_text")

        message = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )

        result = message.content[0].text
        self._total_tokens += message.usage.input_tokens + message.usage.output_tokens
        return result

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens
