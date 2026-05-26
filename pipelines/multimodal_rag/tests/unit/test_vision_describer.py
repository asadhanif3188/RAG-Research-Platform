"""Unit tests for VisionDescriber — mock the Anthropic client."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multimodal_rag.vision_describer import VisionDescriber


def _make_mock_message(text: str, input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return msg


@pytest.fixture
def describer_with_mock() -> tuple[VisionDescriber, AsyncMock]:
    describer = VisionDescriber(api_key="sk-test")
    mock_client = AsyncMock()
    describer._client = mock_client
    return describer, mock_client


class TestVisionDescriberDescribeImage:
    @pytest.mark.asyncio
    async def test_describe_image_returns_description(self, describer_with_mock):
        describer, mock_client = describer_with_mock
        mock_client.messages.create = AsyncMock(
            return_value=_make_mock_message("A bar chart showing revenue growth.")
        )

        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8  # minimal fake PNG header
        result = await describer.describe_image(fake_png)

        assert result == "A bar chart showing revenue growth."

    @pytest.mark.asyncio
    async def test_describe_image_passes_base64(self, describer_with_mock):
        describer, mock_client = describer_with_mock
        mock_client.messages.create = AsyncMock(
            return_value=_make_mock_message("Description")
        )

        image_bytes = b"fake image data"
        await describer.describe_image(image_bytes)

        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        image_content = messages[0]["content"][0]

        assert image_content["type"] == "image"
        expected_b64 = base64.standard_b64encode(image_bytes).decode()
        assert image_content["source"]["data"] == expected_b64

    @pytest.mark.asyncio
    async def test_describe_image_with_context(self, describer_with_mock):
        describer, mock_client = describer_with_mock
        mock_client.messages.create = AsyncMock(
            return_value=_make_mock_message("Annotated figure")
        )

        await describer.describe_image(b"img", context="Annual report 2024")

        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        text_content = messages[0]["content"][1]
        assert "Annual report 2024" in text_content["text"]

    @pytest.mark.asyncio
    async def test_describe_image_tracks_tokens(self, describer_with_mock):
        describer, mock_client = describer_with_mock
        mock_client.messages.create = AsyncMock(
            return_value=_make_mock_message("Result", input_tokens=200, output_tokens=80)
        )

        await describer.describe_image(b"img")
        assert describer.total_tokens_used == 280

    @pytest.mark.asyncio
    async def test_describe_image_accumulates_tokens(self, describer_with_mock):
        describer, mock_client = describer_with_mock
        mock_client.messages.create = AsyncMock(
            return_value=_make_mock_message("R", input_tokens=50, output_tokens=10)
        )

        await describer.describe_image(b"img1")
        await describer.describe_image(b"img2")
        assert describer.total_tokens_used == 120


class TestVisionDescriberExtractTable:
    @pytest.mark.asyncio
    async def test_extract_table_from_text(self, describer_with_mock):
        describer, mock_client = describer_with_mock
        mock_client.messages.create = AsyncMock(
            return_value=_make_mock_message(
                "Sales data table.\n| Q | Revenue |\n|---|---|\n| Q1 | 100 |"
            )
        )

        result = await describer.extract_table(table_text="Q | Revenue\nQ1 | 100")
        assert "Revenue" in result
        assert "|" in result

    @pytest.mark.asyncio
    async def test_extract_table_from_image(self, describer_with_mock):
        describer, mock_client = describer_with_mock
        mock_client.messages.create = AsyncMock(
            return_value=_make_mock_message("Table data.\n| A | B |\n|---|---|\n| 1 | 2 |")
        )

        result = await describer.extract_table(table_image_bytes=b"fake_table_img")
        assert result

        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        image_content = messages[0]["content"][0]
        assert image_content["type"] == "image"

    @pytest.mark.asyncio
    async def test_extract_table_raises_without_input(self, describer_with_mock):
        describer, _ = describer_with_mock
        with pytest.raises(ValueError, match="Must provide"):
            await describer.extract_table()

    @pytest.mark.asyncio
    async def test_extract_table_tracks_tokens(self, describer_with_mock):
        describer, mock_client = describer_with_mock
        mock_client.messages.create = AsyncMock(
            return_value=_make_mock_message("| A |\n|---|\n| 1 |", input_tokens=300, output_tokens=100)
        )

        await describer.extract_table(table_text="A\n1")
        assert describer.total_tokens_used == 400


class TestVisionDescriberLazyConnect:
    @pytest.mark.asyncio
    async def test_connect_on_first_call(self):
        describer = VisionDescriber(api_key="sk-test")
        assert describer._client is None

        mock_msg = _make_mock_message("desc")
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            instance = AsyncMock()
            instance.messages.create = AsyncMock(return_value=mock_msg)
            mock_cls.return_value = instance

            await describer.describe_image(b"img")

        mock_cls.assert_called_once_with(api_key="sk-test")
