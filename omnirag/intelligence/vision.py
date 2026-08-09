"""Visual understanding — the heart of OmniRAG's multimodality.

This module does **not** run OCR and throw the picture away. For every visual it
produces a faithful, searchable *semantic* description (what the chart shows,
what the diagram connects, what the flow does) while the original image stays in
the file store, referenced by :class:`~omnirag.core.models.VisualRef`. When such
a block is later retrieved, the answering model receives the image itself.

Cost control (visual models are the expensive part of the pipeline):

* blank/decorative images are filtered out before any API call;
* results are memoised by image content hash, so re-uploading or re-indexing the
  same figure costs nothing;
* a per-document budget caps the number of visual calls;
* one call returns both the description *and* the transcribed text, instead of
  paying for a separate OCR round-trip.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from omnirag.core.enums import BlockType, Language
from omnirag.core.exceptions import ProviderError
from omnirag.core.models import VisualRef
from omnirag.providers.llm.base import BaseLLMProvider, ImagePart, LLMMessage
from omnirag.providers.llm.context import llm_operation
from omnirag.utils.hashing import short_hash
from omnirag.utils.images import (
    is_probably_blank,
    is_probably_decorative,
    normalize_image,
)
from omnirag.utils.language import detect_language
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You analyse a visual element from a document so it can be found by search and reasoned about later.

Classify the visual as exactly one of:
"chart", "diagram", "table", "screenshot", "photo", "handwriting", "text", "decorative".

Then describe it FAITHFULLY. Never invent details that are not visible.

For a CHART, report: chart type; title; axis labels and units; legend entries; the
data series and their approximate values (read them off the axes — say "approximately");
the overall trend; notable comparisons, maxima, minima and anomalies; and the
conclusion a reader would draw.

For a DIAGRAM / FLOWCHART / TECHNICAL DRAWING, report: every component and its label;
how components connect; arrow directions; the sequence or process flow, step by step;
any hierarchy or grouping; and annotations, dimensions or callouts.

For a TABLE, report its title, column headers, row labels, and the values, preserving
the numeric relationships.

For a SCREENSHOT or PHOTO, describe factually what is shown and any text visible in it.

For HANDWRITING, transcribe what you can actually read and mark unclear words with [?].

Rules:
- Preserve every number, unit, date and proper name EXACTLY as shown. Do not round or recompute.
- Keep the original language of any text. If the visual contains Arabic, keep the Arabic.
- If something is unreadable or ambiguous, say so explicitly. Never guess.
- Write plain prose and short labelled lines. No markdown headings.

Return a single JSON object:
{
  "type": "<one of the categories above>",
  "title": "<title of the visual, or empty>",
  "description": "<the faithful description described above>",
  "text": "<verbatim text visible inside the visual, or empty>",
  "entities": ["<key labels, components, series or column names>"],
  "data_points": ["<'label: value' pairs you can actually read, or empty list>"],
  "confidence": <0.0-1.0>,
  "unreadable": <true|false>
}"""

USER_PROMPT = "Analyse this visual element from a document."

#: Maps the model's category to our block taxonomy.
_TYPE_TO_BLOCK = {
    "chart": BlockType.CHART,
    "diagram": BlockType.DIAGRAM,
    "table": BlockType.TABLE,
    "handwriting": BlockType.HANDWRITING,
    "screenshot": BlockType.IMAGE,
    "photo": BlockType.IMAGE,
    "text": BlockType.IMAGE,
    "decorative": BlockType.IMAGE,
}


@dataclass
class VisualAnalysis:
    """Structured result of analysing one visual."""

    block_type: BlockType = BlockType.IMAGE
    title: str = ""
    description: str = ""
    text: str = ""
    entities: List[str] = field(default_factory=list)
    data_points: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    unreadable: bool = False
    language: Language = Language.UNKNOWN
    decorative: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.searchable_text)

    @property
    def searchable_text(self) -> str:
        """Everything worth embedding, in retrieval-friendly order."""
        parts: List[str] = []
        if self.title:
            parts.append(self.title)
        if self.description:
            parts.append(self.description)
        if self.text:
            parts.append(self.text)
        if self.entities:
            parts.append("Labels: " + ", ".join(self.entities[:40]))
        if self.data_points:
            parts.append("Values: " + "; ".join(self.data_points[:40]))
        return "\n".join(p for p in parts if p.strip()).strip()

    @classmethod
    def failed(cls, error: str) -> "VisualAnalysis":
        return cls(error=error, unreadable=True)

    @classmethod
    def skipped_decorative(cls) -> "VisualAnalysis":
        return cls(decorative=True, error="skipped: decorative or blank image")


class VisionAnalyzer:
    """Analyses visuals through the configured multimodal LLM chain."""

    def __init__(
        self,
        llm: Optional[BaseLLMProvider],
        *,
        max_image_edge: int = 1400,
        jpeg_quality: int = 82,
        min_image_pixels: int = 110 * 110,
        max_output_tokens: int = 1200,
        enabled: bool = True,
    ):
        self.llm = llm
        self.max_image_edge = max_image_edge
        self.jpeg_quality = jpeg_quality
        self.min_image_pixels = min_image_pixels
        self.max_output_tokens = max_output_tokens
        self.enabled = enabled
        self._cache: Dict[str, VisualAnalysis] = {}
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        return bool(self.enabled and self.llm is not None and self.llm.supports_images())

    def analyze(
        self,
        image: bytes,
        *,
        context: str = "",
        expect: Optional[BlockType] = None,
        skip_decorative_check: bool = False,
        document_id: str = "",
        page_number: Optional[int] = None,
    ) -> VisualAnalysis:
        """Describe one visual. Never raises — failures come back as ``error``."""
        if not image:
            return VisualAnalysis.failed("empty image")

        if not skip_decorative_check and is_probably_decorative(
            image, min_pixels=self.min_image_pixels
        ):
            return VisualAnalysis.skipped_decorative()
        if skip_decorative_check and is_probably_blank(image):
            return VisualAnalysis.skipped_decorative()

        if not self.available:
            return VisualAnalysis.failed(
                "Visual understanding is unavailable: no image-capable model is configured."
            )

        provider_identity = "unconfigured"
        if self.llm is not None:
            description = self.llm.describe()
            chain = description.get("chain") or [description]
            provider_identity = "|".join(
                f"{item.get('provider', '')}:{item.get('model', '')}"
                for item in chain
            )
        key = short_hash(
            "|".join(
                [
                    "vision-cache-v2",
                    document_id,
                    str(page_number or ""),
                    short_hash(image, 32),
                    provider_identity,
                    expect.value if expect else "auto",
                ]
            ),
            32,
        )
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        prepared, media_type = normalize_image(
            image, max_edge=self.max_image_edge, jpeg_quality=self.jpeg_quality
        )
        prompt = USER_PROMPT
        if context:
            prompt = f"{USER_PROMPT}\n\nSurrounding document context (for naming only, do not copy):\n{context[:600]}"
        if expect is not None:
            prompt += f"\n\nThis element was extracted as a {expect.value}."

        try:
            self.calls += 1
            with llm_operation("vision_analysis"):
                response = self.llm.complete(  # type: ignore[union-attr]
                    [
                        LLMMessage(
                            text=prompt,
                            images=[ImagePart(data=prepared, media_type=media_type)],
                        )
                    ],
                    system=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=self.max_output_tokens,
                    json_mode=True,
                )
        except ProviderError as exc:
            logger.warning("Visual analysis failed: %s", exc)
            return VisualAnalysis.failed(exc.user_message)
        except Exception as exc:  # noqa: BLE001 - one bad figure must not stop a document
            logger.exception("Unexpected visual-analysis failure")
            return VisualAnalysis.failed(f"visual analysis failed: {type(exc).__name__}")

        analysis = _parse(response.text, expect=expect)
        if analysis.ok:
            with self._lock:
                self._cache[key] = analysis
        return analysis

    def stats(self) -> Dict[str, int]:
        return {"calls": self.calls, "cache_hits": self.cache_hits, "cached": len(self._cache)}


# --------------------------------------------------------------------------- #
def _parse(raw: str, *, expect: Optional[BlockType] = None) -> VisualAnalysis:
    data = _loads(raw)
    if data is None:
        text = (raw or "").strip()
        if not text:
            return VisualAnalysis.failed("empty visual-analysis response")
        # Prose reply: still useful for retrieval, just unstructured.
        return VisualAnalysis(
            block_type=expect or BlockType.IMAGE,
            description=text,
            language=detect_language(text),
        )

    category = str(data.get("type", "")).strip().lower()
    block_type = _TYPE_TO_BLOCK.get(category, expect or BlockType.IMAGE)

    description = str(data.get("description", "")).strip()
    text = str(data.get("text", "")).strip()
    analysis = VisualAnalysis(
        block_type=block_type,
        title=str(data.get("title", "")).strip(),
        description=description,
        text=text,
        entities=_as_str_list(data.get("entities")),
        data_points=_as_str_list(data.get("data_points")),
        confidence=_as_float(data.get("confidence")),
        unreadable=bool(data.get("unreadable", False)),
        decorative=category == "decorative",
        language=detect_language(f"{description}\n{text}"),
    )
    if not analysis.searchable_text:
        analysis.error = "visual analysis produced no usable description"
    return analysis


def _loads(raw: str) -> Optional[dict]:
    payload = (raw or "").strip()
    if not payload:
        return None
    if payload.startswith("```"):
        payload = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", payload).strip()
    try:
        data = json.loads(payload)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def _as_str_list(value: object, limit: int = 60) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value[:limit]:
        text = str(item).strip()
        if text:
            out.append(text[:200])
    return out


def _as_float(value: object) -> Optional[float]:
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def build_vision_analyzer(settings=None) -> VisionAnalyzer:
    """Construct the analyser from configuration (LLM resolved lazily)."""
    from omnirag.config.settings import get_settings

    resolved = settings or get_settings()
    cfg = resolved.vision

    llm: Optional[BaseLLMProvider] = None
    if cfg.enabled and resolved.llm.is_configured:
        try:
            from omnirag.providers.llm.factory import get_llm_provider

            llm = get_llm_provider(resolved)
        except Exception as exc:
            logger.warning("Vision analyser has no LLM available: %s", exc)

    return VisionAnalyzer(
        llm,
        max_image_edge=cfg.max_image_edge,
        jpeg_quality=cfg.jpeg_quality,
        min_image_pixels=cfg.min_image_pixels,
        enabled=cfg.enabled,
    )


__all__ = ["VisionAnalyzer", "VisualAnalysis", "VisualRef", "build_vision_analyzer"]
