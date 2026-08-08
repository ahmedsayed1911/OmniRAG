"""Vision-model OCR — the default backend on Streamlit Community Cloud.

Why this is the default: local OCR engines need system binaries and language
packs (``apt`` packages) that Streamlit Cloud makes awkward, whereas a
multimodal LLM needs nothing but an API key, reads Arabic and English natively,
handles mixed-script pages, and is currently the only practical way to attempt
handwriting.

Honesty rules baked into the prompt:
* transcribe only what is visible, never "complete" a partial word;
* mark unreadable regions with ``[?]`` rather than guessing;
* report a self-assessed confidence, which is stored with the block.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from omnirag.core.enums import Role
from omnirag.core.exceptions import ProviderError
from omnirag.providers.llm.base import BaseLLMProvider, ImagePart, LLMMessage
from omnirag.providers.ocr.base import BaseOCRProvider, OCRResult
from omnirag.utils.images import ensure_min_size, normalize_image
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a precise OCR engine for Arabic and English documents.

Transcribe ALL visible text from the image, exactly as written.

Rules:
- Preserve the original language and script. Never translate.
- Preserve reading order and line breaks. For Arabic, transcribe right-to-left text correctly.
- Preserve numbers, dates and units exactly as printed. Never normalise or recompute them.
- If a word or region is illegible, write [?] instead of guessing.
- Do not describe the image, do not add commentary, do not summarise.
- If there is no text at all, return an empty string for "text".

Return a single JSON object:
{"text": "<verbatim transcription>", "confidence": <0.0-1.0>, "has_handwriting": <true|false>, "notes": "<short note or empty>"}

"confidence" is your honest self-assessment of transcription accuracy."""

HANDWRITING_HINT = """This image is expected to contain handwriting.
Handwriting recognition is unreliable — be conservative:
- transcribe only what you can actually read;
- use [?] for uncertain words;
- lower "confidence" accordingly (handwriting rarely deserves > 0.8)."""

MAX_IMAGE_EDGE = 1600


class VisionOCRProvider(BaseOCRProvider):
    """OCR backed by whichever multimodal LLM chain is configured."""

    name = "vision"
    supports_handwriting = True
    supports_arabic = True

    def __init__(
        self,
        llm: BaseLLMProvider,
        *,
        languages: str = "ara+eng",
        min_confidence: float = 0.35,
        max_output_tokens: int = 2600,
    ):
        super().__init__(languages=languages, min_confidence=min_confidence)
        self.llm = llm
        self.max_output_tokens = max_output_tokens

    def is_available(self) -> bool:
        return self.llm is not None and self.llm.supports_images()

    def recognize(self, image: bytes, *, hint: str = "") -> OCRResult:
        if not image:
            return OCRResult.empty(self.name, error="empty image")
        if not self.is_available():
            return OCRResult.empty(
                self.name,
                error="The configured model cannot read images, so OCR is unavailable.",
            )

        prepared, media_type = normalize_image(ensure_min_size(image), max_edge=MAX_IMAGE_EDGE)
        system = SYSTEM_PROMPT
        if hint == "handwriting":
            system = f"{SYSTEM_PROMPT}\n\n{HANDWRITING_HINT}"

        try:
            response = self.llm.complete(
                [
                    LLMMessage(
                        role=Role.USER,
                        text="Transcribe every piece of text visible in this image.",
                        images=[ImagePart(data=prepared, media_type=media_type)],
                    )
                ],
                system=system,
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
                json_mode=True,
            )
        except ProviderError as exc:
            # Expected failure mode (rate limit, outage, capability) — the page
            # simply has no OCR text and the document keeps going.
            logger.warning("Vision OCR unavailable: %s", exc)
            return OCRResult.empty(self.name, error=exc.user_message)
        except Exception as exc:  # noqa: BLE001 - never abort ingestion
            logger.exception("Vision OCR failed unexpectedly")
            return OCRResult.empty(self.name, error=f"OCR failed: {type(exc).__name__}")

        return self._parse(response.text)

    # ------------------------------------------------------------------ #
    def _parse(self, raw: str) -> OCRResult:
        data = _loads(raw)
        if data is None:
            # The model answered in prose; treat the whole reply as the text but
            # flag it, since we could not verify the confidence.
            text = raw.strip()
            if not text:
                return OCRResult.empty(self.name, error="empty OCR response")
            return OCRResult(
                text=text, confidence=None, engine=self.name, uncertain=True
            ).finalize(min_confidence=self.min_confidence)

        text = str(data.get("text", "")).strip()
        confidence = _as_float(data.get("confidence"))
        handwriting = bool(data.get("has_handwriting", False))

        result = OCRResult(
            text=text,
            confidence=confidence,
            engine=self.name,
            uncertain=handwriting or (confidence is not None and confidence < self.min_confidence),
        )
        if not text:
            result.error = str(data.get("notes") or "") or None
        return result.finalize(min_confidence=self.min_confidence)


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


def _as_float(value: object) -> Optional[float]:
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
