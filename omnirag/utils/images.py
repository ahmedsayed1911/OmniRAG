"""Image helpers used by ingestion, the vision pipeline and the UI.

Pillow is a hard dependency (small, wheels everywhere). All functions degrade
gracefully: a broken image returns ``None`` rather than raising, because one bad
embedded picture must never abort a document.
"""

from __future__ import annotations

import base64
import io
from typing import Optional, Tuple

from omnirag.utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - import guard
    from PIL import Image, ImageStat

    PIL_AVAILABLE = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageStat = None  # type: ignore[assignment]
    PIL_AVAILABLE = False


def open_image(data: bytes) -> Optional["Image.Image"]:
    if not PIL_AVAILABLE or not data:
        return None
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except Exception as exc:
        logger.debug("Could not open image (%d bytes): %s", len(data), exc)
        return None


def image_size(data: bytes) -> Tuple[int, int]:
    image = open_image(data)
    return (image.width, image.height) if image is not None else (0, 0)


def normalize_image(
    data: bytes,
    *,
    max_edge: int = 1400,
    jpeg_quality: int = 82,
    force_format: Optional[str] = None,
) -> Tuple[bytes, str]:
    """Downscale and re-encode an image for API transport.

    Returns ``(bytes, media_type)``. Keeps PNG for images with transparency or
    sharp text edges (screenshots/diagrams), otherwise JPEG for size. On any
    failure the original bytes are returned unchanged.
    """
    image = open_image(data)
    if image is None:
        return data, "image/png"

    try:
        has_alpha = image.mode in ("RGBA", "LA", "P") and "transparency" in image.info
        fmt = force_format or ("PNG" if has_alpha else "JPEG")

        if max(image.width, image.height) > max_edge:
            ratio = max_edge / float(max(image.width, image.height))
            new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
            image = image.resize(new_size, Image.LANCZOS)

        if fmt == "JPEG" and image.mode not in ("RGB", "L"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode in ("RGBA", "LA"):
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert("RGB")

        buffer = io.BytesIO()
        if fmt == "JPEG":
            image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            return buffer.getvalue(), "image/jpeg"
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), "image/png"
    except Exception as exc:
        logger.debug("Image normalisation failed: %s", exc)
        return data, "image/png"


def to_data_url(data: bytes, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def is_probably_blank(data: bytes, *, std_threshold: float = 4.0) -> bool:
    """Detect near-uniform images (blank scans, separator rules, spacer GIFs).

    Skipping these saves real money: they would otherwise each trigger a vision
    API call that returns nothing useful.
    """
    image = open_image(data)
    if image is None or ImageStat is None:
        return False
    try:
        grayscale = image.convert("L")
        if grayscale.width < 4 or grayscale.height < 4:
            return True
        stat = ImageStat.Stat(grayscale)
        return float(stat.stddev[0]) < std_threshold
    except Exception:
        return False


def is_probably_decorative(
    data: bytes, *, min_pixels: int = 110 * 110, max_aspect: float = 12.0
) -> bool:
    """Heuristic filter for logos, bullets, rules and other non-informative art."""
    width, height = image_size(data)
    if width == 0 or height == 0:
        return True
    if width * height < min_pixels:
        return True
    aspect = max(width / height, height / width)
    if aspect > max_aspect:
        return True
    return is_probably_blank(data)


def crop(data: bytes, box: Tuple[float, float, float, float], *, padding: int = 6) -> Optional[bytes]:
    """Crop a region (pixel coordinates) with a little padding."""
    image = open_image(data)
    if image is None:
        return None
    try:
        x0, y0, x1, y1 = box
        region = (
            max(0, int(x0) - padding),
            max(0, int(y0) - padding),
            min(image.width, int(x1) + padding),
            min(image.height, int(y1) + padding),
        )
        if region[2] <= region[0] or region[3] <= region[1]:
            return None
        buffer = io.BytesIO()
        image.crop(region).save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()
    except Exception as exc:
        logger.debug("Crop failed: %s", exc)
        return None


def ensure_min_size(data: bytes, min_edge: int = 320) -> bytes:
    """Upscale very small crops so OCR/vision models can read them."""
    image = open_image(data)
    if image is None:
        return data
    if max(image.width, image.height) >= min_edge:
        return data
    try:
        ratio = min_edge / float(max(1, max(image.width, image.height)))
        resized = image.resize(
            (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
            Image.LANCZOS,
        )
        buffer = io.BytesIO()
        resized.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:
        return data
