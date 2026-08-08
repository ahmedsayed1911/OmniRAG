"""Document intelligence: OCR, handwriting, visual understanding, tables, layout."""

from omnirag.intelligence.handwriting import HandwritingExtractor, HandwritingResult
from omnirag.intelligence.layout import (
    PageLayoutInfo,
    TextSpan,
    analyze_page_layout,
    looks_like_heading,
)
from omnirag.intelligence.ocr import OCREngine, OCRResult, build_ocr_engine
from omnirag.intelligence.tables import build_table, summarize_table, table_to_text
from omnirag.intelligence.vision import (
    VisionAnalyzer,
    VisualAnalysis,
    build_vision_analyzer,
)

__all__ = [
    "HandwritingExtractor",
    "HandwritingResult",
    "OCREngine",
    "OCRResult",
    "PageLayoutInfo",
    "TextSpan",
    "VisionAnalyzer",
    "VisualAnalysis",
    "analyze_page_layout",
    "build_ocr_engine",
    "build_table",
    "build_vision_analyzer",
    "looks_like_heading",
    "summarize_table",
    "table_to_text",
]
