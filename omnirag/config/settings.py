"""Central configuration.

Single source of truth for every tunable in OmniRAG. No other module calls
``os.getenv``. Values are read from the process environment; the Streamlit entry
point copies ``st.secrets`` into the environment *before* importing this module
(see ``omnirag.config.bootstrap``), which keeps the core engine free of any
Streamlit dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Any, Dict, List, Optional

from omnirag.core.exceptions import ConfigurationError, MissingCredentialError

# --------------------------------------------------------------------------- #
# env helpers
# --------------------------------------------------------------------------- #
_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}


def _get(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip()


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name).lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return default


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError as exc:
        raise ConfigurationError(
            f"{name}={raw!r} is not a valid integer",
            user_message=f"Setting `{name}` must be a whole number (got `{raw}`).",
        ) from exc


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name}={raw!r} is not a valid number",
            user_message=f"Setting `{name}` must be a number (got `{raw}`).",
        ) from exc


def _first_env(*names: str, default: str = "") -> str:
    """Return the first non-empty value among ``names``.

    Lets users configure either the generic OmniRAG names (``LLM_API_KEY``) or
    the vendor-native ones they already have (``OPENAI_API_KEY``).
    """
    for name in names:
        value = _get(name)
        if value:
            return value
    return default


# --------------------------------------------------------------------------- #
# Sub-configurations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProviderEndpoint:
    """Credentials + model for one concrete LLM vendor in the chain."""

    provider: str                      # gemini | openrouter | openai | anthropic | mock
    api_key: str = ""
    model: str = ""
    vision_model: str = ""
    base_url: str = ""
    #: Force the image-capability answer instead of inferring it from the model
    #: name (``OPENROUTER_MODEL_SUPPORTS_IMAGES=true|false``).
    supports_images: Optional[bool] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) or self.provider in {"mock", "local"}

    @property
    def effective_vision_model(self) -> str:
        return self.vision_model or self.model


@dataclass(frozen=True)
class LLMSettings:
    """LLM configuration, including the ordered provider chain.

    ``provider``/``api_key``/``model`` describe the *primary* endpoint and are
    kept as flat fields for backwards compatibility with the generic
    ``LLM_PROVIDER`` / ``LLM_API_KEY`` variables.
    """

    provider: str = "gemini"
    api_key: str = ""
    base_url: str = ""
    model: str = "gemini-2.0-flash"
    vision_model: str = ""
    temperature: float = 0.1
    max_output_tokens: int = 1400
    timeout_s: float = 90.0
    enable_multimodal: bool = True
    max_images_per_answer: int = 4

    #: Ordered chain: index 0 is primary, the rest are fallbacks.
    endpoints: tuple[ProviderEndpoint, ...] = ()
    enable_fallback: bool = True
    #: Quick per-provider retries before failing over. Kept small so the
    #: Streamlit request never hangs behind a long backoff loop.
    retry_attempts: int = 2

    @property
    def effective_vision_model(self) -> str:
        return self.vision_model or self.model

    @property
    def configured_endpoints(self) -> tuple[ProviderEndpoint, ...]:
        return tuple(e for e in self.endpoints if e.is_configured)

    @property
    def is_configured(self) -> bool:
        return bool(self.configured_endpoints)

    @property
    def primary_endpoint(self) -> Optional[ProviderEndpoint]:
        endpoints = self.configured_endpoints
        return endpoints[0] if endpoints else None

    @property
    def fallback_active(self) -> bool:
        return self.enable_fallback and len(self.configured_endpoints) > 1

    @property
    def chain_label(self) -> str:
        """``gemini → openrouter`` — for logs and the UI settings panel."""
        return " → ".join(
            f"{e.provider}:{e.model}" for e in self.configured_endpoints
        ) or "not configured"

    def require_key(self) -> str:
        if not self.is_configured:
            raise MissingCredentialError(
                "GEMINI_API_KEY or OPENROUTER_API_KEY", "answer generation"
            )
        primary = self.primary_endpoint
        return primary.api_key if primary else ""


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str = "openai"
    api_key: str = ""
    base_url: str = ""
    model: str = "text-embedding-3-large"
    dimensions: int = 0            # 0 = provider default (probed on first call)
    batch_size: int = 64
    timeout_s: float = 60.0
    max_chars_per_input: int = 8000

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) or self.provider in {"hash", "local", "mock"}


@dataclass(frozen=True)
class VectorStoreSettings:
    provider: str = "auto"          # auto | qdrant | memory
    url: str = ""
    api_key: str = ""
    collection: str = "omnirag_chunks"
    prefer_grpc: bool = False
    timeout_s: float = 30.0
    distance: str = "cosine"

    @property
    def use_qdrant(self) -> bool:
        if self.provider == "qdrant":
            return True
        if self.provider == "memory":
            return False
        return bool(self.url)  # auto


@dataclass(frozen=True)
class RerankSettings:
    provider: str = "auto"          # auto | cohere | jina | llm | heuristic | none
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    timeout_s: float = 30.0
    enabled: bool = True


@dataclass(frozen=True)
class OCRSettings:
    provider: str = "auto"          # auto | vision | tesseract | none
    languages: str = "ara+eng"
    tesseract_cmd: str = ""
    min_confidence: float = 0.35
    timeout_s: float = 60.0


@dataclass(frozen=True)
class VisionSettings:
    enabled: bool = True
    max_images_per_document: int = 40
    min_image_pixels: int = 110 * 110
    max_image_edge: int = 1400
    page_render_dpi: int = 170
    jpeg_quality: int = 82
    describe_page_snapshots: bool = True


@dataclass(frozen=True)
class ChunkingSettings:
    chunk_size: int = 1100          # characters (structure-aware, not blind)
    chunk_overlap: int = 150
    min_chunk_size: int = 120
    max_chunk_size: int = 2600
    respect_tables: bool = True


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = 24                 # candidates fetched per retriever
    rerank_top_k: int = 8           # contexts handed to the LLM
    strategy: str = "hybrid"
    use_keyword: bool = True
    rrf_k: int = 60
    min_score: float = 0.0
    query_rewrite: bool = True
    max_expansions: int = 3
    max_context_chars: int = 22000


@dataclass(frozen=True)
class UploadSettings:
    max_upload_mb: float = 50.0
    max_files: int = 25
    max_pages_per_document: int = 400
    allowed_extensions: tuple[str, ...] = (
        "pdf", "docx", "pptx", "txt", "md", "markdown",
        "jpg", "jpeg", "png", "webp",
    )


@dataclass(frozen=True)
class AppSettings:
    """Root settings object. Build with :func:`get_settings`."""

    app_name: str = "OmniRAG"
    environment: str = "production"
    log_level: str = "INFO"
    debug_panels: bool = False
    workspace_dir: str = ""
    session_ttl_minutes: int = 240
    max_history_messages: int = 10

    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)
    ocr: OCRSettings = field(default_factory=OCRSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    upload: UploadSettings = field(default_factory=UploadSettings)

    # ------------------------------------------------------------------ #
    def validation_issues(self) -> List[str]:
        """Human-readable blocking problems (empty list == ready to answer)."""
        issues: List[str] = []
        if not self.llm.is_configured:
            issues.append(
                "No language-model provider is configured. Set `GEMINI_API_KEY` "
                "(primary) and/or `OPENROUTER_API_KEY` (fallback) in your "
                "environment or Streamlit secrets."
            )
        if not self.embedding.is_configured:
            issues.append(
                "No embedding provider is configured — set `EMBEDDING_API_KEY`."
            )
        return issues

    def warnings(self) -> List[str]:
        """Non-blocking notes worth surfacing in the UI."""
        notes: List[str] = []
        if self.embedding.provider == "hash":
            notes.append(
                "Embeddings are using the offline `hash` provider: retrieval is "
                "lexical-only and semantic quality will be poor. Set "
                "`GEMINI_API_KEY` or `EMBEDDING_API_KEY` for real semantic search."
            )
        if not self.vector_store.use_qdrant:
            notes.append(
                "Vector store is in-memory: your index is lost when the app "
                "restarts. Set `QDRANT_URL` for persistence."
            )
        if not self.vision.enabled:
            notes.append("Visual understanding is disabled (`VISION_ENABLED=false`).")
        if self.llm.is_configured and not self.llm.fallback_active:
            if len(self.llm.configured_endpoints) == 1 and self.llm.enable_fallback:
                only = self.llm.configured_endpoints[0].provider
                other = "OPENROUTER_API_KEY" if only == "gemini" else "GEMINI_API_KEY"
                notes.append(
                    f"Only one LLM provider (`{only}`) is configured, so there is no "
                    f"automatic failover. Add `{other}` to enable it."
                )
        return notes

    @property
    def llm_chain_label(self) -> str:
        return self.llm.chain_label

    @property
    def is_ready(self) -> bool:
        return not self.validation_issues()

    def redacted(self) -> Dict[str, Any]:
        """Safe-to-log view: every secret is masked. Used by the debug panel."""
        return _redact_dataclass(self)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build_settings() -> AppSettings:
    """Read the environment and produce an :class:`AppSettings` (uncached)."""

    llm = _build_llm_settings()
    llm_key = llm.api_key
    llm_provider = llm.provider

    # Embeddings are deliberately independent of the LLM failover chain: a
    # Gemini outage must never invalidate vectors already written to Qdrant.
    emb_provider = _get("EMBEDDING_PROVIDER", "").lower()
    emb_key = _get("EMBEDDING_API_KEY")
    openai_key = _get("OPENAI_API_KEY")
    gemini_key = _first_env("GEMINI_API_KEY", "GOOGLE_API_KEY")

    if not emb_provider:
        # Pick whichever multilingual embedding API the user already has a key
        # for; otherwise fall back to the offline `hash` provider so the app
        # still runs (loudly flagged in `AppSettings.warnings`).
        if emb_key:
            emb_provider = "openai"
        elif openai_key:
            emb_provider = "openai"
        elif gemini_key:
            emb_provider = "gemini"
        else:
            emb_provider = "hash"

    if not emb_key:
        if emb_provider in {"openai", "openai_compatible"}:
            emb_key = openai_key or (llm_key if llm_provider in {"openai", "openai_compatible"} else "")
        elif emb_provider == "gemini":
            emb_key = gemini_key or (llm_key if llm_provider == "gemini" else "")
        elif emb_provider == "cohere":
            emb_key = _get("COHERE_API_KEY")
        elif emb_provider == "jina":
            emb_key = _get("JINA_API_KEY")

    # An explicitly selected remote provider without credentials is a
    # configuration availability problem, not a reason to make ingestion
    # unusable. Select the existing offline provider so the UI also reports
    # the reduced-quality mode accurately.
    if emb_provider not in {"hash", "mock"} and not emb_key:
        emb_provider = "hash"

    embedding = EmbeddingSettings(
        provider=emb_provider,
        api_key=emb_key,
        base_url=_get("EMBEDDING_BASE_URL"),
        model=_get("EMBEDDING_MODEL", _default_embedding_model(emb_provider)),
        dimensions=_get_int("EMBEDDING_DIMENSIONS", 0),
        batch_size=_get_int("EMBEDDING_BATCH_SIZE", 64),
        timeout_s=_get_float("EMBEDDING_TIMEOUT_S", 60.0),
        max_chars_per_input=_get_int("EMBEDDING_MAX_CHARS", 8000),
    )

    vector_store = VectorStoreSettings(
        provider=_get("VECTOR_STORE_PROVIDER", "auto").lower(),
        url=_get("QDRANT_URL"),
        api_key=_get("QDRANT_API_KEY"),
        collection=_get("QDRANT_COLLECTION", "omnirag_chunks"),
        prefer_grpc=_get_bool("QDRANT_PREFER_GRPC", False),
        timeout_s=_get_float("QDRANT_TIMEOUT_S", 30.0),
        distance=_get("QDRANT_DISTANCE", "cosine").lower(),
    )

    rerank_provider = _get("RERANK_PROVIDER", "auto").lower()
    rerank = RerankSettings(
        provider=rerank_provider,
        api_key=_first_env("RERANK_API_KEY", "COHERE_API_KEY", "JINA_API_KEY"),
        model=_get("RERANK_MODEL", _default_rerank_model(rerank_provider)),
        base_url=_get("RERANK_BASE_URL"),
        timeout_s=_get_float("RERANK_TIMEOUT_S", 30.0),
        enabled=_get_bool("RERANK_ENABLED", True),
    )

    ocr = OCRSettings(
        provider=_get("OCR_PROVIDER", "auto").lower(),
        languages=_get("OCR_LANGUAGES", "ara+eng"),
        tesseract_cmd=_get("TESSERACT_CMD"),
        min_confidence=_get_float("OCR_MIN_CONFIDENCE", 0.35),
        timeout_s=_get_float("OCR_TIMEOUT_S", 60.0),
    )

    vision = VisionSettings(
        enabled=_get_bool("VISION_ENABLED", True),
        max_images_per_document=_get_int("MAX_IMAGES_PER_DOCUMENT", 40),
        min_image_pixels=_get_int("MIN_IMAGE_PIXELS", 110 * 110),
        max_image_edge=_get_int("MAX_IMAGE_EDGE", 1400),
        page_render_dpi=_get_int("PAGE_RENDER_DPI", 170),
        jpeg_quality=_get_int("IMAGE_JPEG_QUALITY", 82),
        describe_page_snapshots=_get_bool("DESCRIBE_PAGE_SNAPSHOTS", True),
    )

    chunking = ChunkingSettings(
        chunk_size=_get_int("CHUNK_SIZE", 1100),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 150),
        min_chunk_size=_get_int("MIN_CHUNK_SIZE", 120),
        max_chunk_size=_get_int("MAX_CHUNK_SIZE", 2600),
        respect_tables=_get_bool("CHUNK_RESPECT_TABLES", True),
    )
    if chunking.chunk_overlap >= chunking.chunk_size:
        raise ConfigurationError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE",
            user_message="`CHUNK_OVERLAP` must be smaller than `CHUNK_SIZE`.",
        )

    retrieval = RetrievalSettings(
        top_k=_get_int("TOP_K", 24),
        rerank_top_k=_get_int("RERANK_TOP_K", 8),
        strategy=_get("RETRIEVAL_STRATEGY", "hybrid").lower(),
        use_keyword=_get_bool("USE_KEYWORD_SEARCH", True),
        rrf_k=_get_int("RRF_K", 60),
        min_score=_get_float("MIN_SCORE", 0.0),
        query_rewrite=_get_bool("QUERY_REWRITE", True),
        max_expansions=_get_int("MAX_QUERY_EXPANSIONS", 3),
        max_context_chars=_get_int("MAX_CONTEXT_CHARS", 22000),
    )

    upload = UploadSettings(
        max_upload_mb=_get_float("MAX_UPLOAD_MB", 50.0),
        max_files=_get_int("MAX_FILES", 25),
        max_pages_per_document=_get_int("MAX_PAGES_PER_DOCUMENT", 400),
    )

    return AppSettings(
        environment=_get("APP_ENV", "production"),
        log_level=_get("LOG_LEVEL", "INFO").upper(),
        debug_panels=_get_bool("DEBUG_PANELS", False),
        workspace_dir=_get("OMNIRAG_WORKSPACE"),
        session_ttl_minutes=_get_int("SESSION_TTL_MINUTES", 240),
        max_history_messages=_get_int("MAX_HISTORY_MESSAGES", 10),
        llm=llm,
        embedding=embedding,
        vector_store=vector_store,
        rerank=rerank,
        ocr=ocr,
        vision=vision,
        chunking=chunking,
        retrieval=retrieval,
        upload=upload,
    )


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #
#: Field names whose value is a credential. Matched exactly, so ordinary fields
#: that merely contain the substring "key" (``use_keyword``) are not mangled.
_SECRET_FIELDS = frozenset({"api_key", "key", "token", "secret", "password"})


def _is_secret_field(name: str) -> bool:
    return name in _SECRET_FIELDS or name.endswith("_api_key") or name.endswith("_token")


def _mask(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    return f"{value[:3]}…{value[-2:]}" if len(value) > 8 else "***"


def _redact_value(name: str, value: Any) -> Any:
    if _is_secret_field(name):
        return _mask(value)
    if hasattr(value, "__dataclass_fields__"):
        return _redact_dataclass(value)
    if isinstance(value, (list, tuple)):
        # e.g. LLMSettings.endpoints — each element is itself redacted, so a
        # nested credential can never escape into a log line.
        return [_redact_value(name, item) for item in value]
    return value


def _redact_dataclass(obj: Any) -> Dict[str, Any]:
    return {f.name: _redact_value(f.name, getattr(obj, f.name)) for f in fields(obj)}


def _build_llm_settings() -> LLMSettings:
    """Assemble the ordered LLM provider chain from the environment.

    Default strategy: **Gemini primary, OpenRouter fallback**. Providers with no
    credentials are dropped from the chain, so every combination works:

    * both configured    -> Gemini primary, OpenRouter fallback
    * only Gemini        -> Gemini alone, no fallback
    * only OpenRouter    -> OpenRouter becomes the active provider
    * neither configured -> empty chain; the UI shows a configuration error
    """
    primary_name = _get("PRIMARY_LLM_PROVIDER", "").lower()
    fallback_name = _get("FALLBACK_LLM_PROVIDER", "").lower()
    legacy_provider = _get("LLM_PROVIDER", "").lower()

    if not primary_name:
        primary_name = legacy_provider or "gemini"
    if not fallback_name and not legacy_provider:
        fallback_name = "openrouter"

    order: List[str] = [primary_name]
    if fallback_name and fallback_name not in order and fallback_name != "none":
        order.append(fallback_name)

    # Any other vendor that has a key becomes a last-resort link in the chain.
    for extra in ("gemini", "openrouter", "openai", "anthropic"):
        if extra not in order and _endpoint_key(extra):
            order.append(extra)

    endpoints = tuple(_build_endpoint(name) for name in order)
    configured = tuple(e for e in endpoints if e.is_configured)
    primary = configured[0] if configured else endpoints[0]

    return LLMSettings(
        provider=primary.provider,
        api_key=primary.api_key,
        base_url=primary.base_url,
        model=primary.model,
        vision_model=primary.vision_model,
        temperature=_get_float("LLM_TEMPERATURE", 0.1),
        max_output_tokens=_get_int("LLM_MAX_OUTPUT_TOKENS", 1400),
        timeout_s=_get_float("LLM_TIMEOUT_S", 90.0),
        enable_multimodal=_get_bool("LLM_ENABLE_MULTIMODAL", True),
        max_images_per_answer=_get_int("MAX_IMAGES_PER_ANSWER", 4),
        endpoints=endpoints,
        enable_fallback=_get_bool("ENABLE_PROVIDER_FALLBACK", True),
        retry_attempts=_get_int("LLM_RETRY_ATTEMPTS", 2),
    )


#: Env var names holding the API key for each supported vendor, in priority
#: order. ``LLM_API_KEY`` is the generic escape hatch.
_ENDPOINT_KEY_VARS: Dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openai_compatible": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "mock": (),
    "local": (),
}


def _endpoint_key(provider: str) -> str:
    return _first_env(*_ENDPOINT_KEY_VARS.get(provider, ()))


def _build_endpoint(provider: str) -> ProviderEndpoint:
    """Resolve credentials/model for one vendor, honouring generic overrides."""
    prefix = {
        "gemini": "GEMINI",
        "openrouter": "OPENROUTER",
        "openai": "OPENAI",
        "openai_compatible": "OPENAI",
        "anthropic": "ANTHROPIC",
    }.get(provider, provider.upper())

    api_key = _get(f"{prefix}_API_KEY") or _endpoint_key(provider)
    generic_key = _get("LLM_API_KEY")
    generic_provider = _get("LLM_PROVIDER", "").lower()
    if not api_key and generic_key and (not generic_provider or generic_provider == provider):
        api_key = generic_key

    model = _get(f"{prefix}_MODEL")
    if not model and (not generic_provider or generic_provider == provider):
        model = _get("LLM_MODEL")
    model = model or _default_model(provider)

    vision_model = _get(f"{prefix}_VISION_MODEL") or _get("VISION_MODEL")
    base_url = _get(f"{prefix}_BASE_URL") or _get("LLM_BASE_URL")

    supports_images: Optional[bool] = None
    raw_capability = _get(f"{prefix}_MODEL_SUPPORTS_IMAGES").lower()
    if raw_capability in _TRUE:
        supports_images = True
    elif raw_capability in _FALSE:
        supports_images = False

    return ProviderEndpoint(
        provider=provider,
        api_key=api_key,
        model=model,
        vision_model=vision_model,
        base_url=base_url,
        supports_images=supports_images,
    )


def _default_model(provider: str) -> str:
    return {
        "openai": "gpt-4o-mini",
        "openai_compatible": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-5",
        "gemini": "gemini-2.0-flash",
        # A multimodal default so visual evidence works out of the box.
        "openrouter": "google/gemini-2.0-flash-001",
        "mock": "mock-llm",
    }.get(provider, "gpt-4o-mini")


def _default_embedding_model(provider: str) -> str:
    return {
        "openai": "text-embedding-3-large",
        "openai_compatible": "text-embedding-3-large",
        "gemini": "gemini-embedding-001",
        "cohere": "embed-multilingual-v3.0",
        "jina": "jina-embeddings-v3",
        "hash": "hash-1024",
        "mock": "mock-embed",
    }.get(provider, "text-embedding-3-large")


def _default_rerank_model(provider: str) -> str:
    return {
        "cohere": "rerank-multilingual-v3.0",
        "jina": "jina-reranker-v2-base-multilingual",
    }.get(provider, "")


# --------------------------------------------------------------------------- #
# Cached accessor
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Process-wide settings singleton."""
    return build_settings()


def reset_settings_cache() -> None:
    """Drop the cached settings (used by tests and after secrets change)."""
    get_settings.cache_clear()


__all__ = [
    "AppSettings",
    "LLMSettings",
    "EmbeddingSettings",
    "VectorStoreSettings",
    "RerankSettings",
    "OCRSettings",
    "VisionSettings",
    "ChunkingSettings",
    "RetrievalSettings",
    "UploadSettings",
    "build_settings",
    "get_settings",
    "reset_settings_cache",
]
