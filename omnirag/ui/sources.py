"""Citation and evidence rendering.

The user must be able to inspect the evidence behind any answer: which file,
which page, what type of content, the exact passage, and — for visual sources —
the original image the model actually looked at.
"""

from __future__ import annotations

import html
from typing import Dict, List, Optional, Sequence

import streamlit as st

from omnirag.core.models import Citation, RetrievalResult
from omnirag.storage.files import FileStore
from omnirag.ui.components import BLOCK_LABEL, SOURCE_LABEL, caption, pill
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)


def render_sources(
    citations: Sequence[Citation],
    *,
    used_indices: Optional[set] = None,
    file_store: Optional[FileStore] = None,
    key_prefix: str = "",
    retrieval: Optional[RetrievalResult] = None,
    show_diagnostics: bool = False,
) -> None:
    """Render the source panel beneath an answer."""
    if not citations:
        return

    cited = used_indices if used_indices is not None else {c.index for c in citations}
    cited_count = len([c for c in citations if c.index in cited])
    label = f"📎 Sources ({cited_count} cited of {len(citations)} retrieved)"

    with st.expander(label, expanded=False):
        if retrieval is not None and show_diagnostics:
            _render_retrieval_note(retrieval)

        for citation in citations:
            _render_card(
                citation,
                is_cited=citation.index in cited,
                file_store=file_store,
                key_prefix=key_prefix,
            )


def _render_retrieval_note(retrieval: RetrievalResult) -> None:
    bits: List[str] = [pill(f"{retrieval.strategy} search", "muted")]
    if retrieval.reranked:
        bits.append(pill("reranked", "muted"))
    if retrieval.expanded_queries:
        bits.append(pill(f"+{len(retrieval.expanded_queries)} query variants", "muted"))
    st.markdown(" ".join(bits), unsafe_allow_html=True)
    for note in retrieval.notes:
        caption(note)
    st.write("")


def _render_card(
    citation: Citation,
    *,
    is_cited: bool,
    file_store: Optional[FileStore],
    key_prefix: str,
) -> None:
    kind = BLOCK_LABEL.get(citation.block_type, "Text")
    origin = SOURCE_LABEL.get(citation.source_kind, "")

    badges = [pill(kind, "info")]
    if origin:
        badges.append(pill(origin, "muted"))
    if citation.uncertain:
        badges.append(pill("low confidence", "warn"))
    if citation.confidence is not None:
        badges.append(pill(f"{citation.confidence:.0%}", "muted"))
    if not is_cited:
        badges.append(pill("retrieved, not cited", "muted"))

    css_class = "omni-source" if is_cited else "omni-source omni-source-uncited"
    st.markdown(
        f"""
        <div class="{css_class}">
          <div class="omni-source-head">
            <span class="omni-source-ref">[{citation.index}] {html.escape(citation.filename)}
            — {html.escape(citation.page_label)}</span>
            {' '.join(badges)}
          </div>
          <div class="omni-source-body">{html.escape(citation.snippet)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if citation.visual_asset_id and file_store is not None:
        _render_visual(citation, file_store, key_prefix)


def _render_visual(citation: Citation, file_store: FileStore, key_prefix: str) -> None:
    """Show the original image that backed this source, on demand."""
    key = f"{key_prefix}_visual_{citation.index}_{citation.chunk_id[:8]}"
    with st.expander("🖼️ View original visual", expanded=False):
        try:
            data = file_store.get(citation.visual_asset_id or "")
        except Exception as exc:
            logger.warning("Could not load visual asset: %s", exc)
            data = None

        if not data:
            st.caption(
                "The original image is no longer available in this session's "
                "temporary storage."
            )
            return
        st.image(
            data,
            caption=f"{citation.filename} — {citation.page_label}",
            use_container_width=True,
        )


def render_inline_references(citations: Sequence[Citation], used: set) -> None:
    """Compact one-line reference list under an answer."""
    referenced = [c for c in citations if c.index in used]
    if not referenced:
        return
    parts = [
        f"`[{c.index}]` {c.filename} — {c.page_label}" for c in referenced
    ]
    caption(" · ".join(parts))


def group_citations(citations: Sequence[Citation]) -> Dict[str, List[Citation]]:
    grouped: Dict[str, List[Citation]] = {}
    for citation in citations:
        grouped.setdefault(citation.filename, []).append(citation)
    return grouped


__all__ = ["group_citations", "render_inline_references", "render_sources"]
