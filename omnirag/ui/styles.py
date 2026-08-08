"""Custom CSS.

Kept deliberately small and in one place. Streamlit's own components do the
heavy lifting; this only adds what the theme cannot express — source cards,
status pills, RTL handling for Arabic, and a few spacing corrections.
"""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
/* ---------- layout ---------- */
.block-container { padding-top: 2.2rem; padding-bottom: 5rem; max-width: 1080px; }
section[data-testid="stSidebar"] { width: 358px !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
#MainMenu, footer { visibility: hidden; }

/* ---------- brand ---------- */
.omni-brand { display:flex; align-items:center; gap:.65rem; margin-bottom:.15rem; }
.omni-brand-mark {
  width:36px; height:36px; border-radius:10px; flex:0 0 36px;
  background: linear-gradient(135deg,#6366f1 0%,#0ea5e9 55%,#06b6d4 100%);
  display:flex; align-items:center; justify-content:center;
  color:#fff; font-weight:700; font-size:1rem; letter-spacing:-.02em;
}
.omni-brand-text { display:flex; flex-direction:column; line-height:1.15; }
.omni-brand-title { font-size:1.16rem; font-weight:700; letter-spacing:-.02em; }
.omni-brand-sub { font-size:.72rem; opacity:.62; }

/* ---------- pills ---------- */
.omni-pill {
  display:inline-flex; align-items:center; gap:.32rem;
  padding:.13rem .55rem; border-radius:999px;
  font-size:.7rem; font-weight:600; line-height:1.5;
  border:1px solid transparent; white-space:nowrap;
}
.omni-pill-ok    { background:rgba(16,185,129,.13); color:#059669; border-color:rgba(16,185,129,.28); }
.omni-pill-warn  { background:rgba(245,158,11,.14); color:#b45309; border-color:rgba(245,158,11,.30); }
.omni-pill-err   { background:rgba(239,68,68,.13);  color:#dc2626; border-color:rgba(239,68,68,.28); }
.omni-pill-info  { background:rgba(99,102,241,.13); color:#4f46e5; border-color:rgba(99,102,241,.28); }
.omni-pill-muted { background:rgba(120,120,130,.13); color:#6b7280; border-color:rgba(120,120,130,.24); }

/* ---------- document card ---------- */
.omni-doc { padding:.15rem 0 .3rem 0; }
.omni-doc-name {
  font-size:.86rem; font-weight:600; letter-spacing:-.01em;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.omni-doc-meta { font-size:.71rem; opacity:.6; margin-top:.1rem; }

/* ---------- source cards ---------- */
.omni-source {
  border:1px solid rgba(128,128,140,.22); border-left:3px solid #6366f1;
  border-radius:9px; padding:.6rem .75rem; margin-bottom:.5rem;
  background:rgba(128,128,140,.045);
}
.omni-source-head {
  display:flex; align-items:center; gap:.45rem;
  flex-wrap:wrap; margin-bottom:.35rem;
}
.omni-source-ref { font-size:.79rem; font-weight:650; letter-spacing:-.01em; }
.omni-source-body {
  font-size:.8rem; line-height:1.55; opacity:.86;
  white-space:pre-wrap; word-break:break-word;
}
.omni-source-uncited { border-left-color:rgba(128,128,140,.4); opacity:.72; }

/* ---------- welcome ---------- */
.omni-hero { padding:2.4rem 0 1.1rem 0; text-align:center; }
.omni-hero h1 { font-size:2.05rem; font-weight:700; letter-spacing:-.035em; margin:0 0 .5rem 0; }
.omni-hero p { font-size:.95rem; opacity:.66; margin:0 auto; max-width:33rem; line-height:1.6; }
.omni-feature {
  border:1px solid rgba(128,128,140,.2); border-radius:11px;
  padding:.85rem .9rem; height:100%;
}
.omni-feature-title { font-size:.83rem; font-weight:650; margin-bottom:.2rem; }
.omni-feature-body { font-size:.76rem; opacity:.65; line-height:1.5; }

/* ---------- Arabic / RTL ---------- */
.omni-rtl { direction:rtl; text-align:right; }
[data-testid="stChatMessageContent"] p:lang(ar) { direction:rtl; text-align:right; }

/* ---------- misc ---------- */
.omni-caption { font-size:.72rem; opacity:.55; }
.omni-divider { height:1px; background:rgba(128,128,140,.18); margin:.75rem 0; }
div[data-testid="stChatInput"] textarea { font-size:.92rem; }
</style>
"""


def inject() -> None:
    """Inject the stylesheet once per rerun."""
    st.markdown(CSS, unsafe_allow_html=True)


__all__ = ["CSS", "inject"]
