"""Small reusable controls for chat messages."""

from __future__ import annotations

import html
import json

import streamlit.components.v1 as components

from omnirag.config.settings import get_settings
from omnirag.utils.logging import get_logger

logger = get_logger(__name__)


def copy_component_html(text: str, component_id: str = "copy_message") -> str:
    """Build an isolated clipboard button without exposing hidden metadata."""
    payload = json.dumps(text, ensure_ascii=False).replace("<", "\\u003c")
    safe_id = html.escape(component_id, quote=True)
    return f"""<!doctype html>
<html><body data-component-id="{safe_id}" style="margin:0;background:transparent">
<button id="copy" type="button" title="Copy" aria-label="Copy message">Copy</button>
<span id="status" role="status" aria-live="polite"></span>
<script>
const text = {payload};
const button = document.getElementById('copy');
const status = document.getElementById('status');
button.addEventListener('click', async () => {{
  try {{
    await navigator.clipboard.writeText(text);
    status.textContent = 'Copied';
  }} catch (_) {{
    const area = document.createElement('textarea');
    area.value = text; document.body.appendChild(area); area.select();
    document.execCommand('copy'); area.remove(); status.textContent = 'Copied';
  }}
  setTimeout(() => status.textContent = '', 1400);
}});
</script>
<style>
button {{ border:0; background:transparent; color:#777; cursor:pointer;
  font:12px system-ui; padding:2px 6px; border-radius:6px; }}
button:hover {{ background:rgba(127,127,127,.12); color:#333; }}
#status {{ color:#56845f; font:11px system-ui; margin-left:4px; }}
</style></body></html>"""


def render_copy_button(*, text: str, key: str) -> None:
    # ``components.html`` instances are position-scoped rather than keyed; the
    # stable message key is embedded in the isolated payload for deterministic
    # DOM identity without creating a Streamlit widget/rerun.
    components.html(copy_component_html(text, key), height=28, scrolling=False)
    if get_settings().debug_generation:
        logger.info(
            "Generation lifecycle stage=clipboard component_id=%s clipboard_chars=%d",
            key,
            len(text),
        )


def action_key(action: str, message_id: str) -> str:
    """Stable collision-free Streamlit key for a message action."""
    return f"{action}_{html.escape(message_id, quote=True)}"


__all__ = ["action_key", "copy_component_html", "render_copy_button"]
