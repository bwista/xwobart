"""Shared setup for the review notebooks — import this from each notebook's first cell.

Owns the pieces every notebook used to copy-paste: repo-root discovery, the polars
display defaults, `jload` for results JSON, and `show_fig`. `show_fig` sizes each
figure from its native pixel width instead of a fixed 680px — the old fixed width
postage-stamped wide multi-panel figures like the calibration strip.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import polars as pl
from IPython.display import Image, Markdown, display

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_hide_column_data_types(True)

MAX_WIDTH = 980  # px — roughly the readable width of a rendered notebook


def jload(rel: str):
    """Load a JSON artifact from results/, e.g. jload("stage_C/metrics.json")."""
    return json.loads((RESULTS / rel).read_text())


def _png_width(path: Path) -> int | None:
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        return struct.unpack(">I", head[16:20])[0]
    except OSError:
        return None


def show_fig(rel: str, width: int | None = None, caption: str | None = None):
    """Display a results/ figure at its native width (capped at MAX_WIDTH so hi-dpi
    renders stay crisp instead of huge), with an optional italic caption under it."""
    p = RESULTS / rel
    if not p.exists():
        display(Markdown(f"_missing figure: {rel}_"))
        return
    if width is None:
        native = _png_width(p)
        width = min(native, MAX_WIDTH) if native else MAX_WIDTH
    display(Image(filename=str(p), width=width))
    if caption:
        display(Markdown(f"*{caption}*"))
