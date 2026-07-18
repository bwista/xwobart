"""Acquisition and caching. Pitch data comes from the local monthly Statcast cache
via kinferencetoolkit's loader — no pybaseball.statcast() pulls (spec §5.1). The only
network pulls are the two seasonal leaderboards (sprint speed, expected stats)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import polars as pl

from src.config import Config

KEEP_COLUMNS = [
    "game_pk", "game_date", "game_year", "batter", "events", "description",
    "des", "type", "bb_type", "launch_speed", "launch_angle",
    "launch_speed_angle", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom",
]


def _retry(fn: Callable, tries: int = 3, wait_s: float = 20.0):
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == tries:
                raise
            print(f"  retry {attempt}/{tries} after error: {exc}")
            time.sleep(wait_s * attempt)


def coverage_gaps(season: int, min_date: str, max_date: str, window: tuple[str, str]) -> list[str]:
    """Compare a season cache's date range to the expected regular-season window.
    Dates are 'YYYY-MM-DD' strings; lexicographic comparison is correct."""
    gaps = []
    if min_date > window[0]:
        gaps.append(f"{season}: cache starts {min_date}, season starts {window[0]}")
    if max_date < window[1]:
        gaps.append(f"{season}: cache ends {max_date}, season ends {window[1]}")
    return gaps
