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


def merge_sprint_speed(bbe: pl.DataFrame, sprint: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Left-join seasonal sprint speed by (batter, season); impute the league median
    for that season where missing, flag with imputed_speed, report the rate."""
    med = sprint.group_by("season").agg(league_median=pl.col("sprint_speed").median())
    out = (
        bbe.join(sprint, left_on=["batter", "game_year"], right_on=["player_id", "season"], how="left")
        .join(med, left_on="game_year", right_on="season", how="left")
        .with_columns(imputed_speed=pl.col("sprint_speed").is_null())
        .with_columns(sprint_speed=pl.coalesce("sprint_speed", "league_median"))
        .drop("league_median")
    )
    rate = (
        out.group_by("game_year")
        .agg(imputation_rate=pl.col("imputed_speed").mean())
        .sort("game_year")
    )
    return out, rate


def _record_coverage(paths: dict[int, Path], cfg: Config, gap_path: Path) -> list[str]:
    """Compute + persist coverage gaps from the slim caches. Also runs on the
    pre-existing-cache path so run_v0 can always read coverage.json."""
    all_gaps: list[str] = []
    for y, p in paths.items():
        d = pl.scan_parquet(p).select(
            lo=pl.col("game_date").min(), hi=pl.col("game_date").max()
        ).collect()
        all_gaps.extend(coverage_gaps(y, d["lo"][0], d["hi"][0], cfg.season_windows[y]))
    gap_path.write_text(json.dumps(all_gaps, indent=2))
    if all_gaps:
        print("  Coverage gaps recorded. To fill: run KIT's "
              "`python pipeline/statcast_loader.py --update --date <YYYY-MM-DD>` "
              "for the missing tail, then rebuild with --force-data.")
    return all_gaps


def build_season_caches(cfg: Config, force: bool = False) -> dict[int, Path]:
    """Slim per-season parquets from the monthly cache, via KIT load_statcast.
    Skips the load when every season cache exists (spec §5.1). game_date is stored
    as a 'YYYY-MM-DD' string (spec §14 deviation: lexicographic compares are
    equivalent for the window checks and it sidesteps the cache's mixed dtypes).
    Coverage gaps are reported, recorded to data/raw/coverage.json, never
    auto-pulled (§5.2)."""
    from pipeline.statcast_loader import load_statcast  # KIT import kept local: heavy module

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {y: cfg.raw_dir / f"statcast-{y}-slim.parquet" for y in cfg.all_seasons}
    gap_path = cfg.raw_dir / "coverage.json"
    if not force and all(p.exists() for p in paths.values()):
        if not gap_path.exists():
            _record_coverage(paths, cfg, gap_path)
        return paths

    # load_statcast's documented warmup behavior also loads (start_year - 1) files;
    # the game_year filter below drops those rows.
    df = load_statcast(str(cfg.statcast_dir), start_year=min(cfg.all_seasons),
                       end_year=max(cfg.all_seasons))
    df = (
        df.with_columns(pl.col("game_date").cast(pl.Utf8).str.slice(0, 10))
        .filter((pl.col("game_type") == "R") & pl.col("game_year").is_in(cfg.all_seasons))
        .select(KEEP_COLUMNS)
    )
    for y, p in paths.items():
        part = df.filter(pl.col("game_year") == y)
        part.write_parquet(p)
        print(f"  {y}: {part.height:,} rows, "
              f"{part['game_date'].min()} -> {part['game_date'].max()}")
    _record_coverage(paths, cfg, gap_path)
    return paths


def load_seasons(cfg: Config, seasons: list[int]) -> pl.DataFrame:
    paths = build_season_caches(cfg)
    return pl.concat([pl.read_parquet(paths[y]) for y in seasons])


def fetch_sprint_speed(cfg: Config, force: bool = False) -> pl.DataFrame:
    """Seasonal sprint speed per player (min_opp qualifier), one cached parquet per
    season. Returns columns: player_id, season, sprint_speed."""
    frames = []
    for year in cfg.all_seasons:
        cache = cfg.raw_dir / f"sprint_speed-{year}.parquet"
        if cache.exists() and not force:
            frames.append(pl.read_parquet(cache))
            continue
        from pybaseball import statcast_sprint_speed

        pdf = _retry(lambda: statcast_sprint_speed(year, cfg.sprint_min_opp))
        df = pl.from_pandas(pdf)
        id_col = "player_id" if "player_id" in df.columns else "id"
        out = df.select(
            player_id=pl.col(id_col).cast(pl.Int64),
            sprint_speed=pl.col("sprint_speed").cast(pl.Float64),
        ).with_columns(season=pl.lit(year, dtype=pl.Int64))
        out.write_parquet(cache)
        frames.append(out)
    return pl.concat(frames)


def fetch_expected_stats(cfg: Config, force: bool = False) -> pl.DataFrame:
    """Public seasonal expected stats for the player-level replication check.
    minPA=25 (well under the 100-PA evaluation gate). Returns: player_id, season,
    pa, est_woba."""
    frames = []
    for year in cfg.all_seasons:
        cache = cfg.raw_dir / f"expected_stats-{year}.parquet"
        if cache.exists() and not force:
            frames.append(pl.read_parquet(cache))
            continue
        from pybaseball import statcast_batter_expected_stats

        pdf = _retry(lambda: statcast_batter_expected_stats(year, 25))
        df = pl.from_pandas(pdf)
        out = df.select(
            player_id=pl.col("player_id").cast(pl.Int64),
            pa=pl.col("pa").cast(pl.Int64),
            est_woba=pl.col("est_woba").cast(pl.Float64),
        ).with_columns(season=pl.lit(year, dtype=pl.Int64))
        out.write_parquet(cache)
        frames.append(out)
    return pl.concat(frames)
