"""Typed config loading. STATCAST_PATH env (via .env) overrides the config fallback path."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv


@dataclass(frozen=True)
class StageConfig:
    name: str
    subsample: int | None      # None -> all training rows
    m_trees: int
    tune: int
    draws: int
    chains: int
    store_p: bool
    predict_cap: int | None    # None -> predict on all rows


@dataclass(frozen=True)
class Config:
    seed: int
    train_seasons: list[int]
    holdout_season: int
    statcast_dir: Path
    raw_dir: Path
    results_dir: Path
    season_windows: dict[int, tuple[str, str]]
    sprint_min_opp: int
    stages: dict[str, StageConfig]
    thin_draws: int
    chunk_size: int
    min_pa: int
    reliability_bins: int
    sprint_grid: tuple[float, float, int]

    @property
    def all_seasons(self) -> list[int]:
        return [*self.train_seasons, self.holdout_season]


def load_config(path: str | Path = "config.yaml") -> Config:
    # Resolve .env relative to the run directory (usecwd=True): production runs from
    # the repo root and picks up ./.env; tests that chdir into a tmp dir stay hermetic
    # (no repo .env is discovered, so the config fallback path is exercised).
    load_dotenv(find_dotenv(usecwd=True))
    raw = yaml.safe_load(Path(path).read_text())
    stages = {name: StageConfig(name=name, **vals) for name, vals in raw["model"]["stages"].items()}
    return Config(
        seed=raw["seed"],
        train_seasons=list(raw["seasons"]["train"]),
        holdout_season=raw["seasons"]["holdout"],
        statcast_dir=Path(os.environ.get("STATCAST_PATH", raw["paths"]["statcast_dir"])),
        raw_dir=Path(raw["paths"]["raw_dir"]),
        results_dir=Path(raw["paths"]["results_dir"]),
        season_windows={int(k): tuple(v) for k, v in raw["season_windows"].items()},
        sprint_min_opp=raw["sprint_speed"]["min_opp"],
        stages=stages,
        thin_draws=raw["rollup"]["thin_draws"],
        chunk_size=raw["rollup"]["chunk_size"],
        min_pa=raw["evaluate"]["min_pa"],
        reliability_bins=raw["evaluate"]["reliability_bins"],
        sprint_grid=tuple(raw["evaluate"]["sprint_grid"]),
    )
