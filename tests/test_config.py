import textwrap

from src.config import load_config


def _write_cfg(tmp_path, statcast_dir="/fallback/path"):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(f"""
        seed: 7
        seasons: {{train: [2022, 2023, 2024], holdout: 2025}}
        paths: {{statcast_dir: {statcast_dir}, raw_dir: data/raw, results_dir: results}}
        season_windows:
          2022: ["2022-04-07", "2022-10-05"]
          2023: ["2023-03-30", "2023-10-01"]
          2024: ["2024-03-20", "2024-09-30"]
          2025: ["2025-03-18", "2025-09-28"]
        sprint_speed: {{min_opp: 10}}
        model:
          stages:
            A: {{subsample: 5000, m_trees: 20, tune: 200, draws: 200, chains: 2, store_p: true, predict_cap: 20000}}
            C: {{subsample: null, m_trees: 50, tune: 500, draws: 500, chains: 2, store_p: false, predict_cap: null}}
        rollup: {{thin_draws: 500, chunk_size: 20000}}
        evaluate: {{min_pa: 100, reliability_bins: 10, sprint_grid: [23.0, 31.0, 17]}}
    """))
    return p


def test_loads_fields(tmp_path, monkeypatch):
    monkeypatch.delenv("STATCAST_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = load_config(_write_cfg(tmp_path))
    assert cfg.seed == 7
    assert cfg.train_seasons == [2022, 2023, 2024]
    assert cfg.holdout_season == 2025
    assert cfg.all_seasons == [2022, 2023, 2024, 2025]
    assert str(cfg.statcast_dir) == "/fallback/path"
    assert cfg.stages["A"].m_trees == 20 and cfg.stages["A"].store_p is True
    assert cfg.stages["C"].subsample is None and cfg.stages["C"].predict_cap is None
    assert cfg.season_windows[2022] == ("2022-04-07", "2022-10-05")
    assert cfg.sprint_grid == (23.0, 31.0, 17)


def test_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("STATCAST_PATH", "/from/env")
    cfg = load_config(_write_cfg(tmp_path))
    assert str(cfg.statcast_dir) == "/from/env"
