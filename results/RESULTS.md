# xwobart v0 results

<!-- stage_A -->
## Stage A
- kit_sha: 648b990 | seed: 42 | fit rows: 5000 | predict rows: {'train': 20000, 'holdout': 20000, 'capped': True}
- coverage gaps: ['2022: cache ends 2022-09-30, season ends 2022-10-05', '2023: cache ends 2023-09-30, season ends 2023-10-01']
- fit runtime: 0.4 min | total: 0.6 min
- linear weights: {'out': 0.0159, 'single': 0.9, 'double': 1.25, 'triple': 1.6, 'home_run': 2.0}
- volumes/drops per season: train [{'game_year': 2022, 'n_bbe_raw': 120450, 'n_bunt': 1047, 'n_missing_ls_la': 512, 'pct_missing': 0.43}, {'game_year': 2023, 'n_bbe_raw': 123499, 'n_bunt': 1072, 'n_missing_ls_la': 357, 'pct_missing': 0.29}, {'game_year': 2024, 'n_bbe_raw': 124160, 'n_bunt': 1156, 'n_missing_ls_la': 370, 'pct_missing': 0.3}]; holdout [{'game_year': 2025, 'n_bbe_raw': 124789, 'n_bunt': 1227, 'n_missing_ls_la': 1556, 'pct_missing': 1.26}]
- sprint imputation rates: train [{'game_year': 2022, 'imputation_rate': 0.006308299198425449}, {'game_year': 2023, 'imputation_rate': 0.0046202998279675596}, {'game_year': 2024, 'imputation_rate': 0.00581404830634245}]; holdout [{'game_year': 2025, 'imputation_rate': 0.00581118961362556}]
- replication r — event train 0.845, event holdout 0.840, player train 0.453, player holdout 0.455
- calibration — weighted ECE 0.0801
- ELPD (lppd) -14401.3 ± 100.6 over 20000 events
- undercorrection corr — model 0.028 vs public -0.002
- localization slopes (per ft/s) — grounder 0.0010, barrel -0.0039
- sanity warnings: ['max R-hat on probed mu cells = 1.828 (> 1.1)']
<!-- /stage_A -->

<!-- stage_B -->
## Stage B
- kit_sha: 648b990 | seed: 42 | fit rows: 50000 | predict rows: {'train': 363595, 'holdout': 122006, 'capped': False}
- coverage gaps: ['2022: cache ends 2022-09-30, season ends 2022-10-05', '2023: cache ends 2023-09-30, season ends 2023-10-01']
- fit runtime: 14.2 min | total: 17.1 min
- linear weights: {'out': 0.0159, 'single': 0.9, 'double': 1.25, 'triple': 1.6, 'home_run': 2.0}
- volumes/drops per season: train [{'game_year': 2022, 'n_bbe_raw': 120450, 'n_bunt': 1047, 'n_missing_ls_la': 512, 'pct_missing': 0.43}, {'game_year': 2023, 'n_bbe_raw': 123499, 'n_bunt': 1072, 'n_missing_ls_la': 357, 'pct_missing': 0.29}, {'game_year': 2024, 'n_bbe_raw': 124160, 'n_bunt': 1156, 'n_missing_ls_la': 370, 'pct_missing': 0.3}]; holdout [{'game_year': 2025, 'n_bbe_raw': 124789, 'n_bunt': 1227, 'n_missing_ls_la': 1556, 'pct_missing': 1.26}]
- sprint imputation rates: train [{'game_year': 2022, 'imputation_rate': 0.006308299198425449}, {'game_year': 2023, 'imputation_rate': 0.0046202998279675596}, {'game_year': 2024, 'imputation_rate': 0.00581404830634245}]; holdout [{'game_year': 2025, 'imputation_rate': 0.00581118961362556}]
- replication r — event train 0.915, event holdout 0.916, player train 0.960, player holdout 0.963
- calibration — weighted ECE 0.0376
- ELPD (lppd) -80023.5 ± 246.5 over 122006 events
- undercorrection corr — model 0.033 vs public 0.013
- localization slopes (per ft/s) — grounder 0.0023, barrel -0.0034
- sanity warnings: ['max R-hat on probed mu cells = 1.666 (> 1.1)']
<!-- /stage_B -->
