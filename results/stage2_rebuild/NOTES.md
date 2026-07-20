# Stage 2 — teaching the caches where the ball went

Date: 2026-07-20 · No BART re-fit. Reproduce:
`.venv/bin/python scripts/rebuild_caches.py` then `.venv/bin/python scripts/qc_spray.py`
→ `results/stage2_rebuild/{rebuild_report.json, spray_qc.json, figures/}`.

**Goal.** Add `hc_x`, `hc_y` and `stand` to the slim Statcast caches and derive a
**pull-relative spray angle**, so the Stage-3 surface can finally see *direction* and not
just how hard and how high the ball left the bat. Two things had to be proved before the
27-minute refit was worth starting: that the rebuild changed nothing that already existed,
and that the pull mirror is not backwards.

## Why a rebuild needs a reproduction gate at all

The obvious worry — "adding columns might corrupt the data" — is the one that *cannot*
happen: `build_season_caches` ends in `.select(KEEP_COLUMNS)`, and appending names to a
select list provably cannot change a row.

The real risk is invisible. `build_season_caches(force=True)` re-reads the **upstream
monthly cache**, which KIT may have updated since these caches were built in July. A
silent upstream revision would move 2,636 player-seasons, r 0.4886/0.4669 and the ELPD
anchor of −80107 with **no code change at all** — and every frozen number in this repo
would quietly become incomparable. Gates R1/R2 buy certainty about that for about a minute
of compute, which is why the destructive step backs up to `data/raw/prerebuild/` first and
never auto-restores: when a gate fails, the rebuilt files are the evidence.

## R1–R6 — the rebuild changed nothing (all PASS)

| gate | criterion | result |
|---|---|---|
| **R1** | per-season row counts unchanged | 686,248 / 713,552 / 709,231 / 709,727 — identical |
| **R2** | order-independent digest over the pre-existing 15 columns | `705b37fc…` / `0dbab674…` / `059257d4…` / `3ab16780…` — **identical** |
| **R3** | new columns land | `stand` 100% non-null on BBE; hc null 0.036–0.045% |
| **R4** | Phase-1 talent re-derived from the new caches | 2,636 rows, `max|Δ xwoba_talent|` = **0.000e+00** |
| **R5** | `filter_bbe` counts | 118,891 / 122,070 / 122,634 / 122,006 — exact |
| **R6** | Level-2 re-run against the rebuilt caches | r PA≥30 **0.469817**, PA≥100 **0.490783** — to the sixth decimal |

R2 is the load-bearing one: the digests came back **byte-identical on all four seasons**,
so the upstream cache did not move and every frozen anchor in the repo survives the
rebuild untouched. R5 matters because 118,891 + 122,070 + 122,634 = **363,595** train and
2025 = **122,006** holdout are exactly v0's `predict_rows` — the ELPD anchor stays
comparable. R4/R6 confirm it downstream: the Phase-1 talent table reproduces to *zero*
difference, not merely to tolerance.

Level 2 was re-run here, in Stage 2, rather than in Stage 3. `run_talent2.py` consumes the
caches' **public** `estimated_woba_using_speedangle` and never touches the BART surface, so
the Stage-3 refit provably cannot move its numbers — but the cache rebuild could, so the
re-run belongs to the rebuild's gate.

## The spray transform, in words

`φ_raw = atan2(hc_x − 125.42, 198.27 − hc_y)`, in degrees, is the **raw** direction off
home plate: **negative is left field, positive is right field**. (`hc_y` *decreases* going
out toward the outfield, hence the `198.27 − hc_y` term rather than the other order.)

A right-handed batter **pulls to left field**, which is the negative side; a left-handed
batter pulls to right field, the positive side. So the pull-relative angle negates `φ_raw`
for `stand == "R"` and leaves `stand == "L"` alone:

```
spray_pull = −φ_raw  if stand == "R"       POSITIVE always means PULLED,
spray_pull = +φ_raw  if stand == "L"       for both hands.
```

`stand` also enters the model separately as a numeric `stand_R` (1.0/0.0), because the two
effects are different: **park asymmetries act on raw direction** (the Green Monster is in
left field regardless of who is batting) while **batter skill acts pull-relative**. Keeping
both lets BART recover either.

## S1–S6 — the mirror is not backwards (all PASS)

| season | mean `spray_pull` L / R | HR mean L / R | HR frac pull-side L / R | hc imputed | \|φ_raw\| > 45° |
|---|---|---|---|---|---|
| 2022 | +6.84 / +3.23 | +19.3 / +17.5 | 0.819 / 0.789 | 0.0345% | 9.22% |
| 2023 | +7.36 / +3.62 | +19.2 / +16.3 | 0.808 / 0.779 | 0.0434% | 9.08% |
| 2024 | +7.15 / +3.55 | +20.0 / +18.0 | 0.827 / 0.805 | 0.0375% | 8.98% |
| 2025 | +7.50 / +3.56 | +20.3 / +17.5 | 0.836 / 0.799 | 0.0361% | 9.09% |

2024 named anchors (≥250 BBE from that side): Kyle Schwarber (L) **+13.90**, Isaac Paredes
(R) **+11.88**, Ryan Mountcastle (R) **−5.16** — a genuine opposite-field hitter, and the
gate requires him to stay negative.

**Why six gates and not one "average pull is positive" check.** Enumerate the mirror bugs:
mirror *neither* hand → R reads −3.55 (S1 fails on R); mirror *both* → L reads −7.15 (S1
fails on L); mirror the *wrong* hand → both negative; swap the `atan2` arguments →
`atan2(Δy, Δx) = 90° − atan2(Δx, Δy)`, so RHB mean pull becomes ≈ −93.5° while LHB reads
≈ +82.9° and would sail through a one-hand check. That last case is exactly why S1 demands
**both** hands.

**S5 is the non-obvious one.** The bug it exists to catch is resolving `stand` per
*player-season* (modal hand) instead of per event. That flips the sign only on the
minority-hand events of the 65 switch-hitter batter-seasons, which moves league means by
well under 0.1° — S1, S2 and S3 all still pass, and none of the three named anchors is a
switch hitter, so S4 passes too. Splitting the switch hitters' own rows by `stand` is what
makes the flip visible. Measured by simulating the bug on 2024 data:

| | league mean L / R | HR frac L / R | **switch-only mean L / R** |
|---|---|---|---|
| correct per-event mirror | +7.146 / +3.549 | .827 / .805 | **+8.68 / +5.60** |
| modal-hand mirror (the bug) | +7.118 / +2.991 | .827 / .765 | **+8.52 / −3.97** |
| gate outcome under the bug | S1 PASS | S3 PASS | **S5 FAIL** |

The shipped run reports +8.68 (L, n=9,140) and +5.60 (R, n=4,136) — the correct row, and
decisively far from the bug's −3.97 on 4,136 rows. The tempting alternative gate, "count
batter-seasons with two `stand` values", is **vacuous**: that count is a property of the
input data that no bug in `add_spray` can change, and R3 already guarantees it.

## Two things a future reader should not "fix"

**~9% of BBE have |φ_raw| > 45°, and that is correct.** Caught fouls and pop-ups land
outside the foul lines in hc coordinates. The rate is stable at 8.98–9.22% across all four
seasons. **Do not clamp it** — BART splits on the tail happily, and clamping would erase a
real distinction between a foul pop-up and a ball hit down the line.

**hc missingness is imputed and flagged, never dropped.** Missingness on BBE is
0.0345–0.0434% (≈40–55 rows a season) and is not structural by launch angle. Rows are
imputed by an `(EV bin, LA bin, stand)` median → per-hand median → 0.0 ladder, with the
imputation table built on the **training seasons only** so the holdout never imputes
itself. Dropping them would be the tidier-looking choice and it would be **wrong**: it
moves the holdout event count off 122,006, and 122,006 is what the −80107 ELPD anchor is
defined over. A changed denominator makes the comparison incomparable, not merely worse.

`hc_imputed` is carried as an **audit column, deliberately not a BART feature**. At ~45
rows a season a missing-flag feature is pure split noise, and adding it would make this a
six-feature model when the design specifies five.

## Figures

- `figures/spray_by_hand.png` — 2024 pull-relative spray density by hand. Both curves must
  lean right of zero (LHB peak ≈ +30°, RHB ≈ +20°). This is the S1 gate, drawn.
- `figures/spray_hr_raw_direction.png` — 2024 home runs in **raw** direction. The two
  curves peak on **opposite** sides of zero (RHB ≈ −35°, LHB ≈ +35°). If they ever peak on
  the same side, `stand` is not entering the mirror and S1 is passing by coincidence.

## Carried into Stage 3

`stand` is genuinely per-event: 65 of 647 batter-seasons with BBE in 2024 carry both
values (41 with ≥25 BBE from each side). It must never be joined per player-season. The
Stage-3 feature list is `[launch_speed, launch_angle, spray_pull, stand_R, sprint_speed]`,
and the same imputation-table-from-train discipline applies inside `run_v0.py --variant
spray`.
