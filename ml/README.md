# Learned MODIS → VIIRS conversion (experimental)

A convolutional network that predicts VIIRS S-NPP-equivalent hotspot counts per 0.25° cell and
month from MODIS alone. It is scored against Brasa's hierarchical-ratio model (`pipeline/harmonize.py`)
and the NOAA-20 ceiling on the same hold-out. **Status: trained (`model/unet.pt`) and evaluated; the
U-Net + MLP ensemble beats Brasa M3 on every test year (results below).**

## Results (hold-out 2019–2024, `ml/compare.py`)

Median |log error| of monthly totals, and how often the sign of a strong anomaly is right:

| | 1° | 5° | country | anomaly sign, 5° | 2025, 5° |
|---|---|---|---|---|---|
| Brasa M3 (refit 2012–2018) | 0.304 | 0.246 | 0.199 | 0.768 | 0.300 |
| per-pixel MLP (`context_baseline.py`) | 0.291 | 0.245 | 0.208 | 0.851 | 0.279 |
| U-Net (`model/unet.pt`) | 0.266 | 0.211 | 0.184 | 0.863 | 0.253 |
| **mean of U-Net and MLP** | **0.261** | **0.203** | **0.175** | **0.867** | **0.243** |
| NOAA-20 vs S-NPP (ceiling) | 0.163 | 0.105 | 0.087 | 0.925 | 0.099 |

- Paired bootstrap over months at 5°: ensemble − Brasa −0.030 [−0.035, −0.027]; ensemble − U-Net
  −0.014 [−0.017, −0.011]; MLP − Brasa +0.003 [−0.002, +0.006]. The 50/50 weight was fixed in
  advance, not tuned on the test years.
- The MLP sees the same channels plus 1/3/9/25-cell neighbourhood sums but no learned convolutions.
  It matches the U-Net on anomaly sign but not on regional level, so the learned spatial context is
  what buys the 5° and country gain. They err in different places (the U-Net under-predicts small
  fires, −10.5 % bias in cells with < 1 MODIS det/yr; the MLP is near unbiased), so the mean wins.
- Permutation importance (U-Net, 5° error increase): MODIS count +1.18, FRP total/max +0.83, MODIS
  climatology +0.21, fire days +0.08; neighbouring months only +0.01.
- Weak spots: **Brazil** (country error 0.078 vs 0.054 for Brasa M3) and cells where MODIS saw
  nothing (−12 % bias). Keep Brasa M3 there, and for the Terra-only months before July 2002.
- Run it: `ml/colab_run_model.ipynb` (or `ml/predict_ensemble.py`) writes VIIRS-equivalent counts per
  cell-month for 2002-07 → 2025. `ml/colab_experiment.ipynb` reproduces the comparison.

## What the overlap period shows (2012–2025, `features.parquet`)

| | |
|---|---|
| Overlap years | 2012–2025 (S-NPP); NOAA-20 from 2018 |
| Good detections | MODIS 102.7 M · S-NPP 231.3 M · NOAA-20 127.6 M (2000–2025 totals in the table) |
| Global V/M per year | 3.78 – 4.25 for 2012–2024, **4.47 in 2025** (drifting up as Terra/Aqua age and drift) |
| VIIRS in cell-months where MODIS saw nothing | 4.2–6.3 % of all S-NPP detections, spread over 3.2 M cell-months |
| V/M by cell fire regime | 7.3× in cells with < 1 MODIS det/yr, 3.8× in cells with ≥ 100/yr |
| V/M by night share | 3.7× (day-only) → 4.6× (≥ 50 % night) |
| V/M by Terra share | ~3.9× when Aqua dominates, 3.3× when ≥ 80 % Terra |
| V/M by burstiness | 3.7–4.0× normally, 2.8× when the whole month is one MODIS day |
| Spread at fixed MODIS count (m_n = 5) | S-NPP quartiles 8 / 15 / 26, mean 19.5, variance 380: strongly overdispersed |
| Cell-month log correlation | MODIS vs S-NPP 0.84; **NOAA-20 vs S-NPP 0.90 (ceiling)** |

So the ratio is stable on average, but it is conditional on fire regime, overpass mix and timing,
and the per-cell scatter is large. At cell-month level the headroom is small: MODIS gets 0.84
log correlation against a 0.90 instrument ceiling. At 5° and country scales Brasa sits well
above the ceiling (error 0.26 vs 0.11 in the main README), and that is the gap a learned model
can try to close.

## Design

```
points/*.parquet ──features.py──▶ ml/features.parquet ──build_tensor.py──▶ ml/<name>/{X,Y,N20,S,CL}.npy
                                                                              │
                                              train.py (U-Net, NB loss) ◀─────┘──▶ evaluate.py
```

- **Inputs per cell-month: 15 MODIS channels.** Counts with and without low-confidence
  detections, Terra and night counts, a 4-bin FRP histogram, total and max FRP, fire-days and
  peak-day count, mean pixel footprint, confidence and brightness. Each month also gets the same
  channels for the previous and next month, plus static maps: latitude, longitude, and the MODIS
  climatology for the calendar month and the year. Climatologies use **MODIS only**, so nothing
  about the target leaks in and the model can run before 2012. Filters are identical to
  `pipeline/02_aggregate.py`.
- **Model.** A 3-level U-Net (1.8 M parameters at width 32; receptive field about 17°) predicts a
  correction to the global-ratio baseline: `log μ = log(4.1·M + 0.05) + f(x)`. It starts as
  Brasa's M0 and learns only the departures, so `exp(f)` is a mappable conversion-factor field.
  The spatial context lets it learn the small-fire floor (VIIRS fire where MODIS saw none)
  rather than having it imposed.
- **Loss.** Negative-binomial NLL per pixel, with learned per-pixel overdispersion. This gives
  calibrated uncertainty, which the app's p10–p90 band can use. On top of that is a smooth-L1 on
  log totals over 2° blocks, because regional totals are what the calendar shows.
- **Split.** Matches `pipeline/03_validate.py`: train 2012–2017, early-stop on 2018, test
  2019–2024. Outage months in `docs/validation/model.json` are excluded everywhere.
- **Evaluation.** The same metrics as the main README at 1°, 5° and country scales. The CNN is
  scored alongside Brasa M3 refit on the identical grid and years, and alongside the NOAA-20
  ceiling. `--importance` does permutation importance by input group (swap with another month,
  or spatially shift static maps), which shows which signals the network relies on beyond the
  raw count.

## Run (GPU machine)

```bash
.venv/bin/pip install -r ml/requirements.txt          # pick the CUDA torch wheel for your GPU
.venv/bin/python -I ml/features.py     ~/firms/derived                       # ~3 min on 4 cores
.venv/bin/python -I ml/build_tensor.py ~/firms/derived --years 2012-2025     # ~3.7 GB memmaps
.venv/bin/python -I ml/train.py ~/firms/derived/ml/overlap ~/firms/derived/ml/unet.pt
.venv/bin/python -I ml/evaluate.py ~/firms/derived ~/firms/derived/ml/overlap ~/firms/derived/ml/unet.pt --importance
```

Smoke test (CPU, seconds, 8°×8° Cerrado window):

```bash
.venv/bin/python -I ml/build_tensor.py ~/firms/derived --years 2016-2020 --bbox=-50,-16,-42,-8 --name smoke
.venv/bin/python -I ml/train.py ~/firms/derived/ml/smoke /tmp/smoke.pt --train 2016-2017 --val 2018 --smoke
.venv/bin/python -I ml/evaluate.py ~/firms/derived ~/firms/derived/ml/smoke /tmp/smoke.pt --train 2016-2018 --test 2019-2020 --importance --imp-months 6
```

## Residual check: is there signal left? (`residual_check.py`, no training, ~25 s)

```bash
.venv/bin/python -I ml/residual_check.py ~/firms/derived
```

Brasa M3 is fit on 2012–2018. On held-out 2022–2024 cell-months where MODIS saw fire:

- **Residual variance is 0.239.** About 0.050 of that is instrument noise, measured as the
  NOAA-20 vs S-NPP disagreement.
- **The per-cell MODIS descriptors explain very little of it.** The best single one (fire days)
  explains 2.7%, and the top three together 3.5%. Bin means come from 2019–2021 and are scored
  on 2022–2024.
- **A lookup correction built from them makes regional error worse.** At 5° it goes from 0.305
  to 0.314; for countries, from 0.255 to 0.262.

So nearly all of Brasa's remaining error is invisible to the per-cell MODIS record. The U-Net's
only remaining edge is spatial and temporal context. Expect small gains at best, and extend this
check with neighbourhood sums before spending GPU time.

## Known limits / next steps

- **Terra-only era (2000-11 → 2002-06).** Not handled. The model is trained on Terra+Aqua
  inputs, so keep Brasa's Terra-only model for those months, or build a Terra-only feature
  variant and train a second network.
- **Sensor drift.** V/M rises in 2025. A model frozen on 2012–2018 may under-predict late
  years. Check the per-year bias in the evaluation before trusting 2023–2025.
- **Strong tabular baseline.** Worth adding gradient-boosted trees on the same features plus
  neighbourhood sums, with SHAP, as a second learned model. If it matches the CNN, prefer it.
- **Hyperparameters.** Width, patch size, `--reg-w` and epochs are untuned defaults.
