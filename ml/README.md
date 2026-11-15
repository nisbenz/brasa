# Learned MODIS → VIIRS conversion (experimental)

A convolutional network that predicts VIIRS S-NPP-equivalent hotspot counts per 0.25° cell and
month from MODIS alone. It is scored against Brasa's hierarchical-ratio model (`pipeline/harmonize.py`)
and the NOAA-20 ceiling on the same hold-out. **Status: trained (`model/unet.pt`, epoch 19). This folder is the code as of commit
`117d824`, plus the U-Net + MLP ensemble from `4d18e46` and the VIIRS-schema output layer below.**

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

## Ensemble for the app (`context_baseline.py`, `predict_ensemble.py`)

The app's MODIS-era months use the mean of the U-Net and a per-pixel MLP (restored from commit
`4d18e46`). The MLP has the same residual form as the U-Net, `log μ = log(4.1·M + 0.05) + g(x)`,
but `g` is a 2×128 MLP on 33 hand-built features (`core.context_features`): MODIS counts summed over
1/3/9/25-cell boxes for the previous, current and next month, the other current-month channels,
static maps, the calendar climatology and the season. On 2019–2024 the mean scores 0.203 median
|log error| at 5°, against 0.211 for the U-Net alone and 0.246 for Brasa M3.

```
stack (2012-2025) ──context_baseline.py──▶ ctx_models.pt ─┐
stack (2002-2025) ─────────────────────────────────────────┼─predict_ensemble.py --app─▶ app_nn.parquet ─▶ pipeline/08_merge_unet.py
model/unet.pt ─────────────────────────────────────────────┘
```

```bash
# MLP: train 2012-2017, early-stop on 2018. Holds ~6 GB of samples in memory: use a GPU box or Colab
.venv/bin/python -I ml/context_baseline.py ~/firms/derived/ml/overlap ~/firms/derived/ml --no-predict
# stack over every Terra+Aqua month (MODIS only, no VIIRS needed), then the ensemble
.venv/bin/python -I ml/build_tensor.py ~/firms/derived --years 2002-2025 --name full
.venv/bin/python -I ml/predict_ensemble.py ~/firms/derived/ml/full model/unet.pt ~/firms/derived/ml/ctx_models.pt \
    ~/firms/derived/ml/ensemble --countries ~/firms/derived/app/cell_country.parquet --app ~/firms/derived/ml/app_nn.parquet
```

`predict_ensemble.py` starts at 2002-07 by default: before that only Terra flew, the models were
trained on Terra + Aqua, and `08_merge_unet.py` keeps Brasa M3 for those months. Besides
`app_nn.parquet` (cell, year, month, pred) it writes `cells_<year>.parquet` with both members
(`unet`, `mlp`, `pred`) and `monthly_totals.csv` (MODIS, the three predictions and observed S-NPP,
globally and per country). Only cells with a prediction above 0.001 are kept.
`ctx_models.pt` is not in the repo; the one used for the app came from `ml/colab_run_model.ipynb`
at commit `4d18e46`.

Smoke test (CPU, seconds, the Cerrado stack from the smoke test above; the MLP trained on 2016–2017
here is only good for checking the plumbing):

```bash
.venv/bin/python -I ml/context_baseline.py ~/firms/derived/ml/smoke /tmp/ctx --no-predict
.venv/bin/python -I ml/predict_ensemble.py ~/firms/derived/ml/smoke model/unet.pt /tmp/ctx/ctx_models.pt /tmp/ens \
    --start 2020-08 --end 2020-08 --app /tmp/app_nn.parquet
```

The tile halo in `core.predict_month` is 48 cells (it was 16 at `117d824`): the U-Net's receptive
radius is about 35 cells, and at 16 the tile seams shifted global totals by 1.8 %. This applies to
`evaluate.py` and `to_viirs.py` too.

## Output layer: synthetic VIIRS detections (`viirs_calibrate.py`, `to_viirs.py`)

The U-Net gives an expected S-NPP count per cell-month. `to_viirs.py` turns that, plus the MODIS
detections it was computed from, into point records in the FIRMS VIIRS schema (`latitude,
longitude, bright_ti4, scan, track, acq_date, acq_time, satellite, instrument, confidence,
version, bright_ti5, frp, daynight, type`), laid out like the FIRMS country-yearly archive.

```
MODIS points ─┐                        ┌─ count per cell: floor(mu)+Bernoulli, or NB(mu, alpha)
              ├─▶ U-Net mu, alpha ─────┤
stack (X,S,CL)┘                        └─ attributes: copy a MODIS template from the cell-month,
                                          quantile-map frp / brightness / scan / confidence to
viirs_calibration.json (S-NPP 2020) ─────▶ S-NPP, draw ti5 | ti4 and the local overpass hour
```

```bash
.venv/bin/python -I ml/viirs_calibrate.py ~/firms/derived ~/firms --year 2020
.venv/bin/python -I ml/to_viirs.py ~/firms/derived ~/firms/derived/ml/overlap model/unet.pt --years 2019
# pre-VIIRS years need a stack for them first (inputs are MODIS only):
.venv/bin/python -I ml/build_tensor.py ~/firms/derived --years 2003-2011 --name pre
.venv/bin/python -I ml/to_viirs.py ~/firms/derived ~/firms/derived/ml/pre model/unet.pt --years 2005
```

Output goes to `<derived>/ml/synthetic/viirs-snpp-unet/<year>/` as one CSV per country (or one
Parquet per month with `--format parquet`), plus `cells_<year>.parquet` (MODIS count, mu, alpha
and the drawn count per cell-month) and `manifest.json`. Every row has `version = unet-v1`.

What the synthetic points are and are not:
- **Counts are the model's.** Every point stays inside its cell, so re-aggregating the output to
  0.25° gives back the drawn counts exactly (up to acq_date moving a day across a month edge
  after the UTC conversion).
- **Attributes are plausible, not observed.** They follow S-NPP's 2020 distributions and keep the
  MODIS template's rank, so the hottest MODIS fires become the hottest VIIRS ones. Do not read
  per-point FRP or brightness as measurements. Sub-cell positions are MODIS footprint jitter.
- Only good detections are produced (`type 0`, confidence `n`/`h`), because those are what the
  model predicts. Cells outside every country in `cell_country.parquet` are skipped.

## Known limits / next steps

- **Terra-only era (2000-11 → 2002-06).** Not handled. The model is trained on Terra+Aqua
  inputs, so keep Brasa's Terra-only model for those months, or build a Terra-only feature
  variant and train a second network.
- **Sensor drift.** V/M rises in 2025. A model frozen on 2012–2018 may under-predict late
  years. Check the per-year bias in the evaluation before trusting 2023–2025.
- **Strong tabular baseline.** Worth adding gradient-boosted trees on the same features plus
  neighbourhood sums, with SHAP, as a second learned model. If it matches the CNN, prefer it.
- **Hyperparameters.** Width, patch size, `--reg-w` and epochs are untuned defaults.
