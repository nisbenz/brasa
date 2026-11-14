# Learned MODIS → VIIRS conversion

Neural networks that predict **VIIRS S-NPP-equivalent hotspot counts per 0.25° cell and month from
MODIS alone**, trained on the 2012–2025 overlap and applied to 2000–2011. They are scored against
Brasa's hierarchical-ratio model (`pipeline/harmonize.py`, "Brasa M3") and against NOAA-20, a second
VIIRS instrument that acts as the noise ceiling.

Two model generations exist:

| | old (`model/unet.pt`) | new / v2 (`model/unet_v2.pt`, trained with `run_colab.ipynb`) |
|---|---|---|
| inputs | 15 MODIS channels, Terra + Aqua combined | 30 channels: the same descriptors **per satellite**, plus local overpass time and an Aqua flag |
| Terra-only months (2000-11 … 2002-06) | not covered | covered: trained with random Aqua dropout |
| patch sampling | ∝ √(VIIRS count), i.e. chosen by the target | ∝ √(MODIS count): chosen by the input, so E[y \| x] is not biased |
| normalisation | GroupNorm over the tile (output depends on tile size / padding) | per-pixel LayerNorm (tiled inference = patch training) |
| regional loss | 2° blocks | 1°, 2° and 5° blocks |
| model selection | NB loss on one year (2018) | 5° error of full-map predictions; blocked year-fold CV ensemble |
| file | one network | one `.pt` holding every ensemble member |

Old model on 2019–2024 (median |log error| of monthly totals; lower is better): 1° 0.266, 5° 0.211,
country 0.184, against Brasa M3's 0.304 / 0.246 / 0.199 and the NOAA-20 ceiling's 0.163 / 0.105 / 0.087.
The v2 numbers come from `run_evaluate.py` once v2 has been trained on a GPU (see *Status*).

## Files

One naming scheme: `<role>_<subject>`. `lib_*` modules are imported, never run; `run_*` scripts are the
pipeline steps in order; `test_*` holds the tests.

| file | role |
|---|---|
| `lib_config.py` | Constants (grid, detection filters, baseline K0/B0), the **channel catalogue** and presets, year parsing, and the typed configs stored in checkpoints: `InputSpec` (which channels and in what layout), `ModelConfig` (architecture), `TrainConfig` (training run). |
| `lib_stack.py` | `Stack`: read-only memmap view of a tensor stack. Builds model inputs **by channel name** from an `InputSpec`, which is why one stack serves both models. Supports Aqua removal (`aqua_off`), and channel swaps and static shifts for permutation tests. |
| `lib_model.py` | `UNet` (depth, width and norm configurable; the legacy architecture is depth 3 + GroupNorm), NB likelihood, multi-scale regional loss, `tiled_forward` (full-map inference), `Predictor` (loads **both** checkpoint formats and averages ensemble members), `MonthPrediction` (per-member μ, α; draws from the predictive mixture). |
| `lib_metrics.py` | Dense scoring: `Regions` (1°, 5°, country ids), `score()` (identical to `pipeline/03_validate.py`, plus anomaly slope), paired month-bootstrap, gap-closed share, randomized PIT, Brasa M3 refit as maps (`BrasaMaps`). |
| `run_clean.py` | **Step 1, data cleaning.** FIRMS points → `features.parquet`, one row per cell-month: legacy `m_*` columns (bit-identical to the old table), per-satellite `t_*` / `a_*`, and the targets `v_n` (S-NPP) and `n20_n` (NOAA-20). |
| `run_tensor.py` | **Step 2, tensor creation.** `features.parquet` → stack `X/Y/N20/S/CL.npy` + `meta.json`. `--preset legacy\|v2\|all`, `--bbox`, `--aux` for external covariates. |
| `run_train.py` | **Step 3, training.** One member, a seed ensemble, or blocked year-fold CV (`--folds`); fixed-epoch refits (`--val none`). Every phase-1/3/4 option can be toggled for ablations. |
| `run_evaluate.py` | **Step 4, testing and evaluation.** Scores any number of `.pt` files (old and new) plus Brasa M3 and the ceiling on the same months, and writes `report.md` plus CSVs (see below). |
| `run_predict.py` | **Step 5, prediction.** Any months, including pre-VIIRS → `cells_<year>.parquet` (optional p10/p90), `monthly_totals.csv`, and `app_nn.parquet` for `pipeline/08_merge_unet.py`. |
| `run_colab.ipynb` | The full GPU workflow on Colab: stacks → ablations → CV ensemble → old-vs-new report → backcast → production refit → prediction → app merge. |
| `test_lib.py` | Unit tests (stdlib `unittest`): input layout, Aqua dropout, NB loss vs `torch.distributions`, tile independence, legacy key remap, checkpoint round-trip, `score()` vs the original pandas code, PIT calibration. |
| `requirements.txt` | ML extras on top of `../requirements.txt`. |

```
points/*.parquet ─run_clean─▶ ml/features.parquet ─run_tensor─▶ ml/<stack>/{X,Y,N20,S,CL}.npy, meta.json
                                                                   │
                         run_train ─▶ model.pt ─┬─▶ run_evaluate ─▶ ml/eval/report.md, *.csv
                                                └─▶ run_predict  ─▶ cells_<year>.parquet, app_nn.parquet
                                                                     └─▶ pipeline/08_merge_unet.py
```

## Run

```bash
.venv/bin/pip install -r ml/requirements.txt                       # CUDA machine: matching torch wheel
.venv/bin/python -I ml/test_lib.py                                  # seconds
.venv/bin/python -I ml/run_clean.py  ~/firms/derived                # ~8 min on 4 cores, 200 MB
.venv/bin/python -I ml/run_tensor.py ~/firms/derived --years 2011-2025 --preset v2  --name overlap
.venv/bin/python -I ml/run_tensor.py ~/firms/derived --years 2018-2025 --preset all --name eval
.venv/bin/python -I ml/run_train.py  ~/firms/derived/ml/overlap model/unet_v2_cv.pt --train 2012-2018 --folds 7
.venv/bin/python -I ml/run_evaluate.py ~/firms/derived ~/firms/derived/ml/eval \
    --model old=model/unet.pt --model new=model/unet_v2_cv.pt --test 2019-2025 --terra-only --importance
```

Smoke test (CPU, under a minute, 20° × 20° window in South America):

```bash
.venv/bin/python -I ml/run_tensor.py ~/firms/derived --years 2015-2021 --preset all --bbox=-60,-20,-40,0 --name smoke
.venv/bin/python -I ml/run_train.py ~/firms/derived/ml/smoke /tmp/smoke.pt --train 2015-2017 --val 2018 --smoke
.venv/bin/python -I ml/run_evaluate.py ~/firms/derived ~/firms/derived/ml/smoke --model old=model/unet.pt \
    --model new=/tmp/smoke.pt --test 2019-2020 --terra-only --importance --imp-months 4 --out /tmp/eval_smoke
```

Stack sizes on the default grid (latitude −56…72): 1.47 MB per channel-month, so 2011–2025 v2 is about
7.6 GB, 2018–2025 all is 6.4 GB, and 2000–2025 v2 (for prediction) is 13.8 GB.

## Reading `report.md` (run_evaluate.py)

- **Median |log error|, anomaly sign, bias**: the `pipeline/03_validate.py` metrics at 1°, 5° and country scale.
- **Gap closed**: (Brasa − model) / (Brasa − ceiling). 0 is Brasa, 1 is as good as a second VIIRS instrument.
  NOAA-20 is a noisy reference, not a hard bound, so a mean predictor can beat it on some metrics.
- **Anomaly slope** < 1 means year-to-year swings are damped. Predicted means are smoother than real
  VIIRS, which matters to anyone studying variability or extremes.
- **Paired bootstrap**: 95% interval of the error difference, resampling months. Only a "better" verdict
  whose interval excludes 0 counts as an improvement. Run 2–3 seeds before trusting small differences.
- **By year**: drift. V/M rises in 2025 as Terra and Aqua drift off their overpass times.
- **Fire regimes / MODIS saw nothing**: the small-fire floor, the old model's weak spot (about −15% bias).
- **Calibration**: randomized PIT of the per-cell NB mixture; 80% coverage should be near 0.80.
- **`[Terra-only]` rows** (`--terra-only`): the v2 model with Aqua removed, against a Terra-only Brasa refit.
  This is the only direct check available for the 2000-11 … 2002-06 months.
- **Warnings** flag test years a model saw in training. For example, the old model is in-sample on 2012–2018.

## Methodology checks built in

- **Backcast**: `run_train.py --train 2016-2025 --val 2015`, then `run_evaluate.py --test 2012-2014
  --brasa-train 2016-2024`. Training forward and testing backward mimics the real use (2000–2011).
- **Ablations** (`run_colab.ipynb` step 2): legacy recipe vs v2, and v2 without each change
  (`--norm group`, `--no-location`, `--depth 2`, `--reg-w 0/1`, `--offsets -12,-1,0,1`).
- **Ensembles**: `--folds K --seeds S`. Every member is early-stopped on its own held-out years, and
  `Predictor` averages member means. The spread between members is part of the predictive mixture.
- **External covariates**: `run_tensor.py --aux file.parquet` adds standardized `aux_*` channels.
  They need columns `cell` (+ `year`, `month` if monthly) on the 0.25° grid and must cover 2000 onward.
  Candidates: MCD64A1 burned area, MOD35/MYD35 cloud or clear-sky fraction, MCD12Q1 land cover, VCF
  tree cover. None ships with the repo; they must be downloaded from NASA Earthdata and regridded.
  Train on them with `run_train.py --aux`.

## Known limits

- Inputs are monthly aggregates, so daily co-occurrence with the Aqua / S-NPP 13:30 overpass is only
  summarised by `days`, `maxday` and `hour`. A daily input tensor would be about 30× larger (~200 GB).
- The 2000–2011 error cannot be measured. The backcast and Terra-only rows are the best available proxies.
- `pipeline/08_merge_unet.py` still keeps the Terra-only months on Brasa M3, and its model card (`nn.json`)
  still describes the old U-Net + MLP ensemble. Update both once v2 has been scored.
- Old helper scripts (`compare.py`, `context_baseline.py`, the per-pixel MLP, `residual_check.py`, the
  two Colab notebooks) were replaced by the files above. They remain in git history (commit `4d18e46`).
