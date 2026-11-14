# Evaluation of the MODIS → VIIRS U-Net

I read `features.py`, `build_tensor.py`, `train.py`, `core.py` and `evaluate.py`, plus your latest `comparison.csv` and `train.log`.

## 1. First, the metrics as quoted mix the model with the ceiling

The ranges in the brief aren't ranges across scales. 0.265 is the U-Net at 1°, and 0.163 is the NOAA-20 vs S-NPP ceiling at 1°. The sign-agreement range has the same problem. Here is the actual table:

| | Brasa M3 | U-Net | NOAA-20 ceiling | Share of gap closed |
|---|---|---|---|---|
| Median abs log err, 1° | 0.305 | **0.265** | 0.163 | 28% |
| Median abs log err, 5° | 0.246 | **0.210** | 0.105 | 26% |
| Median abs log err, country | 0.199 | **0.183** | 0.087 | 14% |
| Sign agreement, 1° / 5° / country | 0.65 / 0.77 / 0.78 | **0.82 / 0.86 / 0.87** | 0.85 / 0.93 / 0.93 | — |

So the model has closed about a quarter of the gap to the ceiling at 1° and 5°, and less at country scale. Sign agreement is nearly at the ceiling at 1° but not at 5° or country scale.

Treat the 0.01% bias as luck from global calibration, not a strength. Per-year bias is −2.9% in 2025, and the bias is −15% in cells where MODIS detected nothing.

## 2. Domain background that matters for the model

- **Pixel size.** VIIRS uses 375 m pixels (I-band, at nadir), and they stay at most about 2× that size toward the swath edge. MODIS pixels are 1 km at nadir and grow to about 2 × 4.8 km at the edge. So VIIRS can detect fires roughly an order of magnitude weaker. The cells where VIIRS has fire and MODIS has none are the part MODIS physically cannot see. A model can only infer them from context.
- **Overpass timing is a large, usable signal.** Aqua and S-NPP both cross the equator at about 13:30, and NOAA-20 runs about 50 min ahead of S-NPP in the same orbit. Terra crosses at 10:30, so it sees different fires and different clouds. Aqua detections are therefore much more predictive of VIIRS than Terra detections. Right now only `m_terra` carries this; every channel could be split by satellite.
- **Orbit drift.** Terra (from 2020) and Aqua (from 2022) stopped holding their crossing times and have been drifting since. That is the likely cause of V/M rising to 4.47 in 2025. Your 2019–2024 test set includes drifted years, but the period you actually apply the model to (2000–2011) does not.
- **Clear-sky views are missing.** FIRMS only lists detections. The model can't tell "MODIS looked and saw no fire" from "it was cloudy." That missing information is probably the largest one left.

## 3. Methodology problems, roughly by importance

1. **The test doesn't match the deployment.** You train on 2012–2017 and test forward in time, but you apply the model backward to 2000–2011. That period has Terra-only months (until mid-2002), much larger deforestation-fire regimes (the Amazon peaked around 2004), and no orbit drift. Add a **backcast split**: train on roughly 2016–2025 and test on 2012–2014. That is the closest honest stand-in for the real use case.
2. **Patch sampling is weighted by the target.** `Patches` samples in proportion to √(VIIRS count). When samples are chosen by the outcome, the learned E[y|x] gets skewed. Validation is sampled the same way, so the val loss doesn't reflect full-map performance. Weight by inputs instead (MODIS count or climatology), and validate on whole months using the test metrics.
3. **Selection rests on one validation year.** Early stopping and every hyperparameter choice depend on 2018 alone, and seed variance has never been measured. Before trusting any improvement, run 3–5 seeds and put bootstrap confidence intervals on every metric. Then switch to blocked cross-validation by year. For the final production model, retrain on the full 2012–2025 overlap, which more than doubles the training data.
4. **The regional loss term overfits.** In the log, the 2° regional loss is 0.069 on train vs 0.165 on validation, while the pixel NB loss is 0.59 vs 0.60. The regional term is memorizing region-specific totals.
5. **The receptive field (~17°) plus lat/lon inputs let the model learn "where" instead of "what."** That helps on the 2019–2024 test but is fragile in the 2000s, when fire fronts were in different places. Ablate the lat/lon inputs and try a receptive field of about 2–4°. If accuracy holds, keep the more robust version.
6. **The Terra-only era isn't handled at all.** You can simulate it exactly: drop Aqua detections from the overlap inputs. Train with random Aqua dropout and an availability flag, or train a separate Terra-only model. Unlike most of the 2000s shift, this one can be validated directly.
7. **The output is a conditional mean, which is smoother than real VIIRS data.** Researchers studying variance or extremes in 2000–2011 would be misled. Publish quantiles (you already estimate the NB α), and check calibration: p10–p90 coverage, PIT histograms, and the slope of predicted vs observed anomalies. The slope will likely come out below 1, meaning anomalies are damped.

## 4. Improvements, ordered by expected gain per unit of effort

**Phase 0: measurement (cheap, needed before anything else)**
- Seed variance and confidence intervals.
- The backcast split.
- Errors broken down by year, biome and MODIS-zero cells.
- Calibration metrics.
- A "share of gap closed" column next to each metric.

**Phase 1: training fixes**
- Input-weighted sampling.
- Validation on full maps using the actual test metric.
- Ensembles over year-folds or seeds. In this low-data regime these typically gain a few percent.
- A sweep of `--reg-w` and a loss over several scales.
- The lat/lon and receptive-field ablations.
- The pending gradient-boosted-tree baseline with neighbourhood features. If it matches the U-Net, that tells you the gain comes from context, not convolution.

**Phase 2: new inputs available since 2000 (likely the largest gain).** Your residual check showed that MODIS's own per-cell descriptors explain about 3% of the error. More architecture work on the same inputs won't fix that; new information can.
- **Cloud or clear-observation counts** from MOD35/MYD35 or the MOD14A1/MYD14A1 QA fields. These give the model the denominator.
- **MCD64A1 burned area** (500 m). It records burn scars from fires the active-fire product missed, which targets the MODIS-zero bias directly.
- **Land cover and tree cover** (MCD12Q1, VCF).
- **Per-satellite and day/night versions of every channel.**

The hard constraint is that any input must exist and be consistent from 2000 onward.

**Phase 3: era robustness**
- Aqua-dropout augmentation for Terra-only months.
- Overpass local time as a covariate.
- A final refit on all overlap years.

**Phase 4: architecture (smallest expected gain)**
- Daily MODIS inputs fed to a temporal encoder, summed to the month. This captures co-occurrence with Aqua's same-day overpass.
- Longer temporal context: t±2 months and the same month last year.

Neither is worth the effort until Phases 1–2 stop paying off.

## 5. Realistic limits

- **NOAA-20 is a reference, not a hard bound.** It is itself a noisy measurement, so a mean predictor can beat it on some metrics. Yours already does on log correlation at 1° (0.960 vs 0.956). On the other hand, both VIIRS instruments see the same fires and clouds about 50 minutes apart, and MODIS can't share that. So the true floor sits somewhere around the ceiling, not far below it.
- **My rough estimate**, not measured: architecture and training changes alone bring 1° error to about 0.25–0.26. Adding cloud and burned-area inputs plus ensembling brings it to perhaps 0.22–0.24. Getting close to 0.163 is unlikely, because sub-threshold fires uncorrelated with anything MODIS records are simply unobservable.
- **The 2000–2011 error will be higher than the test error, and it can't be measured directly.** Report it with intervals, use the backcast split as a proxy, and treat Terra-only months (2000–mid-2002) as a separate, lower-quality tier.

If you want this as a shareable page or doc for the team, I can publish one.