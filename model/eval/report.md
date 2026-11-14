# Model comparison, test 2019-2025 (79 months)

Stack `data/derived/ml/eval`, Brasa M3 refit on 2012-2018. Models: `old` (1 member(s), trained on 2012-2018), `new` (3 member(s), trained on 2012-2018).

## Warnings

- old: 1 neighbour months missing from the stack (zeros used); build the stack one month wider than the test period.
- new: 1 neighbour months missing from the stack (zeros used); build the stack one month wider than the test period.
- Ceiling rows in regimes.csv / bands.csv compare NOAA-20 against all test-month S-NPP totals; they are only exact when every test year is >= 2018.

## Median |log error| of monthly region totals (lower is better)

| model | 1deg | 5deg | country |
|---|---|---|---|
| old | 0.274 | 0.216 | 0.189 |
| new | 0.272 | 0.210 | 0.181 |
| new [Terra-only] | 0.402 | 0.301 | 0.263 |
| Brasa M3 | 0.312 | 0.253 | 0.206 |
| Brasa M3 [Terra-only] | 0.445 | 0.350 | 0.302 |
| ceiling NOAA-20 | 0.163 | 0.104 | 0.085 |

## Anomaly sign agreement (higher is better)

| model | 1deg | 5deg | country |
|---|---|---|---|
| old | 0.775 | 0.831 | 0.850 |
| new | 0.780 | 0.833 | 0.846 |
| new [Terra-only] | 0.722 | 0.788 | 0.794 |
| Brasa M3 | 0.604 | 0.738 | 0.759 |
| Brasa M3 [Terra-only] | 0.455 | 0.651 | 0.662 |
| ceiling NOAA-20 | 0.813 | 0.907 | 0.910 |

## Share of the Brasa -> NOAA-20 gap closed

| model | 1deg | 5deg | country |
|---|---|---|---|
| old | 0.255 | 0.248 | 0.140 |
| new | 0.268 | 0.289 | 0.207 |
| new [Terra-only] | -0.604 | -0.322 | -0.471 |
| Brasa M3 | 0.000 | 0.000 | 0.000 |
| Brasa M3 [Terra-only] | -0.893 | -0.651 | -0.793 |
| ceiling NOAA-20 | 1.000 | 1.000 | 1.000 |

## Bias % (sum predicted / sum observed - 1)

| model | 1deg | 5deg | country |
|---|---|---|---|
| old | -0.08 | -0.08 | -0.08 |
| new | -4.50 | -4.50 | -4.50 |
| new [Terra-only] | -4.89 | -4.89 | -4.89 |
| Brasa M3 | -2.90 | -2.90 | -2.90 |
| Brasa M3 [Terra-only] | -5.36 | -5.36 | -5.36 |
| ceiling NOAA-20 | -1.26 | -1.26 | -1.26 |

## Anomaly slope (1 = right year-to-year amplitude, < 1 = damped)

| model | 1deg | 5deg | country |
|---|---|---|---|
| old | 0.710 | 0.751 | 0.684 |
| new | 0.704 | 0.754 | 0.631 |
| new [Terra-only] | 0.593 | 0.658 | 0.487 |
| Brasa M3 | 0.599 | 0.589 | 0.427 |
| Brasa M3 [Terra-only] | 0.485 | 0.496 | 0.318 |
| ceiling NOAA-20 | 0.867 | 0.916 | 0.876 |

## Paired bootstrap (95% interval of a - b, median |log error|)

| scale | a | b | diff | lo | hi | verdict |
|---|---|---|---|---|---|---|
| 1deg | old | Brasa M3 | -0.0146 | -0.0170 | -0.0124 | a better |
| 1deg | new | Brasa M3 | -0.0159 | -0.0184 | -0.0130 | a better |
| 1deg | new [Terra-only] | Brasa M3 | 0.1059 | 0.0975 | 0.1152 | b better |
| 1deg | new | old | -0.0013 | -0.0043 | 0.0018 | no clear difference |
| 1deg | new [Terra-only] | old | 0.1205 | 0.1118 | 0.1304 | b better |
| 1deg | new [Terra-only] | new | 0.1218 | 0.1123 | 0.1310 | b better |
| 1deg | new [Terra-only] | Brasa M3 [Terra-only] | 0.0030 | -0.0014 | 0.0078 | no clear difference |
| 5deg | old | Brasa M3 | -0.0184 | -0.0227 | -0.0132 | a better |
| 5deg | new | Brasa M3 | -0.0241 | -0.0282 | -0.0198 | a better |
| 5deg | new [Terra-only] | Brasa M3 | 0.0637 | 0.0554 | 0.0721 | b better |
| 5deg | new | old | -0.0057 | -0.0095 | -0.0027 | a better |
| 5deg | new [Terra-only] | old | 0.0821 | 0.0716 | 0.0902 | b better |
| 5deg | new [Terra-only] | new | 0.0878 | 0.0781 | 0.0958 | b better |
| 5deg | new [Terra-only] | Brasa M3 [Terra-only] | -0.0238 | -0.0290 | -0.0174 | a better |
| country | old | Brasa M3 | -0.0020 | -0.0091 | 0.0045 | no clear difference |
| country | new | Brasa M3 | -0.0118 | -0.0178 | -0.0052 | a better |
| country | new [Terra-only] | Brasa M3 | 0.0685 | 0.0596 | 0.0777 | b better |
| country | new | old | -0.0097 | -0.0151 | -0.0032 | a better |
| country | new [Terra-only] | old | 0.0705 | 0.0601 | 0.0795 | b better |
| country | new [Terra-only] | new | 0.0803 | 0.0707 | 0.0890 | b better |
| country | new [Terra-only] | Brasa M3 [Terra-only] | -0.0170 | -0.0242 | -0.0084 | a better |

## 5-degree error by year

| year | old | new | new [Terra-only] | Brasa M3 | Brasa M3 [Terra-only] |
|---|---|---|---|---|---|
| 2019 | 0.196 | 0.177 | 0.255 | 0.214 | 0.297 |
| 2020 | 0.207 | 0.205 | 0.269 | 0.248 | 0.336 |
| 2021 | 0.199 | 0.191 | 0.273 | 0.240 | 0.330 |
| 2022 | 0.227 | 0.216 | 0.337 | 0.260 | 0.371 |
| 2023 | 0.206 | 0.197 | 0.294 | 0.242 | 0.344 |
| 2024 | 0.236 | 0.233 | 0.332 | 0.280 | 0.384 |
| 2025 | 0.255 | 0.274 | 0.392 | 0.300 | 0.416 |

## Bias % by year

| year | old | new | new [Terra-only] | Brasa M3 | Brasa M3 [Terra-only] |
|---|---|---|---|---|---|
| 2019 | 1.00 | -1.85 | -0.70 | -1.49 | 1.82 |
| 2020 | -1.93 | -5.00 | -2.52 | -4.34 | 1.87 |
| 2021 | 0.33 | -1.89 | -0.25 | -1.72 | 0.94 |
| 2022 | -1.67 | -4.09 | -4.79 | -3.27 | -7.00 |
| 2023 | 0.80 | -2.24 | -4.88 | -3.14 | -9.08 |
| 2024 | 2.39 | -4.93 | -10.18 | -2.07 | -11.47 |
| 2025 | -2.69 | -13.86 | -13.32 | -4.86 | -18.90 |

## Fire regimes (MODIS detections / yr in the Brasa training years), cell-months

Bias %:

| regime | Brasa M3 | Brasa M3 [Terra-only] | ceiling NOAA-20 | new | new [Terra-only] | old |
|---|---|---|---|---|---|---|
| 0 | -6.3 | 1.9 | -1.3 | -5.9 | -7.2 | -7.0 |
| <1 | 1.7 | 15.6 | -0.5 | -11.7 | -12.9 | -10.8 |
| 1-10 | -3.1 | 4.0 | -0.0 | -8.1 | -8.3 | -5.7 |
| 10-100 | -0.8 | -2.0 | -0.9 | -3.9 | -2.0 | 1.1 |
| >=100 | -4.5 | -11.1 | -1.8 | -3.9 | -6.1 | 0.9 |
| MODIS saw nothing | -1.9 | 71.5 | 2.5 | -18.5 | 24.8 | -15.0 |

Cell median |log error| (pred + obs >= 20):

| regime | Brasa M3 | Brasa M3 [Terra-only] | ceiling NOAA-20 | new | new [Terra-only] | old |
|---|---|---|---|---|---|---|
| 0 | 0.546 | 0.689 | 0.194 | 0.446 | 0.577 | 0.424 |
| <1 | 0.622 | 0.988 | 0.235 | 0.516 | 0.717 | 0.505 |
| 1-10 | 0.520 | 0.783 | 0.251 | 0.437 | 0.607 | 0.427 |
| 10-100 | 0.389 | 0.609 | 0.216 | 0.330 | 0.484 | 0.326 |
| >=100 | 0.263 | 0.509 | 0.154 | 0.229 | 0.366 | 0.225 |
| MODIS saw nothing | 1.609 | 1.254 | 0.288 | 1.022 | 0.871 | 0.938 |

## Bias % by latitude band

| band | old | new | new [Terra-only] | Brasa M3 | Brasa M3 [Terra-only] |
|---|---|---|---|---|---|
| N subtropics / temperate (23.5-50N) | -5.4 | -7.0 | -3.2 | -0.2 | -1.7 |
| S extratropics (<23.5S) | 0.2 | -4.5 | -13.3 | -1.9 | -2.1 |
| boreal (>50N) | -7.1 | -6.7 | -5.1 | -7.9 | -5.6 |
| tropics (23.5S-23.5N) | 1.3 | -4.0 | -4.5 | -2.7 | -6.0 |

## Top-15 fire countries: median monthly |log error|

| country | old | new | new [Terra-only] | Brasa M3 | Brasa M3 [Terra-only] |
|---|---|---|---|---|---|
| Democratic Republic of the Congo | 0.060 | 0.060 | 0.134 | 0.082 | 0.164 |
| Brazil | 0.103 | 0.063 | 0.101 | 0.056 | 0.099 |
| Australia | 0.046 | 0.058 | 0.098 | 0.042 | 0.096 |
| Angola | 0.057 | 0.081 | 0.230 | 0.081 | 0.255 |
| Russian Federation | 0.111 | 0.115 | 0.125 | 0.115 | 0.137 |
| Mozambique | 0.072 | 0.107 | 0.282 | 0.122 | 0.225 |
| Zambia | 0.083 | 0.081 | 0.270 | 0.100 | 0.279 |
| South Sudan | 0.061 | 0.073 | 0.120 | 0.078 | 0.207 |
| Central African Republic | 0.085 | 0.122 | 0.172 | 0.097 | 0.241 |
| Canada | 0.101 | 0.095 | 0.143 | 0.119 | 0.153 |
| India | 0.090 | 0.098 | 0.138 | 0.103 | 0.187 |
| United States | 0.113 | 0.136 | 0.107 | 0.087 | 0.099 |
| Bolivia | 0.127 | 0.074 | 0.113 | 0.080 | 0.107 |
| Nigeria | 0.110 | 0.123 | 0.147 | 0.094 | 0.128 |
| Tanzania | 0.059 | 0.064 | 0.157 | 0.078 | 0.152 |

## Calibration of the per-cell predictive distribution (randomized PIT)

| model | coverage_80 (target 0.80) | PIT mean (target 0.50) | PIT hist (10 bins, flat = 1) |
|---|---|---|---|
| old | 0.775 | 0.510 | 1.01 0.98 0.98 0.98 0.96 0.96 0.96 0.96 0.98 1.24 |
| new | 0.778 | 0.521 | 0.94 0.94 0.94 0.96 0.97 0.96 0.98 1.00 1.03 1.28 |
| new [Terra-only] | 0.781 | 0.517 | 0.96 0.94 0.96 0.96 0.97 0.98 0.99 1.00 1.00 1.23 |

## Permutation importance (increase of 5-degree error when a group is scrambled)

**old**: counts +1.015, FRP +0.308, location + climatology (shifted) +0.232, night share +0.077, fire days / burstiness +0.075, pixel size / confidence / brightness +0.017

**new**: counts +1.311, Aqua block (all a_*) +0.408, FRP +0.383, location + climatology (shifted) +0.290, Terra block (all t_*) +0.286, night share +0.094, fire days / burstiness +0.060, pixel size / confidence / brightness +0.029, local overpass time -0.002
