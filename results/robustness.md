# Robustness analysis

## Main robustness run: 10 worlds, metric v2

| condition | score (mean ± 95% CI) | gain | interference | innate, channel on | battery hidden |
|---|---|---|---|---|---|
| hand-global | -0.212 ± 0.047 | 0.658 | 0.860 | 0.295 | 0.726 |
| hand-local | +0.111 ± 0.014 | 0.127 | 0.006 | 0.961 | 0.380 |
| random-adaptive | +0.285 ± 0.024 | 0.357 | 0.065 | 0.903 | 0.401 |
| ccs-global | +0.409 ± 0.041 | 0.501 | 0.081 | 0.886 | 0.525 |
| ccs-adaptive | +0.395 ± 0.028 | 0.468 | 0.067 | 0.900 | 0.506 |

- ccs-adaptive − random: Δscore = +0.110 ± 0.034; CCS wins in 10/10 worlds (sign test p = 0.002)
- ccs-global − random: Δscore = +0.125 ± 0.052; CCS wins in 9/10 worlds (sign test p = 0.021)
- adaptive − global: Δscore = -0.015 ± 0.055; adaptive wins in 4/10 worlds (p = 0.754)

- ccs-global winners: input {'channel': 10}; code {'linear': 10}; objective {'rec only': 8, 'both': 2}; carries 7.3 latents on average; opens R1 in 10/10; mean plastic steps 600
- ccs-adaptive winners: input {'channel': 10}; code {'linear': 10}; objective {'rec only': 6, 'both': 4}; carries 7.1 latents on average; opens R1 in 7/10; mean plastic steps 390

## Original metric (v1), all worlds: 10 worlds, metric v1

| condition | score (mean ± 95% CI) | gain | interference | innate, channel on | battery hidden |
|---|---|---|---|---|---|
| hand-global | -0.211 ± 0.048 | 0.658 | 0.859 | 0.295 | 0.726 |
| hand-local | +0.117 ± 0.014 | 0.127 | 0.000 | 0.961 | 0.380 |
| ccs-global | +0.488 ± 0.042 | 0.504 | 0.006 | 0.883 | 0.552 |
| ccs-adaptive | +0.476 ± 0.043 | 0.488 | 0.007 | 0.886 | 0.541 |

- adaptive − global: Δscore = -0.012 ± 0.068; adaptive wins in 4/10 worlds (p = 0.754)

- ccs-global winners: input {'channel': 10}; code {'linear': 10}; objective {'rec only': 9, 'both': 1}; carries 7.1 latents on average; opens R1 in 10/10; mean plastic steps 600
- ccs-adaptive winners: input {'channel': 10}; code {'linear': 10}; objective {'rec only': 8, 'both': 2}; carries 7.0 latents on average; opens R1 in 7/10; mean plastic steps 360

## v1 → v2 on the 10 shared worlds

- ccs-global: innate R² with channel ON 0.883 → 0.886; gain 0.504 → 0.501
- ccs-adaptive: innate R² with channel ON 0.886 → 0.900; gain 0.488 → 0.468
