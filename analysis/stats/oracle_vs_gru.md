# Paired comparison: oracle vs gru

Metric: `rouge5`  |  common pages: 3985

**Overall**: mean diff (A-B) = `-0.0055` (A < B), 95% CI `[-0.0083, -0.0026]` -> **significant** (95% CI excludes 0)

| dataset | n pages | mean diff (A-B) | 95% CI | significant |
|---|---|---|---|---|
| cetd | 700 | -0.0133 | [-0.0157, -0.0108] | yes |
| cleaneval | 738 | +0.0165 | [+0.0102, +0.0239] | yes |
| cleanportaleval | 71 | +0.0229 | [-0.0160, +0.0738] | no |
| dragnet | 1379 | -0.0196 | [-0.0245, -0.0146] | yes |
| google-trends-2017 | 180 | +0.0302 | [+0.0075, +0.0560] | yes |
| l3s-gn1 | 621 | -0.0062 | [-0.0125, +0.0010] | no |
| readability | 115 | -0.0025 | [-0.0128, +0.0088] | no |
| scrapinghub | 181 | -0.0046 | [-0.0132, +0.0049] | no |