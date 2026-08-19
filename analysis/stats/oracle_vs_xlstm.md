# Paired comparison: oracle vs xlstm

Metric: `rouge5`  |  common pages: 3985

**Overall**: mean diff (A-B) = `-0.0012` (A < B), 95% CI `[-0.0042, +0.0018]` -> not significant

| dataset | n pages | mean diff (A-B) | 95% CI | significant |
|---|---|---|---|---|
| cetd | 700 | -0.0077 | [-0.0101, -0.0050] | yes |
| cleaneval | 738 | +0.0245 | [+0.0169, +0.0329] | yes |
| cleanportaleval | 71 | -0.0036 | [-0.0208, +0.0217] | no |
| dragnet | 1379 | -0.0216 | [-0.0260, -0.0172] | yes |
| google-trends-2017 | 180 | +0.0526 | [+0.0228, +0.0851] | yes |
| l3s-gn1 | 621 | +0.0042 | [-0.0025, +0.0117] | no |
| readability | 115 | -0.0010 | [-0.0141, +0.0128] | no |
| scrapinghub | 181 | +0.0036 | [-0.0102, +0.0191] | no |