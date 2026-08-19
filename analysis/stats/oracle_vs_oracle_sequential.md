# Paired comparison: oracle_sequential vs oracle

Metric: `rouge5`  |  common pages: 3985

**Overall**: mean diff (A-B) = `+0.0260` (A > B), 95% CI `[+0.0211, +0.0309]` -> **significant** (95% CI excludes 0)

| dataset | n pages | mean diff (A-B) | 95% CI | significant |
|---|---|---|---|---|
| cetd | 700 | +0.0085 | [-0.0025, +0.0187] | no |
| cleaneval | 738 | -0.0374 | [-0.0509, -0.0245] | yes |
| cleanportaleval | 71 | +0.0605 | [+0.0351, +0.0840] | yes |
| dragnet | 1379 | +0.0590 | [+0.0504, +0.0671] | yes |
| google-trends-2017 | 180 | +0.0789 | [+0.0518, +0.1053] | yes |
| l3s-gn1 | 621 | +0.0229 | [+0.0113, +0.0338] | yes |
| readability | 115 | +0.0214 | [-0.0092, +0.0477] | no |
| scrapinghub | 181 | +0.0494 | [+0.0284, +0.0672] | yes |