# Paired comparison: oracle_weighted vs oracle

Metric: `rouge5`  |  common pages: 3985

**Overall**: mean diff (A-B) = `+0.0035` (A > B), 95% CI `[+0.0028, +0.0041]` -> **significant** (95% CI excludes 0)

| dataset | n pages | mean diff (A-B) | 95% CI | significant |
|---|---|---|---|---|
| cetd | 700 | +0.0030 | [+0.0023, +0.0038] | yes |
| cleaneval | 738 | +0.0011 | [-0.0010, +0.0027] | no |
| cleanportaleval | 71 | +0.0041 | [+0.0008, +0.0087] | yes |
| dragnet | 1379 | +0.0051 | [+0.0042, +0.0062] | yes |
| google-trends-2017 | 180 | +0.0065 | [+0.0034, +0.0111] | yes |
| l3s-gn1 | 621 | +0.0025 | [+0.0004, +0.0041] | yes |
| readability | 115 | +0.0025 | [+0.0009, +0.0045] | yes |
| scrapinghub | 181 | +0.0031 | [+0.0012, +0.0055] | yes |