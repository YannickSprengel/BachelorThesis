# Paired comparison: oracle vs bilstm

Metric: `rouge5`  |  common pages: 3985

**Overall**: mean diff (A-B) = `+0.1830` (A > B), 95% CI `[+0.1738, +0.1925]` -> **significant** (95% CI excludes 0)

| dataset | n pages | mean diff (A-B) | 95% CI | significant |
|---|---|---|---|---|
| cetd | 700 | +0.1391 | [+0.1202, +0.1586] | yes |
| cleaneval | 738 | +0.2196 | [+0.1974, +0.2426] | yes |
| cleanportaleval | 71 | +0.1553 | [+0.0870, +0.2327] | yes |
| dragnet | 1379 | +0.1805 | [+0.1646, +0.1968] | yes |
| google-trends-2017 | 180 | +0.2849 | [+0.2399, +0.3305] | yes |
| l3s-gn1 | 621 | +0.1300 | [+0.1100, +0.1502] | yes |
| readability | 115 | +0.3603 | [+0.2948, +0.4292] | yes |
| scrapinghub | 181 | +0.2016 | [+0.1603, +0.2457] | yes |