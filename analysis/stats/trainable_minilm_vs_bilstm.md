# Paired comparison: trainable_minilm vs bilstm_combined

Metric: `rouge5`  |  common pages: 3985

**Overall**: mean diff (A-B) = `+0.1492` (A > B), 95% CI `[+0.1405, +0.1581]` -> **significant** (95% CI excludes 0)

| dataset | n pages | mean diff (A-B) | 95% CI | significant |
|---|---|---|---|---|
| cetd | 700 | +0.1506 | [+0.1332, +0.1689] | yes |
| cleaneval | 738 | +0.1751 | [+0.1544, +0.1968] | yes |
| cleanportaleval | 71 | +0.1367 | [+0.0687, +0.2138] | yes |
| dragnet | 1379 | +0.1408 | [+0.1263, +0.1558] | yes |
| google-trends-2017 | 180 | +0.1779 | [+0.1371, +0.2203] | yes |
| l3s-gn1 | 621 | +0.0990 | [+0.0794, +0.1188] | yes |
| readability | 115 | +0.3096 | [+0.2423, +0.3782] | yes |
| scrapinghub | 181 | +0.1485 | [+0.1073, +0.1916] | yes |