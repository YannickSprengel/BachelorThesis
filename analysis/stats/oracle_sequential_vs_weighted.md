# Paired comparison: oracle_sequential vs oracle_weighted

Metric: `rouge5`  |  common pages: 3985

**Overall**: mean diff (A-B) = `+0.0226` (A > B), 95% CI `[+0.0177, +0.0273]` -> **significant** (95% CI excludes 0)

| dataset | n pages | mean diff (A-B) | 95% CI | significant |
|---|---|---|---|---|
| cetd | 700 | +0.0055 | [-0.0053, +0.0156] | no |
| cleaneval | 738 | -0.0384 | [-0.0518, -0.0256] | yes |
| cleanportaleval | 71 | +0.0563 | [+0.0320, +0.0782] | yes |
| dragnet | 1379 | +0.0538 | [+0.0454, +0.0617] | yes |
| google-trends-2017 | 180 | +0.0724 | [+0.0458, +0.0977] | yes |
| l3s-gn1 | 621 | +0.0204 | [+0.0092, +0.0310] | yes |
| readability | 115 | +0.0189 | [-0.0115, +0.0445] | no |
| scrapinghub | 181 | +0.0463 | [+0.0255, +0.0639] | yes |