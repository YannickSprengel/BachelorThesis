# Paired comparison: oracle vs trainable_minilm

Metric: `rouge5`  |  common pages: 3985

**Overall**: mean diff (A-B) = `+0.0338` (A > B), 95% CI `[+0.0291, +0.0385]` -> **significant** (95% CI excludes 0)

| dataset | n pages | mean diff (A-B) | 95% CI | significant |
|---|---|---|---|---|
| cetd | 700 | -0.0116 | [-0.0161, -0.0071] | yes |
| cleaneval | 738 | +0.0445 | [+0.0339, +0.0558] | yes |
| cleanportaleval | 71 | +0.0186 | [-0.0089, +0.0562] | no |
| dragnet | 1379 | +0.0397 | [+0.0306, +0.0491] | yes |
| google-trends-2017 | 180 | +0.1070 | [+0.0761, +0.1407] | yes |
| l3s-gn1 | 621 | +0.0310 | [+0.0218, +0.0413] | yes |
| readability | 115 | +0.0507 | [+0.0248, +0.0808] | yes |
| scrapinghub | 181 | +0.0530 | [+0.0312, +0.0769] | yes |