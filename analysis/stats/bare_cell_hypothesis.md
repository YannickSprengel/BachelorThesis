# Bare-cell hypothesis: has-bare-cell vs. no-bare-cell pages, by dataset

`bare cell` = a kept block whose own tag is td/th/li/dt/dd (lost its table/list wrapper during segmentation -- see analysis/oracle_investigation.md).

**Overall**: has-bare-cell mean rouge5=0.8957 (n=1960) vs. no-bare-cell mean=0.9067 (n=2025), diff=-0.0110, 95% CI [-0.0176, -0.0045] -> significant: yes

| dataset | prevalence (has_bare_cell) | n has-bare | n no-bare | mean rouge5 (has) | mean rouge5 (no) | diff | 95% CI | significant |
|---|---|---|---|---|---|---|---|---|
| cetd | 65.1% | 456 | 244 | 0.9105 | 0.9359 | -0.0254 | [-0.0347, -0.0161] | yes |
| cleaneval | 54.1% | 399 | 339 | 0.9172 | 0.9253 | -0.0081 | [-0.0257, +0.0092] | no |
| cleanportaleval | 69.0% | 49 | 22 | 0.9095 | 0.9063 | +0.0032 | [-0.0349, +0.0414] | no |
| dragnet | 43.1% | 595 | 784 | 0.8662 | 0.8880 | -0.0218 | [-0.0332, -0.0106] | yes |
| google-trends-2017 | 45.0% | 81 | 99 | 0.8153 | 0.8698 | -0.0545 | [-0.1019, -0.0099] | yes |
| l3s-gn1 | 46.5% | 289 | 332 | 0.9200 | 0.9184 | +0.0016 | [-0.0128, +0.0167] | no |
| readability | 33.9% | 39 | 76 | 0.9126 | 0.9178 | -0.0052 | [-0.0509, +0.0415] | no |
| scrapinghub | 28.7% | 52 | 129 | 0.9029 | 0.9072 | -0.0042 | [-0.0262, +0.0176] | no |