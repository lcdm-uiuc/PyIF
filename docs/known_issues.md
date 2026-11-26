# Known estimator issues

## Legacy KSG implementation

The following issues were identified during Jingyuan Chen's Fall 2026 research experiments. They remain open for investigation.

| Issue | Observed behavior | Status |
|---|---|---|
| History count allocation | Potential incorrect count-array sizing for larger embeddings | Requires investigation |
| Neighbor radius calculation | Potential inconsistency between selected neighbors and projected counting radii | Requires mathematical review |
| Repeated observations and exact zeros | Large positive or negative estimates on known-answer mixed-data benchmarks | Alternative estimator available |
| Constant target series | Nonzero estimates despite zero population transfer entropy | Alternative estimator available |
| Extreme numerical magnitudes | Internal distance calculations may overflow | Requires KSG-specific fix |
| Large integer observations | Conversion to floating point may collapse distinct observations | Requires KSG-specific fix |

The Mesner-Shalizi estimator provides an alternative for mixed-data estimation. Its test suite includes known-answer benchmarks, independent neighbor-counting verification, constant series, extreme numerical scaling, integer precision, and near-perfect prediction.

These tests do not establish that the corresponding issues in the legacy KSG implementation have been resolved.

Future KSG changes should include reproducible regression tests and independent numerical validation.