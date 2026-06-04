
# Results

## AG News - MILE Proof of Concept

Date: 2026-05-24

Configuration:
- Dataset: AG News
- Datapoints: 1000
- Train Split: 70%
- Validation Split: 10%
- Test Split: 20%
- Chains: 2
- Warmup Steps: 500
- Samples per Chain: 500
- Thinning: 2

Results:
- Deep Ensemble Accuracy: 0.6567
- Bayesian Deep Ensemble Accuracy: 0.6169

Runtime:
- Warmstart: 0.43 min
- BDE Sampling: 39.37 min
- Total: 39.8 min

Notes:
- Initial proof-of-concept run to verify the MILE pipeline on AG News.
- Full-scale experiments will be conducted with larger datasets and additional posterior approximation methods.
