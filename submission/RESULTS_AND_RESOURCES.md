# Final results and resource usage

## Canonical result

- Run: `runs/agent-openai-v3/`
- Validation-best proposal: iteration 2
- Configuration: `loss='pairwise'`, `sampler='user'`, `feature_fn='base'`,
  `k=16`, learning rate `0.001`
- Selection: three-seed validation mean, guarded by `baseline.is_real()`
- Five-seed confirmation: iteration 5, explicitly logged as one manual
  intervention

### Validation

| Metric | Official baseline | Agent | Raw-point delta |
|---|---:|---:|---:|
| GAUC | 0.667400 | 0.668762 | +0.001362 |
| nDCG@5 | 0.535700 | 0.536793 | +0.001093 |
| primary | 0.601600 | 0.602778 | +0.001178 |

Using the organizer's scoring formula, the mean of the two metric deltas is
`(+0.001362 + +0.001093) / 2 = +0.001228`. The small difference from subtracting
the displayed primary values is caused by rounding in the published baseline.

### Organizer-defined test split

| Metric | Official baseline | Agent three-seed mean | Raw-point delta |
|---|---:|---:|---:|
| GAUC | 0.661000 | 0.663094 | +0.002094 |
| nDCG@5 | 0.528200 | 0.530406 | +0.002206 |
| primary / dataset score | 0.594600 | 0.596750 | +0.002150 |

The uploaded CSV is regenerated as a single seed-0 validation-best checkpoint;
the organizer's hidden evaluator remains the authoritative final score.

## Convergence accounting

The challenge rule uses epsilon `0.002` and N `3`. The canonical history first
satisfies `logger.check_convergence()` after logged iteration 4.

| Resource | Required through first convergence | Complete canonical audit trail |
|---|---:|---:|
| Logged iterations | 4 / 50 | 6 / 50 |
| LLM-generated proposals | 3 | 4 |
| Input tokens | 20,575 | 28,071 |
| Output tokens | 14,951 | 20,122 |
| Total LLM tokens | 35,526 | 48,193 |
| Agent wall-clock | 972.5 seconds (16.2 min) | 1,259.2 seconds (21.0 min) |
| Manual interventions | 0 | 1 |
| Local GPU-hours required | 0 | 0 |

The full trail is reported conservatively even though the final kept state was
already established at iteration 2. The sole manual intervention was a
five-seed confirmation of the kept configuration; it did not change the model.
The canonical FM evaluation is CPU/NumPy. Hosted API compute is represented by
token counts rather than claimed as local GPU-hours.

## Iteration synopsis

| Iteration | Event | Outcome |
|---:|---|---|
| 1 | Establish original FM baseline | Valid primary 0.601440 |
| 2 | Pairwise within-user loss + user-grouped sampler | Kept; valid primary 0.602778 |
| 3 | LambdaRank-style refinement | Reverted; no real improvement |
| 4 | Equal per-user pairwise weighting | Reverted; degraded |
| 5 | Manual five-seed confirmation | Confirmed; no state change |
| 6 | Sharpen pairwise temperature | Reverted; no real improvement |

See `runs/agent-openai-v3/iterations.jsonl` for complete hypotheses, diffs,
metrics, reflections, and recovery records, and `summary.json` for totals.

## Additional experiments not used for ranking

Example collection later found a GAUC-weighted pairwise variant with stronger
local metrics, but that occurred in collection mode with normal convergence
deliberately ignored; it is not presented as the protocol-compliant canonical
result. A 48-example Gemma 4 E2B LoRA pilot improved initial JSON-schema
compliance from 0/3 to 2/3 in matched trials but produced 0/3 applicable code
patches, so it has no downstream result and is not part of the final model.
