# Submission package

This directory contains the generated artifacts and results evidence for the
KuaiRand-Pure submission.

| File | Purpose |
|---|---|
| `RESULTS_AND_RESOURCES.md` | Official-baseline deltas, convergence, tokens, time, and interventions |
| `kuairand_pure_submission.csv` | Generated final prediction rows |
| `kuairand_pure_fm_seed0.npz` | Generated NumPy FM checkpoint |
| `submission_manifest.json` | Provenance, configuration, hashes, and validation metrics |

Regenerate the model output, checkpoint, and manifest from the repository root:

```bash
python3 make_final_submission.py
python3 submit.py submission/kuairand_pure_submission.csv --check
```

The final recommender is fully offline and requires only Python and NumPy.
The OpenAI API was used by the research agent during development and is
reported transparently in the Devpost draft and run logs.
