# KuaiRand-Pure Autonomous Research Agent

An autonomous ML research agent that improves a Factorization Machine (FM)
recommender on the [KuaiRand-Pure](https://kuairand.com) within-user ranking
benchmark. An LLM (Claude or GPT) proposes a code change, the harness applies
it to a sandboxed copy of the pipeline, evaluates it over multiple seeds, and
keeps the change only if it clears a statistical significance bar — otherwise
it's reverted and the loop tries again. Every iteration is logged.

This README documents the project, setup, results, limitations, and steps
needed to reproduce the final model. Regenerate the offline prediction file,
NumPy checkpoint, and provenance hashes with:

```bash
python3 make_final_submission.py
python3 submit.py submission/kuairand_pure_submission.csv --check
```

## Final submission artifacts

The repository includes the required final output and its reproducibility
evidence:

| File | Purpose |
|---|---|
| [`submission/kuairand_pure_submission.csv`](submission/kuairand_pure_submission.csv) | Final predictions in the required `row_id,user_id,video_id,score` schema |
| [`submission/kuairand_pure_fm_seed0.npz`](submission/kuairand_pure_fm_seed0.npz) | Validation-best NumPy FM checkpoint |
| [`submission/submission_manifest.json`](submission/submission_manifest.json) | Configuration, provenance, validation metrics, and SHA-256 hashes |
| [`submission/RESULTS_AND_RESOURCES.md`](submission/RESULTS_AND_RESOURCES.md) | Baseline deltas, convergence, token usage, wall-clock time, and interventions |

The dataset remains excluded because it is downloaded separately. Running the
commands above regenerates and validates these artifacts locally.

## Project overview

- **Task**: KuaiRand-Pure within-user ranking over logged impressions, using
  the native `long_view` relevance label. Metrics are GAUC, nDCG@5, and
  `primary = mean(GAUC, nDCG@5)`. The frozen `evaluate.py` implements the
  evaluation contract used throughout the project.
- **Starting point**: a from-scratch FM baseline (`baseline.py`), no
  external ML libraries, just numpy.
- **The agent loop** (`agent.py`): each iteration, an LLM sees the full
  source of the editable files, the run history, and a set of established
  facts (noise floor, known dead ends, current best), and returns one
  proposal — a hypothesis plus a set of file changes. The harness:
  1. rejects the proposal outright if any changed file isn't on the
     editable allowlist (before writing anything),
  2. applies the change to a scratch workspace, never the real repo,
  3. evaluates it in an isolated subprocess with `run_multiseed` (3 seeds),
  4. accepts it only if `is_real()` says the improvement clears seed noise,
     otherwise reverts every changed file,
  5. logs the full record — kept, reverted, or errored — to
     `runs/<run_id>/iterations.jsonl`,
  6. checks `check_convergence()` and stops early once progress plateaus.
- **Result so far**: a within-user pairwise (BPR-style) ranking loss,
  combined with a user-grouped batch sampler, raised the FM's test primary
  from the published baseline 0.5946 to 0.5968 — see Results below.

## Setup

Requirements: Python 3.9+, `numpy` (for the baseline pipeline itself), plus
`python-dotenv` and the SDK for whichever remote provider you use with the
agent. Remote-provider packages are not needed to run `baseline.py` alone.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Select `.venv/bin/python` as your IDE interpreter. `python-dotenv` loads this
repository's ignored `.env` file for CLI and IDE runs.

### Dataset

Download from Zenodo (no registration required) and extract into
`./KuaiRand-Pure/`:

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

The dataset (`KuaiRand-Pure/`, `KuaiRand-Pure.tar.gz`) is **not committed** —
see `.gitignore`. Everyone who clones this repo downloads it themselves.

### Credentials

The agent calls an LLM provider's API. Set the key for whichever provider
you're using:

```bash
export ANTHROPIC_API_KEY=...   # for --provider anthropic (default)
export OPENAI_API_KEY=...      # for --provider openai
```

Alternatively, put `OPENAI_API_KEY=...` or `ANTHROPIC_API_KEY=...` in this
repository's ignored `.env` file. `llm.py` loads that file automatically, but
an already-exported shell variable takes precedence. Never commit a key.

## Harness self-checks

**Before touching any code**, run these three and confirm your environment
reproduces the published reference numbers. If they don't roughly match,
something about your environment or data is off — fix that before running
the agent.

```bash
python3 baseline.py --model random
python3 baseline.py --model pop
python3 baseline.py --model fm --multiseed
```

| model | test primary | notes |
|---|---|---|
| `random` | ≈ 0.475 | single-seed run; published reference (`baseline_scores.json`) is a 5-seed mean of 0.4753 — expect noise around it, not an exact hit |
| `pop` | **0.5715** | deterministic (Bayesian-smoothed popularity, no RNG) — should match exactly |
| `fm --multiseed` | **0.5950 ± 0.0003** | 3-seed mean (this repo's convention); published reference is a 5-seed mean of 0.5946 ± 0.0008 — consistent within noise |

`pop` is your canary: if it doesn't hit 0.5715 exactly, the data load, split
boundaries, or evaluation code has diverged from what this repo expects.

## Architecture

### Editable / frozen file split

The agent can only ever propose changes to four files. Everything else is
off-limits by construction.

| File | Status | Role |
|---|---|---|
| `data.py` | editable | loading, splits, feature construction (`FEATURE_FNS`) |
| `losses.py` | editable | swappable loss functions (`LOSSES`) |
| `config.py` | editable | hyperparameters + which loss/feature/sampler is active |
| `baseline.py` | editable | the FM model, training loop, CLI |
| `evaluate.py` | **frozen** | the metric contract — GAUC/nDCG@5/primary. Never touch this; it defines what "improvement" means |
| `submit.py`, `logger.py`, `agent.py`, `llm.py` | **frozen** | submission tooling and the harness itself |

### Allowlist enforcement

`agent.py`'s `_is_allowlisted()` does an actual path check (rejects path
separators/traversal too) against `EDITABLE_ALLOWLIST = {'data.py',
'losses.py', 'config.py', 'baseline.py'}` — not a comment, not a convention.
A proposal can touch up to 3 files in one change (`changes: [...]`); **if
any one of them fails the check, the whole proposal is rejected before a
single byte is written.**

### Subprocess isolation

Nothing the agent proposes is ever written to this checked-out copy of the
repo. Each run gets a scratch workspace (`runs/<run_id>/workspace/`), copied
from the real files (or from another run's kept state via `--seed-from`).
Every evaluation runs `run_multiseed` in a **separate subprocess**, with its
own interpreter and a hard timeout, against that workspace. A syntax error,
an import failure, an uncaught exception, a NaN metric, or a timeout are all
caught, logged with a full traceback, and the workspace is rolled back to
the pre-change snapshot — none of them crash the loop.

### Multi-seed gating

A single seed's noise (~0.0008 std on FM's test primary) swamps most real
deltas. Every evaluation is `run_multiseed(seeds=(0,1,2))`, and a change is
only kept if `is_real(new_mean, new_std, base_mean, base_std)` passes: the
improvement must exceed `max(2·(new_std+base_std)/√3, 0.001)` — a 2-sigma
band on the difference of means, floored at 0.001 absolute. Anything smaller
is reverted, no matter how the run "feels."

### Logger schema

`RunLogger` (`logger.py`) appends one JSON record per iteration to
`runs/<run_id>/iterations.jsonl` and rewrites `runs/<run_id>/summary.json`
after every call.

Per-iteration record: `iteration`, `parent_iteration` (which state this one
branched from — lets a reverted dead end not look like a chain link),
`timestamp`, `hypothesis` (written before the run), `notes` (post-hoc
reflection: expected vs. actual), `diff` (unified diff across every changed
file), `config` (the actual `Config` in effect, parsed from the file
content, not assumed), `metrics` (full `run_multiseed` output or `null`),
`beat_previous_best`, `error`/`traceback`/`handled`, `manual_intervention`,
`provider`/`model`, `tokens_in`/`tokens_out`, `wall_clock_sec`.

`summary.json`: `total_iterations`, token/wall-clock totals,
`best_valid_primary` (only `is_real`-gated keeps — never a reverted
attempt's number), `best_observed_valid_primary` (highest mean ever seen,
including reverted attempts — a diagnostic, not the real best),
`final_delta_over_baseline` (the *kept* state's test-primary delta over the
published 0.5946 baseline — reads ≈0 if nothing has ever been kept),
`manual_interventions` (count of human-initiated log entries, e.g.
higher-seed confirmations).

## Running the agent

```bash
python3 agent.py [--max-iterations N] [--run-id ID] [--dry-run | --no-dry-run]
            [--provider anthropic|openai|ollama|mlx] [--model NAME] [--seed-from RUN_ID]
            [--collect-examples] [--ignore-convergence]
```

| Flag | Default | Meaning |
|---|---|---|
| `--max-iterations` | `3` | proposals to attempt this invocation (a "converged" plateau can stop it earlier) |
| `--run-id` | timestamp-generated | which log under `runs/` to write to (and resume, if it already has history) |
| `--dry-run` / `--no-dry-run` | `--dry-run` (on) | dry-run generates and logs proposals but never writes or evaluates them — use this to sanity-check the LLM/prompt plumbing before letting it touch anything. `--no-dry-run` is required for a real run |
| `--provider` | `anthropic` | which LLM API to call (`llm.py`'s `PROVIDERS`) |
| `--model` | provider's default (`claude-opus-5` / `gpt-5` / `gemma4:e4b-mlx` / `mlx-community/gemma-4-e2b-it-4bit`) | override the model name; MLX accepts `BASE::ADAPTER_DIR` for a LoRA adapter |
| `--seed-from` | none | on a *fresh* `run_id`, copy another run's current best-so-far workspace as the starting point instead of the pristine repo files — use this to keep exploring from a kept result under a clean log/convergence window |
| `--collect-examples` | off | save the exact prompt, raw/repair responses, final normalized proposal, and verification metadata under `runs/<run_id>/examples/<iteration>/` for later review and fine-tuning export |
| `--ignore-convergence` | off | run all `--max-iterations` even if the normal research plateau rule fires; useful for controlled dataset collection, not routine research |

## Collecting fine-tuning examples

Collection is opt-in and works with dry or real runs. Real runs provide the
strongest verification because the final proposal reaches the three-seed
evaluator:

```bash
python3 agent.py --max-iterations 1 --run-id collect-qwen \
  --no-dry-run --provider ollama --model qwen3.5:9b-mlx \
  --collect-examples
```

Each proposal produces:

```text
runs/collect-qwen/examples/0002/
  prompt.txt
  raw_response.txt
  responses.json
  repaired_proposal.json
  verification.json
```

List collected examples, inspect their files and metrics, then record a human
semantic review. Automatic checks alone do not approve training data:

```bash
python3 example_data.py list --run-id collect-qwen
python3 example_data.py review runs/collect-qwen/examples/0002 \
  --label approved --notes "Compiles, matches its stated hypothesis, and was evaluated"
python3 example_data.py review runs/collect-qwen/examples/0003 \
  --label rejected --notes "Uses the wrong tuple indices"
```

At the end of every `--collect-examples` run, the agent prints the small set of
verified improvements and valid examples at or above validation primary `0.6035`
to review. Invalid, unevaluated, and lower-scoring proposals are auto-rejected.
It never approves an example. Re-run triage manually or use a different fixed
review floor when needed:

```bash
python3 example_data.py triage --run-id collect-qwen --apply
python3 example_data.py triage --run-id collect-qwen --min-valid-primary 0.6040 --apply
```

When you have explicitly decided to accept every fully evaluated pending
example at a fixed score floor, record that manual decision in one batch and
refresh the non-destructive approved view:

```bash
python3 example_data.py approve-above --run-id collect-qwen \
  --min-valid-primary 0.6035
```

This writes the human approval label and a `manual_intervention=True` run-log
entry, then creates `runs/approved/`. That directory contains links back to
the original example evidence plus a `manifest.json` with the approved and
deduplicated-exportable counts. It reports how many unique approved examples
remain to reach the default 100-example target; it does not move or duplicate
the source evidence.

Export approved, fully evaluated examples into deterministic MLX-LM chat
splits (80/10/10 once there are 100 examples):

```bash
python3 example_data.py export --run-id collect-qwen --output-dir finetune-data
```

The output contains `train.jsonl`, `valid.jsonl`, `test.jsonl`, and a
`manifest.json` mapping every exported row back to its source run. Compile-only
examples require the explicit `--allow-compile-only` flag during both approval
and export; do not use that override unless a human has verified the proposal.

For 100 proposals, use ten independent batches of ten so each prompt history
stays bounded. This performs real three-seed evaluations and can take hours:

```bash
for batch in {01..10}; do
  python3 agent.py --max-iterations 10 --ignore-convergence \
    --run-id "collect-qwen-${batch}" --no-dry-run \
    --provider ollama --model qwen3.5:9b-mlx \
    --seed-from api-best-seed --collect-examples
done
```

This creates 100 candidates, not automatically 100 approved examples. Review
each candidate and collect additional batches until 100 genuinely correct,
approved examples remain after rejection and deduplication.

Resuming an existing `--run-id` skips re-evaluating the baseline (it reuses
the logged best-so-far — every kept/reverted proposal leaves the workspace
at exactly that state) and continues from wherever that run left off.

## Fine-tuning Gemma 4 E2B locally

The local Apple-silicon pilot targets
`mlx-community/gemma-4-e2b-it-4bit`. Gemma 4 E4B inference fits this machine,
but its MLX-VLM QLoRA training step does not fit in 24 GB of unified memory.
E2B trains with the same proposal format using a 2,048-token cap that fits the
longest batches within 24 GB. The compacted prompts place the proposal response
before that cutoff, although the tail of some long responses is truncated. The
wrapper uses completion-only QLoRA, batch size 1, LoRA rank 8, and never trains
on the held-out validation or test splits.

Install the optional training dependencies, export the approved examples, and
run the one-step memory canary:

```bash
.venv/bin/python -m pip install -r requirements-finetune.txt
python3 example_data.py export --output-dir finetune-data
.venv/bin/python finetune_gemma.py --mode smoke
```

The smoke adapter proves that the training path works; it is not a useful
fine-tuned model. Run the two-pass pilot after the smoke test succeeds:

```bash
.venv/bin/python finetune_gemma.py --mode pilot
```

With the current 48 unique approved examples, the deterministic split contains
39 training, 3 validation, and 6 test examples, so the default pilot performs
78 optimizer steps. More diverse approved examples are still preferable before
calling the result final.

Benchmark the untuned E2B checkpoint and the E2B checkpoint plus adapter on the
same held-out prompts and agent settings. The earlier E4B and Qwen runs are
useful historical references but are not a valid before/after control for E2B.
Do not treat training loss or the one-step smoke result as an agent-quality
improvement; proposal validity and downstream three-seed evaluation remain the
actual measures.

The `mlx` provider runs inference in a child process so model memory is freed
before FM evaluation. Use `BASE::ADAPTER_DIR` to load the tuned adapter. These
are the matched benchmark commands for generation seed 0:

```bash
MLX_SEED=0 .venv/bin/python agent.py --max-iterations 1 \
  --run-id mlx-e2b-untuned-v1 --no-dry-run --provider mlx \
  --model mlx-community/gemma-4-e2b-it-4bit --seed-from api-best-seed

MLX_SEED=0 .venv/bin/python agent.py --max-iterations 1 \
  --run-id mlx-e2b-pilot-v1 --no-dry-run --provider mlx \
  --model 'mlx-community/gemma-4-e2b-it-4bit::finetune-output/pilot' \
  --seed-from api-best-seed
```

The initial three-pair comparison used generation seeds 0, 1, and 2, with a
fresh run and identical research state for every trial:

| E2B variant | valid first response schema | exact replacements valid | reached FM evaluation |
|---|---:|---:|---:|
| Untuned | 0/3 | 0/3 | 0/3 |
| 48-example pilot adapter | 2/3 | 0/3 | 0/3 |

The pilot improved initial schema compliance but did not produce an applicable
proposal. Therefore it has no downstream metric and cannot yet be called
better than untuned E2B. The `0.6028` shown at the end of these runs is the
unchanged `api-best-seed` FM state, not a score achieved by either proposal.
The evidence is under `runs/mlx-e2b-{untuned,pilot}-*`.

## Reproducing the result

The kept result is a config change, not a code change you need to hand-apply:

```bash
python3 baseline.py --model fm --multiseed --loss pairwise --sampler user
```

This runs the within-user pairwise (BPR-style) loss under the user-grouped
batch sampler over 3 seeds and prints valid/test GAUC, nDCG@5, and primary.
Expect valid primary ≈ 0.6028 ± 0.0002, test primary ≈ 0.5968 ± 0.0001 (see
Results below). The pairwise loss and user sampler are available in the root
pipeline, while `config.py` intentionally retains the original logloss/row
defaults; the command-line flags above activate the achieved configuration.

## Results

Baseline is this repo's own 3-seed reproduction of the published FM
baseline (bit-identical training path; `--multiseed` with the default
`loss='logloss'`, `sampler='row'`). Achieved is the kept result,
`loss='pairwise'`, `sampler='user'`.

| | valid GAUC | valid nDCG@5 | valid primary | test GAUC | test nDCG@5 | test primary |
|---|---|---|---|---|---|---|
| Baseline (3-seed) | 0.6672 ± 0.0001 | 0.5357 ± 0.0004 | 0.6014 ± 0.0003 | 0.6614 ± 0.0005 | 0.5285 ± 0.0001 | 0.5950 ± 0.0003 |
| **Achieved (3-seed)** | **0.6688 ± 0.0003** | **0.5368 ± 0.0002** | **0.6028 ± 0.0002** | **0.6631 ± 0.0001** | **0.5304 ± 0.0002** | **0.5968 ± 0.0001** |
| Achieved (5-seed confirm) | — | — | 0.6026 ± 0.0003 | — | — | 0.5966 ± 0.0002 |
| Published baseline (5-seed) | 0.6674 | 0.5357 | 0.6016 | 0.6610 | 0.5282 | **0.5946** |

- **Δ over published baseline**: +0.0014 valid primary, +0.0022 test primary
  — real (clears the `is_real` significance bar), confirmed at 5 seeds (the
  5-seed re-check landed within noise of the 3-seed result, not a lucky draw).
- **Cost**: found at **iteration 2 of a 50-iteration budget** (run
  `agent-openai-v3`); across that run's full 6 logged iterations (the kept
  proposal, two reverted refinements, a manual 5-seed confirmation, and one
  more reverted refinement after resuming) — **48,193 tokens** (28,071 in /
  20,122 out), **~21 minutes** wall-clock, **1 manual intervention** (the
  5-seed confirmation, logged with its reason).

## Limitations

**The prompt leaked its own hint.** `losses.py`'s comments (written in an
earlier refactor pass, before the agent existed) named "pairwise (BPR)" and
"listwise" explicitly — the same techniques the project's own README
excludes from the agent's suggested-directions list. Because the agent
prompt includes the full verbatim source of every editable file, this
comment was silently handed to the LLM as a hint rather than something it
derived. Caught with a targeted grep audit across the editable files for
technique names; fixed by rewording the comments to describe only the
mechanism (`uids` lets a loss group examples by user) without naming any
specific technique. Worth an occasional re-audit if editable-file comments
keep evolving.

**A silent batch-composition bug meant two loss experiments tested
nothing.** The original training loop shuffled rows i.i.d. before batching.
With ~1.14M rows spread across ~27,000 train users and an 8,192-row batch,
almost no batch contained more than one row from the same user. Two earlier
iterations that proposed within-user pairwise and listwise losses (which
need multiple rows per user to form a comparison) both silently fell back
to their pointwise path on nearly every batch — they ran, returned metrics,
and got reverted as "not an improvement," but they were never actually
exercising the loss they were supposed to be testing. This went undetected
until the batch composition was checked directly. Fixed by adding a
`config.sampler` seam (`'row'` default, `'user'` groups a shuffled user's
rows contiguously so batches never split a user); verified with a synthetic
test showing within-user row-sharing went from ~28% of rows under `'row'`
to 100% under `'user'`.

**The proposal schema itself blocked a fix for a while.** Early on, a
proposal could touch exactly one file. Adding a new loss to `losses.py`
does nothing by itself — it has to be selected as the default in
`config.py` to ever run. One proposal added `losses.py`'s pairwise loss and
said (in its own reasoning) it intended to activate it via
`sampler='user'`, but since it could only target one file, `config.py`
never changed, and the run trained with the untouched defaults the whole
time — confirmed by the resulting metrics being bit-identical to the
baseline to the last decimal. Fixed by widening the schema to a `changes`
list (up to 3 files per proposal, all-or-nothing allowlist rejection, one
evaluation, one revert) and telling the prompt explicitly that a new
loss/feature must be activated via `config.py` in the same proposal.

**The kept result's margin is real but narrow.** +0.0014 valid / +0.0022
test primary clears the significance bar and held up at 5 seeds, but it is
not a large improvement, and three further refinements of the same pairwise
loss (weighting pairs by nDCG@5 impact, averaging per user instead of per
pair, and temperature-sharpening the pairwise objective) each failed to
improve on it. That specific family — reweighting/resharpening this one
within-user pairwise objective — appears exhausted; further gains likely
need a different approach, not another variant of this one.

**Two smaller things worth knowing about:** convergence detection
(`check_convergence`) looks at the whole run's history, not just the
current invocation's new proposals — resuming a run with a lot of prior
(especially manual) history can trigger "converged" after just one or two
fresh attempts, which is mechanically correct but can feel like it
under-uses a large `--max-iterations` budget. And reasoning models (e.g.
`gpt-5` via `--provider openai`) spend part of their output-token budget on
internal reasoning before the visible answer — too tight a `max_tokens` in
`llm.py` produces a truncated, unparseable response (logged and recovered
from, not a crash, but wasted an iteration) until the budget was raised.
