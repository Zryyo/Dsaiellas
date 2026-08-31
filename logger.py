"""Per-run iteration logging for an automated FM-improvement loop.

RunLogger.log_iteration appends one JSON record per line to
runs/<run_id>/iterations.jsonl and rewrites runs/<run_id>/summary.json after
every call. check_convergence() reads that same record shape to decide when
an agent looping over configs/hypotheses should stop.

Record shape (one line of iterations.jsonl):
    {
      "iteration": int,                  # 1-based, assigned by RunLogger
      "parent_iteration": int | null,    # iteration this one built on. Defaults to
                                          # the immediately preceding iteration (linear
                                          # continuation); pass explicitly when reverting
                                          # and branching from an earlier iteration, so
                                          # the log reads as a search tree, not a flat
                                          # sequence. null only for the first iteration.
      "timestamp": str,                  # UTC ISO-8601
      "hypothesis": str,                 # what this iteration was trying, written
                                          # before the experiment ran
      "notes": str | null,               # post-hoc reflection, written after: what was
                                          # expected vs. what happened, and what that
                                          # implies for the next attempt
      "diff": str,                       # unified diff of changed files, "" if none
      "config": {...},                   # config.Config used, as a dict
      "metrics": {...} | null,           # run_multiseed() output, or null if the
                                          # iteration errored before metrics existed
      "beat_previous_best": bool | null, # is_real() vs. best-so-far valid primary
      "error": str | null,               # exception message
      "traceback": str | null,           # full traceback text
      "handled": str | null,             # how the error was recovered from
      "manual_intervention": bool,
      "provider": str | null,            # LLM provider that produced this proposal
                                          # ("anthropic"/"openai"), null when no LLM
                                          # was involved (e.g. baseline establishment)
      "model": str | null,               # LLM model name, paired with provider
      "tokens_in": int,
      "tokens_out": int,
      "wall_clock_sec": float
    }
"""
import dataclasses
import datetime
import json
import os
import traceback as tb_module

from baseline import is_real

BASELINE_TEST_PRIMARY = 0.5946  # published FM baseline, test split (README)

def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def _to_jsonable(cfg):
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, '__dict__'):
        return dict(vars(cfg))
    return cfg

def load_iterations(run_id, run_dir='runs'):
    """Read back every record logged so far for a run (possibly by a
    previous process) as a list of dicts, oldest first."""
    path = os.path.join(run_dir, run_id, 'iterations.jsonl')
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]

class RunLogger:
    def __init__(self, run_id, run_dir='runs'):
        self.run_id = run_id
        self.dir = os.path.join(run_dir, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.iterations_path = os.path.join(self.dir, 'iterations.jsonl')
        self.summary_path = os.path.join(self.dir, 'summary.json')
        self._history = load_iterations(run_id, run_dir)          # resume if reopened
        self._manual_interventions = sum(1 for r in self._history if r.get('manual_intervention'))
        self._best_valid_primary = None                            # (mean, std); is_real-gated keeps only
        self._best_observed_valid_primary = None                   # (mean, std); highest mean ever seen
        self._best_test_primary = None                             # test primary of the kept best (for summary)
        for r in self._history:
            self._update_best(r.get('metrics'), r.get('beat_previous_best'))

    @property
    def history(self):
        return list(self._history)

    def _update_best(self, metrics, beat_previous_best):
        if not metrics:
            return
        p = metrics.get('valid', {}).get('primary')
        if not p:
            return
        if self._best_observed_valid_primary is None or p['mean'] > self._best_observed_valid_primary[0]:
            self._best_observed_valid_primary = (p['mean'], p['std'])
        if beat_previous_best and (
                self._best_valid_primary is None or p['mean'] > self._best_valid_primary[0]):
            self._best_valid_primary = (p['mean'], p['std'])
            tp = metrics.get('test', {}).get('primary')
            self._best_test_primary = tp['mean'] if tp else None

    def log_iteration(self, hypothesis, diff, cfg, metrics, tokens_in, tokens_out,
                       wall_clock_sec, error=None, traceback=None, handled=None,
                       manual_intervention=False, parent_iteration=None, notes=None,
                       provider=None, model=None):
        """
        hypothesis       : str, what this iteration was trying and why, written
                            before the experiment ran
        diff             : str, unified diff of the files changed this iteration
                            ("" if none)
        cfg              : config.Config (or dict/object) used for this iteration
        metrics          : dict shaped like baseline.run_multiseed()'s return value
                            ({'valid': {'GAUC': {'mean','std'}, 'nDCG@5': {...},
                              'primary': {...}}, 'test': {...same shape...}}),
                            or None if the iteration errored before metrics existed
        error            : exception message, or None
        traceback        : full traceback text, or None
        handled          : how the error was recovered from (e.g. "reverted to
                            last-good config"), or None if nothing needed handling
        parent_iteration : iteration number this one built on. Defaults to the
                            immediately preceding iteration; pass explicitly when
                            reverting and branching from an earlier iteration
                            (e.g. abandoning a diverged run and trying something
                            else from the last-known-good one) so the log records
                            the actual search tree instead of a flat sequence.
        notes            : str, post-hoc reflection written after the result is
                            in — what was expected vs. what happened, and what
                            that implies for the next attempt. This is the single
                            field the Innovation criterion is judged on: what the
                            agent identified as worth trying next, and why.
        provider          : LLM provider name ("anthropic"/"openai"), or None when
                            no LLM was involved in this iteration.
        model             : LLM model name, paired with provider, or None.
        """
        beat_previous_best = None
        if metrics is not None:
            p = metrics['valid']['primary']
            if self._best_valid_primary is None:
                beat_previous_best = True
            else:
                base_mean, base_std = self._best_valid_primary
                beat_previous_best = is_real(p['mean'], p['std'], base_mean, base_std)
            self._update_best(metrics, beat_previous_best)

        iteration = len(self._history) + 1
        if parent_iteration is None:
            parent_iteration = iteration - 1 if self._history else None

        record = {
            'iteration': iteration,
            'parent_iteration': parent_iteration,
            'timestamp': _now_iso(),
            'hypothesis': hypothesis,
            'notes': notes,
            'diff': diff,
            'config': _to_jsonable(cfg),
            'metrics': metrics,
            'beat_previous_best': beat_previous_best,
            'error': error,
            'traceback': traceback,
            'handled': handled,
            'manual_intervention': manual_intervention,
            'provider': provider,
            'model': model,
            'tokens_in': tokens_in,
            'tokens_out': tokens_out,
            'wall_clock_sec': wall_clock_sec,
        }
        with open(self.iterations_path, 'a') as fh:
            fh.write(json.dumps(record) + '\n')
        self._history.append(record)
        if manual_intervention:
            self._manual_interventions += 1
        self._write_summary()
        return record

    def log_error(self, hypothesis, cfg, exc, handled, diff='', tokens_in=0, tokens_out=0,
                  wall_clock_sec=0.0, manual_intervention=False, parent_iteration=None,
                  notes=None, provider=None, model=None):
        """Convenience wrapper: log a failed iteration from a caught exception,
        capturing its traceback automatically."""
        tb_text = ''.join(tb_module.format_exception(type(exc), exc, exc.__traceback__))
        return self.log_iteration(
            hypothesis=hypothesis, diff=diff, cfg=cfg, metrics=None,
            tokens_in=tokens_in, tokens_out=tokens_out, wall_clock_sec=wall_clock_sec,
            error=f'{type(exc).__name__}: {exc}', traceback=tb_text, handled=handled,
            manual_intervention=manual_intervention, parent_iteration=parent_iteration,
            notes=notes, provider=provider, model=model,
        )

    def _write_summary(self):
        total_tokens_in = sum(r.get('tokens_in') or 0 for r in self._history)
        total_tokens_out = sum(r.get('tokens_out') or 0 for r in self._history)

        summary = {
            'run_id': self.run_id,
            'total_iterations': len(self._history),
            'total_tokens_in': total_tokens_in,
            'total_tokens_out': total_tokens_out,
            'total_tokens': total_tokens_in + total_tokens_out,
            'total_wall_clock_sec': sum(r.get('wall_clock_sec') or 0 for r in self._history),
            'best_valid_primary': self._best_valid_primary[0] if self._best_valid_primary else None,
            'best_observed_valid_primary': (
                self._best_observed_valid_primary[0] if self._best_observed_valid_primary else None
            ),
            'baseline_test_primary': BASELINE_TEST_PRIMARY,
            # Delta of the KEPT best state's test primary over the published baseline —
            # not the last-attempted iteration's, which may have been reverted. Reads
            # ~0 when nothing has ever beaten the baseline (best-so-far == baseline).
            'final_delta_over_baseline': (
                self._best_test_primary - BASELINE_TEST_PRIMARY if self._best_test_primary is not None else None
            ),
            'manual_interventions': self._manual_interventions,
        }
        with open(self.summary_path, 'w') as fh:
            json.dump(summary, fh, indent=2)
        return summary

def check_convergence(history, epsilon=0.002, N=3):
    """True once the best valid primary hasn't improved by more than epsilon
    over the last N iterations — i.e. best(last N) - best(everything before
    that) <= epsilon. Mirrors the README's convergence rule (eps=0.002 ~
    2.5 sigma given FM's ~0.0008 seed std, N=3).

    Records with manual_intervention: true are excluded from both windows —
    this measures whether the agent's automated search has plateaued, not
    whether a human's manual verification runs have.

    history: list of iteration records (dicts with a 'metrics' key), e.g.
    RunLogger(...).history or logger.load_iterations(run_id).
    """
    primaries = []
    for rec in history:
        if not isinstance(rec, dict) or rec.get('manual_intervention'):
            continue
        m = rec.get('metrics')
        p = m.get('valid', {}).get('primary') if m else None
        if p:
            primaries.append(p['mean'])
    if len(primaries) <= N:
        return False
    best_before = max(primaries[:-N])
    best_recent = max(primaries[-N:])
    return (best_recent - best_before) <= epsilon

if __name__ == '__main__':
    import time
    from config import Config

    logger = RunLogger('demo')

    # --- fake iteration 1: reproduce the official baseline, succeeds ---
    cfg1 = Config()
    metrics1 = {
        'valid': {'GAUC': {'mean': 0.6672, 'std': 0.0001},
                  'nDCG@5': {'mean': 0.5357, 'std': 0.0004},
                  'primary': {'mean': 0.6014, 'std': 0.0003}},
        'test': {'GAUC': {'mean': 0.6614, 'std': 0.0005},
                 'nDCG@5': {'mean': 0.5285, 'std': 0.0001},
                 'primary': {'mean': 0.5950, 'std': 0.0003}},
    }
    logger.log_iteration(
        hypothesis='Reproduce the official FM baseline (logloss, 5 fields, k=16) as iteration 0.',
        diff='',
        cfg=cfg1,
        metrics=metrics1,
        notes=('Matched the published baseline numbers exactly, as expected — confirms the '
               'refactored train() path is behavior-preserving. No new signal here about '
               'where headroom is; next attempt should actually vary something.'),
        tokens_in=1500, tokens_out=400,
        wall_clock_sec=142.3,
    )

    # --- fake iteration 2: bump the learning rate, training diverges, auto-reverted ---
    cfg2 = dataclasses.replace(cfg1, lr=0.05)
    t0 = time.time()
    try:
        raise FloatingPointError('loss became NaN at epoch 3 (lr=0.05 diverged)')
    except FloatingPointError as exc:
        logger.log_error(
            hypothesis='Try a 50x higher learning rate (lr=0.05) to speed up convergence.',
            cfg=cfg2,
            exc=exc,
            handled='Reverted lr to 0.001 (last-good config) and re-queued the iteration.',
            diff=(
                '--- a/config.py\n+++ b/config.py\n'
                '@@ -8,1 +8,1 @@\n-    lr: float = 0.001\n+    lr: float = 0.05\n'
            ),
            notes=('Expected a higher lr to converge faster; instead it diverged by epoch 3. '
                   'A 50x jump was too aggressive — next attempt should try something an order '
                   'of magnitude smaller (e.g. lr=0.003) rather than another large jump.'),
            tokens_in=1200, tokens_out=250,
            wall_clock_sec=round(time.time() - t0 + 8.1, 1),
            manual_intervention=True,
        )

    # --- fake iteration 3: abandon the diverged lr branch, revert to iteration 1's
    #     config and try widening embedding capacity instead ---
    cfg3 = dataclasses.replace(cfg1, k=32)
    metrics3 = {
        'valid': {'GAUC': {'mean': 0.6601, 'std': 0.0002},
                  'nDCG@5': {'mean': 0.5319, 'std': 0.0005},
                  'primary': {'mean': 0.5960, 'std': 0.0004}},
        'test': {'GAUC': {'mean': 0.6540, 'std': 0.0006},
                 'nDCG@5': {'mean': 0.5250, 'std': 0.0003},
                 'primary': {'mean': 0.5895, 'std': 0.0005}},
    }
    logger.log_iteration(
        hypothesis=('Branch back from iteration 1 (last known-good) rather than continuing '
                    'the diverged lr branch. Try k=32 to test whether embedding capacity, '
                    'not the learning rate, is the bottleneck.'),
        diff=(
            '--- a/config.py\n+++ b/config.py\n'
            '@@ -6,1 +6,1 @@\n-    k: int = 16\n+    k: int = 32\n'
        ),
        cfg=cfg3,
        metrics=metrics3,
        notes=('Expected more capacity to help if the model was underfitting; instead '
               'valid primary dropped from 0.6014 to 0.5960. Capacity is not the bottleneck '
               '(consistent with the k=8/16/32 ablation already in the README) — next '
               'attempts should target the loss function or feature construction instead.'),
        parent_iteration=1,
        tokens_in=1400, tokens_out=380,
        wall_clock_sec=138.9,
    )

    print(f"wrote {logger.iterations_path}")
    print(f"wrote {logger.summary_path}")
    print(f"converged? {check_convergence(logger.history)}")
