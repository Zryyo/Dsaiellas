"""Autonomous FM-improvement loop. Agent._step() calls an LLM (via llm.py's
provider-agnostic propose()) to generate one proposal per iteration from the
full pipeline source + run history; in --dry-run mode (the default) proposals
are generated and logged but never written or evaluated, so the plumbing can
be checked before it's allowed to touch anything.

Never edits the repo in place: all proposals are applied to a scratch copy
of the pipeline under runs/<run_id>/workspace/, evaluated there in a
subprocess, and only kept in the in-memory best-so-far snapshot — the
original .py files in this directory are never touched.
"""
import argparse, ast, dataclasses, difflib, json, math, re, shutil, subprocess, sys, time, traceback
from pathlib import Path

import llm
import config
from config import Config
from logger import RunLogger, check_convergence
from baseline import is_real

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / 'KuaiRand-Pure' / 'data'

# Files copied into the scratch workspace so run_multiseed can execute there.
# evaluate.py is included (baseline.py imports it) but is NOT in the allowlist
# below, so no proposal can ever target it.
PIPELINE_FILES = ('config.py', 'losses.py', 'data.py', 'baseline.py', 'evaluate.py')
EDITABLE_ALLOWLIST = {'data.py', 'losses.py', 'config.py', 'baseline.py'}

RUNNER_NAME = '_agent_runner.py'
RUNNER_SCRIPT = '''\
import json, sys
from data import load
from baseline import run_multiseed

def to_native(obj):
    """run_multiseed's mean/std are numpy scalars (statistics.mean/pstdev
    over numpy floats); json can only serialize native Python types."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    return obj

def main():
    data_dir = sys.argv[1]
    splits = load(data_dir)
    res = run_multiseed(splits, seeds=(0, 1, 2))
    print(json.dumps(to_native(res)))

if __name__ == '__main__':
    main()
'''

SUBPROCESS_TIMEOUT_SEC = 900


class EvalFailure(Exception):
    """Any failure to produce clean metrics: bad exit code, unparseable
    output, or a NaN metric. Carries the subprocess's traceback text."""
    def __init__(self, message, traceback_text=''):
        super().__init__(message)
        self.traceback_text = traceback_text


def _is_allowlisted(target_file):
    """Actual path check, not a comment: reject path separators / traversal
    as well as filenames simply outside the allowlist."""
    return Path(target_file).name == target_file and target_file in EDITABLE_ALLOWLIST


def _find_nan(obj, path='metrics'):
    if isinstance(obj, dict):
        for k, v in obj.items():
            found = _find_nan(v, f'{path}.{k}')
            if found:
                return found
    elif isinstance(obj, float) and math.isnan(obj):
        return path
    return None


def _syntax_errors(files):
    """Return syntax errors without importing or executing proposed code."""
    errors = []
    for name, source in files.items():
        try:
            compile(source, name, 'exec')
        except SyntaxError as e:
            lines = source.splitlines()
            line = lines[e.lineno - 1].strip() if e.lineno and e.lineno <= len(lines) else ''
            errors.append(f'{name}:{e.lineno}:{e.offset}: {e.msg}; source={line!r}')
    return errors


def _describe_exception(e):
    """(message, traceback_text) for either an EvalFailure (subprocess-side
    traceback) or a plain local exception (this process's traceback)."""
    if isinstance(e, EvalFailure):
        return str(e), e.traceback_text
    return f'{type(e).__name__}: {e}', traceback.format_exc()


def _parse_config_fields(source):
    """Extract {field: literal_value} from a config.py's `class Config`
    body via the AST, so a proposal's new_content can be reflected accurately
    in logs without executing untrusted code."""
    tree = ast.parse(source)
    fields = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'Config':
            for stmt in node.body:
                if (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
                        and stmt.value is not None):
                    try:
                        fields[stmt.target.id] = ast.literal_eval(stmt.value)
                    except ValueError:
                        pass
            break
    return fields


def _cfg_from_files(files):
    fields = _parse_config_fields(files.get('config.py', ''))
    merged = dataclasses.asdict(Config())
    merged.update({k: v for k, v in fields.items() if k in merged})
    return Config(**merged)


TOTAL_BUDGET_ITERATIONS = 50   # framing for the prompt only; --max-iterations caps this run

KNOWN_DEAD_ENDS = (
    "- Static features: adding CWM's item-side + user-side feature domains on top of the "
    "5-field baseline produced no validation gain beyond seed noise.\n"
    "- Embedding capacity: k=8/16/32 produced no validation gain -- capacity is not the bottleneck.\n"
    "- Pure user-side first-order (linear) terms contribute exactly zero to ranking quality, "
    "because the task ranks within a user and a term constant within a user cannot change that "
    "user's ranking order."
)

SAMPLER_FACTS = (
    "- config.Config has a 'sampler' field: 'row' (default) or 'user'. 'row' draws batch "
    "rows i.i.d. from the whole train set. 'user' shuffles user order and fills batches by "
    "emitting each user's rows contiguously, so a user's rows are never split across two "
    "batches -- this is what lets a loss that groups by uids actually see multiple rows "
    "per user within a batch. It is editable via config.py like any other field.\n"
    "- Measured: sampler='user' with plain logloss gives valid primary 0.5996 +/- 0.0011, "
    "versus 0.6014 +/- 0.0003 with sampler='row'. The user sampler costs ~0.0018 on its "
    "own -- a real degradation, not noise -- and is ~4x noisier across seeds.\n"
    "- A prior run tested within-user pairwise and listwise losses under sampler='row', "
    "where almost no batch contained multiple rows for any one user. Both losses fell "
    "back to their pointwise path almost every batch and so were never actually "
    "evaluated as intended -- those particular reverted results are not informative "
    "about whether a within-user loss can help. (That question has since been answered "
    "-- see CURRENT BEST below.)"
)

CURRENT_BEST_FACTS = (
    "- The current best-so-far is a within-user pairwise (BPR-style) loss under "
    "sampler='user': valid primary 0.6028 +/- 0.0002, a real gain of +0.0014 over the "
    "0.6014 +/- 0.0003 baseline. Confirmed at 5 seeds (valid 0.6026 +/- 0.0003) -- "
    "consistent with the 3-seed result within noise, not a lucky draw. Its canonical "
    "seed workspace is runs/api-best-seed/.\n"
    "- Three refinements of that pairwise loss were tried on top of it and none improved "
    "on it: weighting pairs by their nDCG@5 impact (LambdaRank-style), averaging the "
    "loss per user instead of per pair, and sharpening it with a temperature to "
    "emphasize hard pairs. That family of refinements (reweighting/resharpening this "
    "same within-user pairwise objective) appears exhausted -- a further gain likely "
    "needs a different idea, not another variant of this one."
)

NOISE_FLOOR_NOTE = (
    "3-seed mean has standard deviation sigma ~= 0.0003 on valid primary. Anything under "
    "+0.001 improvement is not distinguishable from seed noise."
)

ANCHOR_SCORES_NOTE = "Use validation primary for every iterative decision; test is final reporting only."


def _format_history(history):
    if not history:
        return '(no iterations yet)'
    lines = []
    for r in history:
        if r.get('metrics'):
            vp = r['metrics']['valid']['primary']
            outcome = 'KEPT' if r.get('beat_previous_best') else 'reverted'
            metrics_txt = f"valid primary {vp['mean']:.4f}+/-{vp['std']:.4f}"
        else:
            outcome = 'ERROR' if r.get('error') else 'rejected'
            metrics_txt = 'no metrics'
        line = (f"iteration {r['iteration']} (parent {r.get('parent_iteration')}): {outcome} -- "
                f"{r['hypothesis']} [{metrics_txt}]")
        if r.get('error'):
            line += f" error: {r['error']}"
        lines.append(line)
    return '\n'.join(lines)


MAX_CHANGES_PER_PROPOSAL = 3
MAX_REPLACEMENTS_PER_FILE = 4
MAX_REPLACEMENT_REPAIR_ATTEMPTS = 2
MAX_SYNTAX_REPAIR_ATTEMPTS = 2

def _parse_proposal_json(text):
    """Best-effort strict-JSON parse of an LLM response. Returns (dict, None)
    on success or (None, error_message) on any structural problem — never
    raises, so a malformed response is a value the caller can log, not an
    exception that can crash the loop.

    Each change accepts either a complete ``new_content`` file or a compact
    list of exact ``replacements`` with ``old`` and ``new`` strings. Compact
    replacements keep local-model generations short; full content remains
    accepted for compatibility with existing API-model prompts and logs.
    """
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = re.sub(r'^```[a-zA-Z]*\n', '', stripped)
        stripped = re.sub(r'\n```\s*$', '', stripped)
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', stripped, re.DOTALL)
        if not m:
            return None, 'response contained no JSON object'
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e2:
            return None, f'response was not valid JSON: {e2}'

    if not isinstance(obj, dict):
        return None, f'response JSON was a {type(obj).__name__}, expected an object'

    base_required = ('hypothesis', 'reasoning', 'expected_effect')
    missing = [k for k in base_required if k not in obj]
    if missing:
        return None, f'response JSON missing required keys: {missing}'
    non_strings = [k for k in base_required if not isinstance(obj[k], str)]
    if non_strings:
        return None, f'response JSON keys must be strings, got non-strings for: {non_strings}'

    if 'changes' in obj:
        changes = obj['changes']
        if not isinstance(changes, list) or not changes:
            return None, "'changes' must be a non-empty list"
        if len(changes) > MAX_CHANGES_PER_PROPOSAL:
            return None, f"'changes' has {len(changes)} entries, max is {MAX_CHANGES_PER_PROPOSAL}"
        norm = []
        for i, c in enumerate(changes):
            if not isinstance(c, dict):
                return None, f'changes[{i}] is a {type(c).__name__}, expected an object'
            if 'target_file' not in c or not isinstance(c['target_file'], str):
                return None, f'changes[{i}] target_file must be a string'
            has_content = 'new_content' in c
            has_replacements = 'replacements' in c
            if has_content == has_replacements:
                return None, (f'changes[{i}] must contain exactly one of new_content '
                              'or replacements')
            if has_content:
                if not isinstance(c['new_content'], str):
                    return None, f'changes[{i}] new_content must be a string'
                norm.append({'target_file': c['target_file'], 'new_content': c['new_content']})
                continue
            replacements = c['replacements']
            if not isinstance(replacements, list) or not replacements:
                return None, f'changes[{i}] replacements must be a non-empty list'
            if len(replacements) > MAX_REPLACEMENTS_PER_FILE:
                return None, (f'changes[{i}] has {len(replacements)} replacements, max is '
                              f'{MAX_REPLACEMENTS_PER_FILE}')
            normalized_replacements = []
            for j, replacement in enumerate(replacements):
                if not isinstance(replacement, dict):
                    return None, f'changes[{i}].replacements[{j}] must be an object'
                old, new = replacement.get('old'), replacement.get('new')
                if not isinstance(old, str) or not isinstance(new, str) or not old:
                    return None, (f'changes[{i}].replacements[{j}] old/new must be strings '
                                  'and old must be non-empty')
                normalized_replacements.append({'old': old, 'new': new})
            norm.append({'target_file': c['target_file'],
                         'replacements': normalized_replacements})
        targets = [c['target_file'] for c in norm]
        if len(set(targets)) != len(targets):
            return None, f"'changes' lists the same target_file more than once: {targets}"
        obj = dict(obj)
        obj['changes'] = norm
        return obj, None

    if 'target_file' in obj and 'new_content' in obj:
        if not isinstance(obj['target_file'], str) or not isinstance(obj['new_content'], str):
            return None, 'target_file/new_content must be strings'
        obj = dict(obj)
        obj['changes'] = [{'target_file': obj['target_file'], 'new_content': obj['new_content']}]
        return obj, None

    return None, "response JSON missing 'changes' (or legacy target_file/new_content)"


class Agent:
    def __init__(self, run_id, max_iterations=3, run_dir='runs', dry_run=True,
                 provider='anthropic', model=None, seed_from=None, collect_examples=False,
                 ignore_convergence=False):
        llm.check_credentials(provider)   # fail fast at startup, not mid-run
        self.run_id = run_id
        self.max_iterations = max_iterations
        self.dry_run = dry_run
        self.provider = provider
        self.model = model or llm.DEFAULT_MODELS[provider]
        # run_id of another run whose CURRENT workspace files seed this run's
        # starting point on a fresh setup, instead of the pristine repo files
        # (e.g. resuming exploration from a previously-kept result under a
        # new run_id/log, rather than from the original baseline).
        self.seed_from = seed_from
        self.collect_examples = collect_examples
        self.ignore_convergence = ignore_convergence
        self.logger = RunLogger(run_id, run_dir=run_dir)
        self.workspace = REPO_DIR / run_dir / run_id / 'workspace'
        self.tokens_in = 0    # running totals, for the final print; per-call counts are what
        self.tokens_out = 0   # actually get logged per iteration (see _step)
        self.best_files = None       # {filename: content}, best-so-far
        self.best_metrics = None     # run_multiseed() output, best-so-far
        self.best_iteration = None   # RunLogger iteration number of best-so-far
        self._example_trace = None

    # ---------------- workspace management ----------------
    def _setup_workspace(self):
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True)
        source_dir = (REPO_DIR / 'runs' / self.seed_from / 'workspace') if self.seed_from else REPO_DIR
        for name in PIPELINE_FILES:
            shutil.copy2(source_dir / name, self.workspace / name)
        (self.workspace / RUNNER_NAME).write_text(RUNNER_SCRIPT, encoding='utf-8')

    def _read_files(self):
        return {name: (self.workspace / name).read_text(encoding='utf-8') for name in PIPELINE_FILES}

    def _write_files(self, files):
        for name, content in files.items():
            (self.workspace / name).write_text(content, encoding='utf-8')

    # ---------------- optional fine-tuning example collection ----------------
    def _begin_example_collection(self, prompt):
        if not self.collect_examples:
            return
        self._example_trace = {
            'history_len': len(self.logger.history),
            'predicted_iteration': len(self.logger.history) + 1,
            'prompt': prompt,
            'responses': [],
            'final_proposal': None,
            'automatic': {
                'schema_valid': False,
                'allowlist_valid': False,
                'replacements_valid': False,
                'syntax_valid': False,
                'evaluation_attempted': False,
            },
        }

    def _collect_response(self, stage, text, tokens_in, tokens_out):
        if self._example_trace is not None:
            self._example_trace['responses'].append({
                'stage': stage,
                'text': text,
                'tokens_in': tokens_in,
                'tokens_out': tokens_out,
            })

    def _collect_proposal(self, proposal):
        if self._example_trace is not None:
            self._example_trace['final_proposal'] = proposal
            self._example_trace['automatic']['schema_valid'] = True

    def _collect_flag(self, name, value=True):
        if self._example_trace is not None:
            self._example_trace['automatic'][name] = value

    def _finish_example_collection(self):
        trace = self._example_trace
        self._example_trace = None
        if trace is None:
            return

        history = self.logger.history
        record = history[-1] if len(history) > trace['history_len'] else None
        iteration = record['iteration'] if record else trace['predicted_iteration']
        example_dir = Path(self.logger.dir) / 'examples' / f'{iteration:04d}'
        example_dir.mkdir(parents=True, exist_ok=True)
        (example_dir / 'prompt.txt').write_text(trace['prompt'], encoding='utf-8')

        responses = trace['responses']
        raw_response = responses[0]['text'] if responses else ''
        (example_dir / 'raw_response.txt').write_text(raw_response, encoding='utf-8')
        (example_dir / 'responses.json').write_text(
            json.dumps(responses, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

        proposal = trace['final_proposal']
        if proposal is not None:
            (example_dir / 'repaired_proposal.json').write_text(
                json.dumps(proposal, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

        automatic = dict(trace['automatic'])
        automatic['evaluation_completed'] = bool(record and record.get('metrics') is not None)
        automatic['kept'] = record.get('beat_previous_best') if record else None
        verification = {
            'run_id': self.run_id,
            'iteration': iteration,
            'provider': self.provider,
            'model': self.model,
            'automatic': automatic,
            'metrics': record.get('metrics') if record else None,
            'error': record.get('error') if record else 'iteration ended without a log record',
            'handled': record.get('handled') if record else None,
            'human_review': {
                'label': 'pending',
                'notes': '',
            },
        }
        (example_dir / 'verification.json').write_text(
            json.dumps(verification, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    # ---------------- evaluation ----------------
    def _evaluate(self):
        """Run run_multiseed on the current workspace in a subprocess.
        Raises EvalFailure on a bad exit code, unparseable output, timeout,
        or a NaN metric — the only success path is a clean metrics dict."""
        try:
            proc = subprocess.run(
                [sys.executable, str(self.workspace / RUNNER_NAME), str(DATA_DIR)],
                cwd=self.workspace, capture_output=True, text=True,
                timeout=SUBPROCESS_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired as e:
            stderr = e.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode('utf-8', 'replace')
            raise EvalFailure(
                f'evaluation subprocess timed out after {SUBPROCESS_TIMEOUT_SEC}s',
                traceback_text=stderr or '(no output captured before timeout)',
            ) from e

        if proc.returncode != 0:
            raise EvalFailure(
                f'evaluation subprocess exited with code {proc.returncode}',
                traceback_text=proc.stderr or '(no stderr captured)',
            )
        try:
            metrics = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as e:
            raise EvalFailure(
                f'could not parse subprocess output as JSON: {e}',
                traceback_text=f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}',
            ) from e
        bad_path = _find_nan(metrics)
        if bad_path:
            raise EvalFailure(f'NaN metric at {bad_path}', traceback_text=json.dumps(metrics, indent=2))
        return metrics

    # ---------------- iterations ----------------
    def _resume_best(self):
        """The record currently representing best-so-far, per the same
        is_real-gated tracking RunLogger itself did when logging it (the
        highest valid-primary mean among beat_previous_best: true records),
        or None if this run has no metrics-bearing history yet."""
        best = None
        for r in self.logger.history:
            if r.get('metrics') and r.get('beat_previous_best'):
                p = r['metrics']['valid']['primary']['mean']
                if best is None or p > best['metrics']['valid']['primary']['mean']:
                    best = r
        return best

    def establish_baseline(self):
        """On a fresh run_id, actually evaluate the unmodified pipeline. When
        resuming a run_id that already has a logged best-so-far, reuse it
        instead of re-running the (expensive, and redundant) baseline eval —
        every kept/reverted proposal leaves the workspace at exactly the
        best-so-far state, so its files are still right there on disk."""
        resumed = self._resume_best()
        if resumed is not None and self.workspace.exists() and all(
                (self.workspace / name).exists() for name in PIPELINE_FILES):
            self.best_files = self._read_files()
            self.best_metrics = resumed['metrics']
            self.best_iteration = resumed['iteration']
            print(f"[baseline] resumed best-so-far from iteration {self.best_iteration}: "
                  f"valid primary {self.best_metrics['valid']['primary']['mean']:.4f} "
                  f"± {self.best_metrics['valid']['primary']['std']:.4f}")
            return

        self._setup_workspace()
        files = self._read_files()
        cfg_here = _cfg_from_files(files)
        hypothesis = (
            f"Establish best-so-far baseline seeded from run {self.seed_from!r}'s kept "
            f"state (loss={cfg_here.loss!r}, sampler={cfg_here.sampler!r}), not the "
            f"original pipeline defaults."
            if self.seed_from else
            'Establish best-so-far baseline from the unmodified pipeline before evaluating proposals.'
        )
        t0 = time.time()
        try:
            metrics = self._evaluate()
        except Exception as e:
            error_msg, tb_text = _describe_exception(e)
            self.logger.log_iteration(
                hypothesis=hypothesis,
                diff='', cfg=cfg_here, metrics=None,
                error=error_msg, traceback=tb_text,
                handled='Could not establish a baseline; aborting run.',
                tokens_in=self.tokens_in, tokens_out=self.tokens_out,
                wall_clock_sec=round(time.time() - t0, 1),
            )
            raise SystemExit(f'baseline evaluation failed: {error_msg} '
                              f'(see runs/{self.run_id}/iterations.jsonl)')

        record = self.logger.log_iteration(
            hypothesis=hypothesis,
            diff='', cfg=cfg_here, metrics=metrics,
            notes=(f"Seeded from run {self.seed_from!r}'s kept state; no proposal applied yet."
                   if self.seed_from else 'Baseline for comparison; no proposal applied yet.'),
            tokens_in=self.tokens_in, tokens_out=self.tokens_out,
            wall_clock_sec=round(time.time() - t0, 1),
        )
        self.best_files, self.best_metrics, self.best_iteration = files, metrics, record['iteration']
        print(f"[baseline] valid primary {metrics['valid']['primary']['mean']:.4f} "
              f"± {metrics['valid']['primary']['std']:.4f}")

    # ---------------- LLM proposer ----------------
    def _build_prompt(self):
        active_cfg = _cfg_from_files(self.best_files)
        active_state = (
            f"This run is currently evaluating loss={active_cfg.loss!r}, "
            f"sampler={active_cfg.sampler!r}, feature_fn={active_cfg.feature_fn!r}, "
            f"k={active_cfg.k}. This is the actual workspace state; the research-wide "
            "best below may be different."
        )
        files_block = '\n\n'.join(
            f"=== {name} ==={' (fixed — never edit this one)' if name == 'evaluate.py' else ''}\n"
            f"{self.best_files[name]}"
            for name in PIPELINE_FILES
        )
        next_iteration = len(self.logger.history) + 1
        return f"""You are iterating on a Factorization Machine recommender pipeline for the \
KuaiRand-Pure benchmark. Task: within-user ranking. Metrics: GAUC, nDCG@5, primary \
(= mean of the two), each reported as a 3-seed mean +/- pstdev via run_multiseed.

CURRENT PIPELINE SOURCE:

{files_block}

RUN HISTORY SO FAR:
{_format_history(self.logger.history)}

KNOWN DEAD ENDS (already tried — do not repeat these):
{KNOWN_DEAD_ENDS}

SAMPLER FACTS (established, not a suggestion — use as context for reading the history below):
{SAMPLER_FACTS}

CURRENT BEST (established, not a suggestion):
{CURRENT_BEST_FACTS}

ACTIVE RUN STATE:
{active_state}

{NOISE_FLOOR_NOTE}

{ANCHOR_SCORES_NOTE}

BUDGET STATE: this would be iteration {next_iteration} of {TOTAL_BUDGET_ITERATIONS}. \
Convergence is declared once validation primary hasn't improved by more than epsilon=0.002 \
over the last N=3 iterations.

Propose one change to try next. It may touch more than one file (up to 3) when the idea \
genuinely needs it — in particular, a newly added loss or feature function does nothing on \
its own: it must also be activated by name via config.py (setting it as the default) in this \
SAME proposal, or it will never actually run. Each file you touch must be one of: data.py, \
losses.py, config.py, baseline.py (evaluate.py can never be changed). Respond with a single \
strict JSON object and nothing else — no markdown code fences, no prose before or after — \
with exactly these keys:
  "hypothesis": short string, what you are trying and why
  "reasoning": string, your reasoning for why this might move the primary metric
  "changes": a list of 1 to 3 objects, each with:
      "target_file": one of "data.py", "losses.py", "config.py", "baseline.py"
      "replacements": a list of 1 to 4 exact edits, each with:
          "old": the smallest useful exact source substring, which must occur exactly once
          "new": the replacement substring
Use compact replacements rather than repeating complete files. Include enough unchanged context \
in each old string to make it unique. Replacements for one file run sequentially in list order.
  "expected_effect": short string, what you expect to happen to the metrics and why"""

    def _build_syntax_repair_prompt(self, proposal, errors):
        """Ask the same model for one syntax-only correction before evaluation."""
        return f"""Your proposed Python edits failed a non-executing syntax check.
Fix only the syntax errors while preserving the hypothesis and intended algorithm.
The exact `old` strings still refer to the current source and must remain applicable.
Return one strict JSON object in the same proposal schema, with no markdown or prose.

SYNTAX ERRORS:
{chr(10).join(errors)}

ORIGINAL PROPOSAL:
{json.dumps(proposal, ensure_ascii=False)}"""

    def _build_json_repair_prompt(self, text, error):
        """Ask the same model to make one malformed response valid proposal JSON."""
        return f"""Your proposal response could not be parsed as the required JSON schema.
Repair formatting/schema only while preserving the intended hypothesis and code edits.
Return one strict JSON object with exactly hypothesis, reasoning, changes, and
expected_effect. Each change needs target_file plus replacements, where every
replacement has string fields old and new. Do not use markdown or add prose.

PARSE ERROR:
{error}

MALFORMED RESPONSE:
{text}"""

    def _build_replacement_repair_prompt(self, proposal, errors):
        """Ask the same model to correct exact-match anchors against current source."""
        targets = [c['target_file'] for c in proposal['changes'] if _is_allowlisted(c['target_file'])]
        sources = '\n\n'.join(f'=== {name} ===\n{self.best_files[name]}' for name in targets)
        return f"""Your proposal could not be applied because one or more exact `old`
strings did not occur exactly once in the current source. Preserve the hypothesis and
intended algorithm, but correct the replacements. Copy every `old` string verbatim
from CURRENT TARGET SOURCE, keep replacements minimal and unique, and remove no-op
edits. Return one strict JSON proposal object with no markdown or prose.

REPLACEMENT ERRORS:
{chr(10).join(errors)}

ORIGINAL PROPOSAL:
{json.dumps(proposal, ensure_ascii=False)}

CURRENT TARGET SOURCE:
{sources}"""

    def _step(self):
        """One iteration: call the LLM for a proposal, then (in dry-run mode)
        just log it, or (in real mode) apply+evaluate it via _run_proposal.
        Any failure — API error or a malformed response — is caught, logged
        with a traceback, and the loop moves on to the next iteration."""
        prompt = self._build_prompt()
        self._begin_example_collection(prompt)
        try:
            return self._step_with_prompt(prompt)
        finally:
            self._finish_example_collection()

    def _step_with_prompt(self, prompt):
        t0 = time.time()
        try:
            text, tokens_in, tokens_out = llm.propose(prompt, self.model, self.provider)
        except Exception as e:
            wall = round(time.time() - t0, 1)
            error_msg, tb_text = _describe_exception(e)
            self.logger.log_iteration(
                hypothesis='(LLM call failed)', diff='', cfg=_cfg_from_files(self.best_files),
                metrics=None, error=error_msg, traceback=tb_text,
                handled='LLM call failed; no proposal generated this iteration.',
                tokens_in=0, tokens_out=0, wall_clock_sec=wall,
                parent_iteration=self.best_iteration,
                provider=self.provider, model=self.model,
            )
            print(f'[llm-error] {error_msg}')
            return

        self._collect_response('initial', text, tokens_in, tokens_out)
        call_wall = time.time() - t0
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        proposal, err = _parse_proposal_json(text)

        if proposal is None:
            print(f'[json-repair] {err}')
            repair_started = time.time()
            try:
                repaired_text, repair_tokens_in, repair_tokens_out = llm.propose(
                    self._build_json_repair_prompt(text, err), self.model, self.provider)
            except Exception as e:
                repair_wall = time.time() - repair_started
                error_msg, tb_text = _describe_exception(e)
                self.logger.log_iteration(
                    hypothesis='(malformed LLM response)', diff='',
                    cfg=_cfg_from_files(self.best_files), metrics=None,
                    error=f'JSON repair failed: {error_msg}', traceback=tb_text,
                    handled='Proposal discarded after its JSON repair call failed.',
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    wall_clock_sec=round(call_wall + repair_wall, 1),
                    parent_iteration=self.best_iteration,
                    provider=self.provider, model=self.model,
                )
                print(f'[json-repair-error] {error_msg}')
                return

            repair_wall = time.time() - repair_started
            self._collect_response(
                'json_repair', repaired_text, repair_tokens_in, repair_tokens_out)
            self.tokens_in += repair_tokens_in
            self.tokens_out += repair_tokens_out
            tokens_in += repair_tokens_in
            tokens_out += repair_tokens_out
            call_wall += repair_wall
            proposal, repair_err = _parse_proposal_json(repaired_text)
            if proposal is None:
                self.logger.log_iteration(
                    hypothesis='(malformed LLM response)', diff='',
                    cfg=_cfg_from_files(self.best_files), metrics=None,
                    error=f'malformed proposal JSON after one repair: {repair_err}',
                    traceback=repaired_text[:4000],
                    handled='Proposal discarded after one JSON repair attempt.',
                    notes=f'Initial parse error: {err}',
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    wall_clock_sec=round(call_wall, 1),
                    parent_iteration=self.best_iteration,
                    provider=self.provider, model=self.model,
                )
                print(f'[json-repair-rejected] {repair_err}')
                return

        self._run_proposal(proposal, tokens_in=tokens_in, tokens_out=tokens_out,
                            extra_wall_sec=call_wall)

    # ---------------- apply + evaluate a proposal ----------------
    def _run_proposal(self, proposal, tokens_in=0, tokens_out=0, extra_wall_sec=0.0,
                      syntax_repair_attempts=0, replacement_repair_attempts=0):
        self._collect_proposal(proposal)
        changes = proposal['changes']   # canonicalized by _parse_proposal_json: 1-3 entries
        targets = [c['target_file'] for c in changes]

        bad = [t for t in targets if not _is_allowlisted(t)]
        if bad:
            self.logger.log_iteration(
                hypothesis=proposal.get('hypothesis', ''), diff='',
                cfg=_cfg_from_files(self.best_files), metrics=None,
                error='target_file not in editable allowlist',
                traceback=None,
                handled=f'Rejected before any write: {bad!r} not in {sorted(EDITABLE_ALLOWLIST)} '
                        f'(proposal touched {targets!r} — one disallowed file rejects the whole '
                        f'proposal, none of it is applied).',
                notes=f'Reasoning: {proposal.get("reasoning", "")} '
                      f'Expected: {proposal.get("expected_effect", "")}. '
                      f'Actual: proposal rejected, never run.',
                tokens_in=tokens_in, tokens_out=tokens_out, wall_clock_sec=round(extra_wall_sec, 1),
                parent_iteration=self.best_iteration,
                provider=self.provider, model=self.model,
            )
            print(f"[rejected] target_file(s)={bad!r} not in allowlist")
            return

        self._collect_flag('allowlist_valid')

        new_files = {}
        replacement_errors = []
        for change in changes:
            target = change['target_file']
            if 'new_content' in change:
                new_files[target] = change['new_content']
                continue
            content = self.best_files[target]
            for i, replacement in enumerate(change['replacements']):
                occurrences = content.count(replacement['old'])
                if occurrences != 1:
                    replacement_errors.append(
                        f'{target} replacement {i} old string occurs {occurrences} times')
                    break
                content = content.replace(replacement['old'], replacement['new'], 1)
            else:
                new_files[target] = content

        if (replacement_errors
                and replacement_repair_attempts < MAX_REPLACEMENT_REPAIR_ATTEMPTS):
            next_attempt = replacement_repair_attempts + 1
            print(f'[replacement-repair {next_attempt}/{MAX_REPLACEMENT_REPAIR_ATTEMPTS}] '
                  f'{"; ".join(replacement_errors)}')
            repair_started = time.time()
            try:
                text, repair_tokens_in, repair_tokens_out = llm.propose(
                    self._build_replacement_repair_prompt(proposal, replacement_errors),
                    self.model,
                    self.provider,
                )
            except Exception as e:
                repair_wall = time.time() - repair_started
                error_msg, tb_text = _describe_exception(e)
                self.logger.log_iteration(
                    hypothesis=proposal['hypothesis'], diff='',
                    cfg=_cfg_from_files(self.best_files), metrics=None,
                    error=f'replacement repair failed: {error_msg}', traceback=tb_text,
                    handled='Rejected before evaluation; exact replacements were not applicable.',
                    notes=f'Initial replacement errors: {"; ".join(replacement_errors)}',
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    wall_clock_sec=round(extra_wall_sec + repair_wall, 1),
                    parent_iteration=self.best_iteration,
                    provider=self.provider, model=self.model,
                )
                print(f'[replacement-repair-error] {error_msg}')
                return

            repair_wall = time.time() - repair_started
            self._collect_response(
                'replacement_repair', text, repair_tokens_in, repair_tokens_out)
            self.tokens_in += repair_tokens_in
            self.tokens_out += repair_tokens_out
            repaired, repair_error = _parse_proposal_json(text)
            total_tokens_in = tokens_in + repair_tokens_in
            total_tokens_out = tokens_out + repair_tokens_out
            total_wall = extra_wall_sec + repair_wall
            if repaired is None:
                self.logger.log_iteration(
                    hypothesis=proposal['hypothesis'], diff='',
                    cfg=_cfg_from_files(self.best_files), metrics=None,
                    error=f'malformed replacement repair: {repair_error}', traceback=text[:4000],
                    handled='Rejected before evaluation; replacement repair was not valid JSON.',
                    notes=f'Initial replacement errors: {"; ".join(replacement_errors)}',
                    tokens_in=total_tokens_in, tokens_out=total_tokens_out,
                    wall_clock_sec=round(total_wall, 1),
                    parent_iteration=self.best_iteration,
                    provider=self.provider, model=self.model,
                )
                print(f'[replacement-repair-malformed] {repair_error}')
                return
            return self._run_proposal(
                repaired,
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                extra_wall_sec=total_wall,
                syntax_repair_attempts=syntax_repair_attempts,
                replacement_repair_attempts=next_attempt,
            )

        if replacement_errors:
            self.logger.log_iteration(
                hypothesis=proposal['hypothesis'], diff='',
                cfg=_cfg_from_files(self.best_files), metrics=None,
                error='invalid exact replacement', traceback='; '.join(replacement_errors),
                handled='Rejected before any write: exact replacement was not uniquely applicable.',
                notes=f'Reasoning: {proposal["reasoning"]} Expected: {proposal["expected_effect"]}.',
                tokens_in=tokens_in, tokens_out=tokens_out, wall_clock_sec=round(extra_wall_sec, 1),
                parent_iteration=self.best_iteration, provider=self.provider, model=self.model,
            )
            print(f'[rejected] {"; ".join(replacement_errors)}')
            return

        self._collect_flag('replacements_valid')

        syntax_errors = _syntax_errors(new_files)
        if syntax_errors and syntax_repair_attempts < MAX_SYNTAX_REPAIR_ATTEMPTS:
            next_attempt = syntax_repair_attempts + 1
            print(f'[syntax-repair {next_attempt}/{MAX_SYNTAX_REPAIR_ATTEMPTS}] '
                  f'{"; ".join(syntax_errors)}')
            repair_started = time.time()
            try:
                text, repair_tokens_in, repair_tokens_out = llm.propose(
                    self._build_syntax_repair_prompt(proposal, syntax_errors),
                    self.model,
                    self.provider,
                )
            except Exception as e:
                repair_wall = time.time() - repair_started
                error_msg, tb_text = _describe_exception(e)
                self.logger.log_iteration(
                    hypothesis=proposal['hypothesis'], diff='',
                    cfg=_cfg_from_files(self.best_files), metrics=None,
                    error=f'syntax repair failed: {error_msg}', traceback=tb_text,
                    handled='Rejected before evaluation; original proposal had invalid Python.',
                    notes=f'Initial syntax errors: {"; ".join(syntax_errors)}',
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    wall_clock_sec=round(extra_wall_sec + repair_wall, 1),
                    parent_iteration=self.best_iteration,
                    provider=self.provider, model=self.model,
                )
                print(f'[syntax-repair-error] {error_msg}')
                return

            repair_wall = time.time() - repair_started
            self._collect_response('syntax_repair', text, repair_tokens_in, repair_tokens_out)
            self.tokens_in += repair_tokens_in
            self.tokens_out += repair_tokens_out
            repaired, repair_error = _parse_proposal_json(text)
            total_tokens_in = tokens_in + repair_tokens_in
            total_tokens_out = tokens_out + repair_tokens_out
            total_wall = extra_wall_sec + repair_wall
            if repaired is None:
                self.logger.log_iteration(
                    hypothesis=proposal['hypothesis'], diff='',
                    cfg=_cfg_from_files(self.best_files), metrics=None,
                    error=f'malformed syntax repair: {repair_error}', traceback=text[:4000],
                    handled='Rejected before evaluation; syntax repair was not valid proposal JSON.',
                    notes=f'Initial syntax errors: {"; ".join(syntax_errors)}',
                    tokens_in=total_tokens_in, tokens_out=total_tokens_out,
                    wall_clock_sec=round(total_wall, 1),
                    parent_iteration=self.best_iteration,
                    provider=self.provider, model=self.model,
                )
                print(f'[syntax-repair-malformed] {repair_error}')
                return
            return self._run_proposal(
                repaired,
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                extra_wall_sec=total_wall,
                syntax_repair_attempts=next_attempt,
                replacement_repair_attempts=replacement_repair_attempts,
            )

        if syntax_errors:
            self.logger.log_iteration(
                hypothesis=proposal['hypothesis'], diff='',
                cfg=_cfg_from_files(self.best_files), metrics=None,
                error=(f'syntax repair still produced invalid Python after '
                       f'{MAX_SYNTAX_REPAIR_ATTEMPTS} attempts'),
                traceback='; '.join(syntax_errors),
                handled=(f'Rejected before evaluation after '
                         f'{MAX_SYNTAX_REPAIR_ATTEMPTS} syntax-repair attempts.'),
                notes=f'Syntax errors after repair: {"; ".join(syntax_errors)}',
                tokens_in=tokens_in, tokens_out=tokens_out,
                wall_clock_sec=round(extra_wall_sec, 1),
                parent_iteration=self.best_iteration,
                provider=self.provider, model=self.model,
            )
            print(f'[syntax-repair-rejected] {"; ".join(syntax_errors)}')
            return

        self._collect_flag('syntax_valid')

        diff_text = '\n'.join(
            ''.join(difflib.unified_diff(
                self.best_files[c['target_file']].splitlines(keepends=True),
                new_files[c['target_file']].splitlines(keepends=True),
                fromfile=f"a/{c['target_file']}", tofile=f"b/{c['target_file']}",
            ))
            for c in changes
        )

        if self.dry_run:
            self.logger.log_iteration(
                hypothesis=proposal['hypothesis'], diff=diff_text,
                cfg=_cfg_from_files({**self.best_files, **new_files}),
                metrics=None,
                notes=f'Reasoning: {proposal["reasoning"]} Expected: {proposal["expected_effect"]}',
                handled='Dry-run: proposal logged but not written or evaluated.',
                tokens_in=tokens_in, tokens_out=tokens_out, wall_clock_sec=round(extra_wall_sec, 1),
                parent_iteration=self.best_iteration,
                provider=self.provider, model=self.model,
            )
            print(f"[dry-run] {proposal['hypothesis']}")
            return

        # --- real apply + evaluate path: all changed files together, one
        # evaluation, revert all together on failure or non-improvement ---
        snapshot = self._read_files()   # == self.best_files, re-read to be safe
        # A local Ollama model can occupy most available accelerator memory.
        # Release its weights before the FM subprocess starts; the daemon stays
        # running and the next proposal call reloads the model automatically.
        if not llm.release(self.model, self.provider):
            self.logger.log_iteration(
                hypothesis=proposal['hypothesis'], diff=diff_text,
                cfg=_cfg_from_files(snapshot), metrics=None,
                error='local model could not be unloaded before evaluation', traceback=None,
                handled=('Proposal not evaluated: close any interactive `ollama run` chat, '
                         'then retry so the FM has enough memory.'),
                notes=f'Reasoning: {proposal["reasoning"]} Expected: {proposal["expected_effect"]}.',
                tokens_in=tokens_in, tokens_out=tokens_out, wall_clock_sec=round(extra_wall_sec, 1),
                parent_iteration=self.best_iteration, provider=self.provider, model=self.model,
            )
            print('[evaluation skipped] Ollama model is still loaded; close its interactive chat and retry')
            return
        self._write_files(new_files)

        t0 = time.time()
        self._collect_flag('evaluation_attempted')
        try:
            metrics = self._evaluate()
        except Exception as e:
            wall = round(time.time() - t0 + extra_wall_sec, 1)
            error_msg, tb_text = _describe_exception(e)
            self._write_files(snapshot)   # revert every file back to best-so-far
            self.logger.log_iteration(
                hypothesis=proposal['hypothesis'], diff=diff_text,
                cfg=_cfg_from_files(snapshot), metrics=None,
                error=error_msg, traceback=tb_text,
                handled='Reverted working copy to best-so-far snapshot.',
                notes=f'Reasoning: {proposal["reasoning"]} Expected: {proposal["expected_effect"]}. '
                      f'Actual: evaluation failed ({error_msg}).',
                tokens_in=tokens_in, tokens_out=tokens_out, wall_clock_sec=wall,
                parent_iteration=self.best_iteration,
                provider=self.provider, model=self.model,
            )
            print(f"[error] {proposal['hypothesis']} -> {error_msg}")
            return

        wall = round(time.time() - t0 + extra_wall_sec, 1)
        new_p, best_p = metrics['valid']['primary'], self.best_metrics['valid']['primary']
        real = is_real(new_p['mean'], new_p['std'], best_p['mean'], best_p['std'])
        cfg_used = _cfg_from_files({**snapshot, **new_files})

        if real:
            handled = (f'Kept: valid primary {new_p["mean"]:.4f} beats best-so-far '
                       f'{best_p["mean"]:.4f} (is_real).')
        else:
            handled = (f'Reverted: valid primary {new_p["mean"]:.4f} vs best-so-far '
                       f'{best_p["mean"]:.4f} is not a real improvement (is_real=False).')
            self._write_files(snapshot)

        record = self.logger.log_iteration(
            hypothesis=proposal['hypothesis'], diff=diff_text, cfg=cfg_used, metrics=metrics,
            notes=(f'Reasoning: {proposal["reasoning"]} '
                   f'Expected: {proposal["expected_effect"]} '
                   f'Actual: valid primary {new_p["mean"]:.4f} (best-so-far was '
                   f'{best_p["mean"]:.4f}) -> {"kept" if real else "reverted"}.'),
            handled=handled,
            tokens_in=tokens_in, tokens_out=tokens_out, wall_clock_sec=wall,
            parent_iteration=self.best_iteration,
            provider=self.provider, model=self.model,
        )
        if real:
            self.best_files = self._read_files()
            self.best_metrics = metrics
            self.best_iteration = record['iteration']
        print(f"[{'kept' if real else 'reverted'}] {proposal['hypothesis']} "
              f"-> valid primary {new_p['mean']:.4f}")

    def run(self):
        # A previous interactive Ollama chat can leave model weights resident.
        # The baseline does not need an LLM, so free that memory before training.
        if not llm.release(self.model, self.provider):
            raise SystemExit(
                'Ollama model is still loaded. Exit any interactive `ollama run` chat '
                '(/bye or Ctrl-D), then retry; the Ollama server itself should remain running.'
            )
        self.establish_baseline()
        for i in range(self.max_iterations):
            self._step()
            if not self.ignore_convergence and check_convergence(self.logger.history):
                print(f'converged after {i + 1} proposal(s)')
                break
        print(f"done. best-so-far valid primary = {self.best_metrics['valid']['primary']['mean']:.4f} "
              f"(iteration {self.best_iteration}). tokens: {self.tokens_in} in / {self.tokens_out} out.")
        if self.collect_examples:
            # Human approval remains mandatory; this queue removes the need
            # to scan every rejected proposal after a collection batch.
            from example_data import triage_run
            report = triage_run(Path(self.logger.dir), apply=True)
            candidates = [result for result in report['results']
                          if result['decision'] == 'candidate']
            if candidates:
                review_ids = ', '.join(f'{result["iteration"]:04d}' for result in candidates)
                print(f'[example-triage] All other pending examples were auto-rejected. '
                      f'Review examples: {review_ids}.')
            else:
                approved = [result['iteration'] for result in report['reviewed_results']
                            if result['label'] == 'approved']
                if approved:
                    approved_ids = ', '.join(f'{iteration:04d}' for iteration in approved)
                    print(f'[example-triage] All remaining pending examples were auto-rejected. '
                          f'Already approved: {approved_ids}.')
                else:
                    print(f'[example-triage] All pending examples were auto-rejected; '
                          'no examples need review.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-iterations', type=int, default=3)
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--dry-run', action=argparse.BooleanOptionalAction, default=True,
                    help='generate and log proposals without writing or evaluating them (default: on)')
    ap.add_argument('--provider', choices=sorted(llm.PROVIDERS), default='anthropic')
    ap.add_argument('--model', default=None,
                    help=f'defaults per --provider: {llm.DEFAULT_MODELS}')
    ap.add_argument('--seed-from', default=None,
                    help="run_id whose current best-so-far workspace seeds this run's "
                         "starting files on a fresh setup, instead of the pristine repo "
                         "defaults (only takes effect for a run_id with no prior history)")
    ap.add_argument('--collect-examples', action='store_true',
                    help='save prompt/response/proposal/verification artifacts for fine-tuning')
    ap.add_argument('--ignore-convergence', action='store_true',
                    help='run all requested iterations even when the research plateau rule fires')
    a = ap.parse_args()
    run_id = a.run_id or f'agent-{time.strftime("%Y%m%dT%H%M%S")}'
    try:
        agent_obj = Agent(run_id, max_iterations=a.max_iterations, dry_run=a.dry_run,
                          provider=a.provider, model=a.model, seed_from=a.seed_from,
                          collect_examples=a.collect_examples,
                          ignore_convergence=a.ignore_convergence)
    except llm.MissingAPIKeyError as e:
        raise SystemExit(f'startup error: {e}')
    agent_obj.run()
