"""Review collected agent examples and export an MLX-LM chat dataset.

Examples are created by ``agent.py --collect-examples`` under
``runs/<run-id>/examples/<iteration>/``. This tool keeps human approval
separate from automatic schema/apply/compile/evaluation checks.
"""
import argparse
import datetime
import hashlib
import json
import os
import random
from pathlib import Path


REVIEW_LABELS = ('pending', 'approved', 'rejected')
TRIAGE_DECISIONS = ('candidate', 'rejected', 'needs_review')
DEFAULT_MIN_VALID_PRIMARY = 0.6035


def _verification_path(path):
    path = Path(path)
    return path if path.name == 'verification.json' else path / 'verification.json'


def _load_json(path):
    with Path(path).open(encoding='utf-8') as fh:
        return json.load(fh)


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _find_example_dirs(runs_dir, run_ids=None):
    runs_dir = Path(runs_dir)
    run_dirs = [runs_dir / run_id for run_id in run_ids] if run_ids else sorted(runs_dir.glob('*'))
    found = []
    for run_dir in run_dirs:
        examples_dir = run_dir / 'examples'
        if not examples_dir.is_dir():
            continue
        found.extend(
            path.parent for path in sorted(examples_dir.glob('*/verification.json')))
    return found


def _automatic_summary(verification):
    automatic = verification.get('automatic', {})
    checks = ('schema_valid', 'allowlist_valid', 'replacements_valid', 'syntax_valid')
    basic_valid = all(automatic.get(name) is True for name in checks)
    evaluated = automatic.get('evaluation_completed') is True
    return basic_valid, evaluated


def _valid_primary(metrics):
    value = (metrics or {}).get('valid', {}).get('primary', {}).get('mean')
    return float(value) if isinstance(value, (int, float)) else None


def _triage_decision(verification, min_valid_primary=DEFAULT_MIN_VALID_PRIMARY):
    """Classify an example without substituting for human approval.

    A kept evaluation is a review candidate. Everything else is unsuitable
    for the default, performance-focused fine-tuning corpus. The human-review
    label remains authoritative and is never changed by this helper.
    """
    basic_valid, evaluated = _automatic_summary(verification)
    automatic = verification.get('automatic', {})
    if not basic_valid:
        return ('rejected', 'automatic schema, allowlist, replacement, or syntax check failed',
                'invalid', None)
    if not evaluated:
        return 'rejected', 'evaluation did not complete', 'unevaluated', None
    if automatic.get('kept') is True:
        return ('candidate', 'passed the is_real() significance gate; human review required',
                'verified_improvement', None)
    if automatic.get('kept') is False:
        primary = _valid_primary(verification.get('metrics'))
        if primary is not None and primary >= min_valid_primary:
            return ('candidate',
                    f'valid primary {primary:.6f} meets the review floor {min_valid_primary:.6f}',
                    'above_primary_floor', None)
        return ('rejected', 'evaluated but did not pass the is_real() significance gate',
                'insignificant', None)
    return ('needs_review', 'evaluation completed but no kept/reverted outcome was recorded',
            'missing_outcome', None)


def _human_label(verification):
    return verification.get('human_review', {}).get('label', 'pending')


def _stored_triage(verification):
    return verification.get('triage', {}).get('decision', 'untriaged')


def _record_manual_intervention(run_dir, hypothesis, notes):
    """Record a human corpus-curation decision without treating it as an experiment."""
    # Import lazily so ordinary listing and export do not need to initialize the
    # full training stack.
    from logger import RunLogger

    run_dir = Path(run_dir)
    logger = RunLogger(run_dir.name, run_dir=str(run_dir.parent))
    previous = logger.history[-1] if logger.history else {}
    logger.log_iteration(
        hypothesis=hypothesis,
        diff='',
        cfg=previous.get('config', {}),
        metrics=None,
        tokens_in=0,
        tokens_out=0,
        wall_clock_sec=0.0,
        manual_intervention=True,
        notes=notes,
        provider=None,
        model=None,
    )


def _set_human_review(verification_path, label, notes):
    verification = _load_json(verification_path)
    verification['human_review'] = {
        'label': label,
        'notes': notes or '',
        'reviewed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _write_json(verification_path, verification)


def sync_approved_folder(runs_dir, output_dir=None):
    """Create a navigable, non-destructive view of all approved examples.

    The source evidence remains under each run's ``examples/`` directory.  The
    approved directory contains relative symlinks so one example cannot become
    detached from its prompt, response, verification metadata, or run history.
    """
    runs_dir = Path(runs_dir)
    output_dir = Path(output_dir) if output_dir else runs_dir / 'approved'
    output_dir.mkdir(parents=True, exist_ok=True)
    approved = []
    unique_exportable = {}
    for example_dir in _find_example_dirs(runs_dir):
        verification = _load_json(example_dir / 'verification.json')
        if _human_label(verification) != 'approved':
            continue
        run_id = example_dir.parent.parent.name
        iteration = verification.get('iteration')
        iteration_label = f'{iteration:04d}' if isinstance(iteration, int) else example_dir.name
        link_path = output_dir / f'{run_id}-{iteration_label}'
        if link_path.exists() or link_path.is_symlink():
            if not link_path.is_symlink():
                raise SystemExit(f'cannot create approved view: {link_path} already exists and is not a symlink')
        else:
            target = os.path.relpath(example_dir, output_dir)
            link_path.symlink_to(target, target_is_directory=True)
        approved.append({
            'run_id': run_id,
            'iteration': iteration,
            'source': str(example_dir),
            'link': str(link_path),
        })
        exportable, _ = _exportable_example(example_dir)
        if exportable is not None:
            unique_exportable.setdefault(exportable[0], str(example_dir))

    manifest = {
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'approved_count': len(approved),
        'unique_exportable_count': len(unique_exportable),
        'examples': approved,
    }
    _write_json(output_dir / 'manifest.json', manifest)
    return output_dir, manifest


def cmd_list(args):
    examples = _find_example_dirs(args.runs_dir, args.run_id)
    shown = 0
    for example_dir in examples:
        verification = _load_json(example_dir / 'verification.json')
        label = _human_label(verification)
        if args.label and label != args.label:
            continue
        triage = _stored_triage(verification)
        if args.triage and triage != args.triage:
            continue
        basic_valid, evaluated = _automatic_summary(verification)
        metric = verification.get('metrics') or {}
        primary = metric.get('valid', {}).get('primary', {}).get('mean')
        score = f'{primary:.6f}' if isinstance(primary, (int, float)) else '-'
        print(
            f'{example_dir}  label={label:<8} triage={triage:<12} valid={str(basic_valid):<5} '
            f'evaluated={str(evaluated):<5} valid_primary={score}'
        )
        shown += 1
    print(f'{shown} example(s)')


def triage_run(run_dir, apply=False, min_valid_primary=DEFAULT_MIN_VALID_PRIMARY):
    """Triage one run without substituting for human approval."""
    run_dir = Path(run_dir)
    examples = _find_example_dirs(run_dir.parent, [run_dir.name])
    counts = {decision: 0 for decision in TRIAGE_DECISIONS}
    reviewed = changed = 0
    results = []
    reviewed_results = []
    for example_dir in examples:
        verification_path = example_dir / 'verification.json'
        verification = _load_json(verification_path)
        label = _human_label(verification)
        if label != 'pending':
            reviewed += 1
            iteration = verification.get('iteration')
            reviewed_results.append({
                'iteration': iteration if isinstance(iteration, int) else -1,
                'label': label,
            })
            continue
        iteration = verification.get('iteration')
        decision, reason, category, delta = _triage_decision(
            verification, min_valid_primary)
        counts[decision] += 1
        result = {
            'example_dir': example_dir,
            'iteration': iteration if isinstance(iteration, int) else -1,
            'decision': decision,
            'reason': reason,
            'category': category,
            'primary': _valid_primary(verification.get('metrics')),
            'delta': delta,
        }
        results.append(result)
        if apply:
            existing = verification.get('triage', {})
            triage = {
                'decision': decision,
                'reason': reason,
                'category': category,
                'valid_primary': result['primary'],
                'minimum_valid_primary': min_valid_primary,
                'automated': True,
            }
            if any(existing.get(key) != value for key, value in triage.items()):
                triage['triaged_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                verification['triage'] = triage
                _write_json(verification_path, verification)
                changed += 1

    return {
        'counts': counts,
        'reviewed': reviewed,
        'changed': changed,
        'results': results,
        'reviewed_results': reviewed_results,
    }


def cmd_triage(args):
    """Report or save conservative automatic fine-tuning triage decisions."""
    runs_dir = Path(args.runs_dir)
    run_ids = args.run_id or [path.name for path in sorted(runs_dir.glob('*')) if path.is_dir()]
    totals = {decision: 0 for decision in TRIAGE_DECISIONS}
    reviewed = changed = 0
    candidates = []
    for run_id in run_ids:
        report = triage_run(runs_dir / run_id, args.apply, args.min_valid_primary)
        for decision, count in report['counts'].items():
            totals[decision] += count
        reviewed += report['reviewed']
        changed += report['changed']
        candidates.extend(result for result in report['results']
                          if result['decision'] == 'candidate')

    for result in candidates:
        print(f"REVIEW: {result['example_dir']}  {result['reason']}")
    print(
        'triage: '
        f"candidates={totals['candidate']} rejected={totals['rejected']} "
        f"needs_review={totals['needs_review']} already_human_reviewed={reviewed}"
    )
    if args.apply:
        print(f'saved {changed} automatic triage decision(s); no examples were approved')
    else:
        print('dry run only; rerun with --apply to save decisions')


def cmd_review(args):
    verification_path = _verification_path(args.example)
    if not verification_path.is_file():
        raise SystemExit(f'not a collected example: {verification_path}')
    verification = _load_json(verification_path)
    basic_valid, evaluated = _automatic_summary(verification)
    if args.label == 'approved':
        if not basic_valid:
            raise SystemExit(
                'cannot approve: schema/allowlist/replacement/syntax checks did not all pass')
        if not evaluated and not args.allow_compile_only:
            raise SystemExit(
                'cannot approve: evaluation did not complete; pass --allow-compile-only '
                'only after manually verifying the proposal')
        example_dir = verification_path.parent
        for name in ('prompt.txt', 'repaired_proposal.json'):
            if not (example_dir / name).is_file():
                raise SystemExit(f'cannot approve: missing {example_dir / name}')

    _set_human_review(verification_path, args.label, args.notes)
    example_dir = verification_path.parent
    _record_manual_intervention(
        example_dir.parent.parent,
        f"Manual fine-tuning example review: marked example {example_dir.name} as {args.label}.",
        args.notes or 'Human review label changed; no model code, configuration, or metrics were changed.',
    )
    approved_dir, manifest = sync_approved_folder(example_dir.parent.parent.parent)
    print(f'{verification_path.parent}: {args.label}')
    print(f'approved view: {approved_dir} ({manifest["approved_count"]} approved example(s))')


def cmd_approve_above(args):
    """Apply one explicit human approval decision to a score-qualified batch."""
    run_ids = args.run_id
    approved_by_run = {}
    skipped = 0
    for example_dir in _find_example_dirs(args.runs_dir, run_ids):
        verification_path = example_dir / 'verification.json'
        verification = _load_json(verification_path)
        if _human_label(verification) != 'pending':
            skipped += 1
            continue
        basic_valid, evaluated = _automatic_summary(verification)
        primary = _valid_primary(verification.get('metrics'))
        if not (basic_valid and evaluated and primary is not None and
                primary >= args.min_valid_primary):
            skipped += 1
            continue
        note = args.notes or (
            'User-approved batch: fully evaluated with valid primary '
            f'>= {args.min_valid_primary:.4f}.')
        _set_human_review(verification_path, 'approved', note)
        run_id = example_dir.parent.parent.name
        approved_by_run.setdefault(run_id, []).append((example_dir.name, primary))

    runs_dir = Path(args.runs_dir)
    for run_id, items in approved_by_run.items():
        example_ids = ', '.join(name for name, _ in items)
        _record_manual_intervention(
            runs_dir / run_id,
            'Manual fine-tuning corpus approval; not an FM experiment.',
            f'User approved examples {example_ids} because each completed all automatic '
            f'checks and had valid primary >= {args.min_valid_primary:.4f}. '
            'No model code, configuration, or metrics were changed.',
        )

    approved_dir, manifest = sync_approved_folder(runs_dir, args.approved_dir)
    approved_count = sum(len(items) for items in approved_by_run.values())
    remaining = max(0, args.target_count - manifest['unique_exportable_count'])
    print(f'approved {approved_count} pending example(s); skipped {skipped}')
    print(
        f'approved view: {approved_dir} contains {manifest["approved_count"]} approved '
        f'example(s), {manifest["unique_exportable_count"]} unique exportable example(s).')
    print(f'{remaining} more unique approved example(s) are needed to reach {args.target_count}.')


def _exportable_example(example_dir, allow_compile_only=False):
    verification = _load_json(example_dir / 'verification.json')
    if verification.get('human_review', {}).get('label') != 'approved':
        return None, 'not approved'
    basic_valid, evaluated = _automatic_summary(verification)
    if not basic_valid:
        return None, 'automatic checks incomplete'
    if not evaluated and not allow_compile_only:
        return None, 'evaluation incomplete'

    prompt_path = example_dir / 'prompt.txt'
    proposal_path = example_dir / 'repaired_proposal.json'
    if not prompt_path.is_file() or not proposal_path.is_file():
        return None, 'missing prompt or proposal'
    prompt = prompt_path.read_text(encoding='utf-8')
    proposal = _load_json(proposal_path)
    required = ('hypothesis', 'reasoning', 'changes', 'expected_effect')
    if not isinstance(proposal, dict) or any(key not in proposal for key in required):
        return None, 'proposal schema incomplete'

    assistant = json.dumps(proposal, ensure_ascii=False, separators=(',', ':'))
    digest = hashlib.sha256((prompt + '\0' + assistant).encode('utf-8')).hexdigest()
    row = {
        'messages': [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': assistant},
        ]
    }
    metadata = {
        'id': digest,
        'source': str(example_dir),
        'run_id': verification.get('run_id'),
        'iteration': verification.get('iteration'),
        'provider': verification.get('provider'),
        'model': verification.get('model'),
        'metrics': verification.get('metrics'),
        'human_review': verification.get('human_review'),
    }
    return (digest, row, metadata), None


def _split_counts(total, valid_ratio, test_ratio):
    if total <= 1:
        return total, 0, 0
    if total == 2:
        return 1, 1, 0
    valid = max(1, round(total * valid_ratio))
    test = max(1, round(total * test_ratio))
    if valid + test >= total:
        overflow = valid + test - (total - 1)
        while overflow and (valid > 1 or test > 1):
            if test >= valid and test > 1:
                test -= 1
            elif valid > 1:
                valid -= 1
            overflow -= 1
    return total - valid - test, valid, test


def _split_items_by_run(items, valid_ratio, test_ratio, seed):
    """Keep examples from one evolving run together to prevent prompt leakage."""
    groups = {}
    for item in items:
        digest, (_, metadata) = item
        group = metadata.get('run_id') or digest
        groups.setdefault(group, []).append(item)
    group_names = list(groups)
    random.Random(seed).shuffle(group_names)
    train_count, valid_count, test_count = _split_counts(
        len(group_names), valid_ratio, test_ratio)
    split_groups = {
        'train': group_names[:train_count],
        'valid': group_names[train_count:train_count + valid_count],
        'test': group_names[train_count + valid_count:train_count + valid_count + test_count],
    }
    return {
        split: [item for group in names for item in groups[group]]
        for split, names in split_groups.items()
    }


def _write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def cmd_export(args):
    examples = _find_example_dirs(args.runs_dir, args.run_id)
    unique = {}
    skipped = []
    for example_dir in examples:
        item, reason = _exportable_example(example_dir, args.allow_compile_only)
        if item is None:
            skipped.append((str(example_dir), reason))
            continue
        digest, row, metadata = item
        unique.setdefault(digest, (row, metadata))

    split_items = _split_items_by_run(
        list(unique.items()), args.valid_ratio, args.test_ratio, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_examples = []
    for split, values in split_items.items():
        _write_jsonl(output_dir / f'{split}.jsonl', [value[1][0] for value in values])
        for digest, (_, metadata) in values:
            manifest_examples.append({**metadata, 'id': digest, 'split': split})

    manifest = {
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'seed': args.seed,
        'split_unit': 'run_id',
        'counts': {split: len(values) for split, values in split_items.items()},
        'allow_compile_only': args.allow_compile_only,
        'examples': manifest_examples,
        'skipped_count': len(skipped),
    }
    _write_json(output_dir / 'manifest.json', manifest)
    counts = manifest['counts']
    print(
        f"exported {sum(counts.values())} unique approved example(s): "
        f"train={counts['train']} valid={counts['valid']} test={counts['test']}"
    )
    print(f'skipped {len(skipped)} example(s); manifest: {output_dir / "manifest.json"}')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    list_parser = sub.add_parser('list', help='list collected examples and verification status')
    list_parser.add_argument('--runs-dir', default='runs')
    list_parser.add_argument('--run-id', action='append', help='limit to a run; repeatable')
    list_parser.add_argument('--label', choices=REVIEW_LABELS)
    list_parser.add_argument('--triage', choices=('untriaged',) + TRIAGE_DECISIONS)
    list_parser.set_defaults(func=cmd_list)

    triage_parser = sub.add_parser(
        'triage', help='flag kept results for review and auto-reject non-improvements')
    triage_parser.add_argument('--runs-dir', default='runs')
    triage_parser.add_argument('--run-id', action='append', help='limit to a run; repeatable')
    triage_parser.add_argument(
        '--apply', action='store_true',
        help='save automatic triage metadata; never approves an example')
    triage_parser.add_argument(
        '--min-valid-primary', type=float, default=DEFAULT_MIN_VALID_PRIMARY,
        help=f'queue valid evaluated examples at or above this validation primary (default: {DEFAULT_MIN_VALID_PRIMARY})')
    triage_parser.set_defaults(func=cmd_triage)

    review_parser = sub.add_parser('review', help='set the human-review label for one example')
    review_parser.add_argument('example', help='example directory or verification.json path')
    review_parser.add_argument('--label', required=True, choices=REVIEW_LABELS)
    review_parser.add_argument('--notes', default='')
    review_parser.add_argument('--allow-compile-only', action='store_true')
    review_parser.set_defaults(func=cmd_review)

    approve_parser = sub.add_parser(
        'approve-above',
        help='record a user-requested batch approval for evaluated examples at a score floor')
    approve_parser.add_argument('--runs-dir', default='runs')
    approve_parser.add_argument('--run-id', action='append', required=True,
                                help='run containing the pending examples to approve; repeatable')
    approve_parser.add_argument('--min-valid-primary', type=float,
                                default=DEFAULT_MIN_VALID_PRIMARY,
                                help=f'approve valid primary at or above this floor (default: {DEFAULT_MIN_VALID_PRIMARY})')
    approve_parser.add_argument('--approved-dir', default=None,
                                help='non-destructive approved-example symlink view (default: RUNS_DIR/approved)')
    approve_parser.add_argument('--target-count', type=int, default=100,
                                help='report remaining unique examples against this target (default: 100)')
    approve_parser.add_argument('--notes', default='', help='human-review note stored with each approval')
    approve_parser.set_defaults(func=cmd_approve_above)

    export_parser = sub.add_parser('export', help='export approved examples as MLX-LM JSONL')
    export_parser.add_argument('--runs-dir', default='runs')
    export_parser.add_argument('--run-id', action='append', help='limit to a run; repeatable')
    export_parser.add_argument('--output-dir', default='finetune-data')
    export_parser.add_argument('--seed', type=int, default=0)
    export_parser.add_argument('--valid-ratio', type=float, default=0.1)
    export_parser.add_argument('--test-ratio', type=float, default=0.1)
    export_parser.add_argument('--allow-compile-only', action='store_true')
    export_parser.set_defaults(func=cmd_export)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, 'valid_ratio', 0) < 0 or getattr(args, 'test_ratio', 0) < 0:
        parser.error('split ratios must be non-negative')
    if getattr(args, 'valid_ratio', 0) + getattr(args, 'test_ratio', 0) >= 1:
        parser.error('valid-ratio + test-ratio must be less than 1')
    if not 0 <= getattr(args, 'min_valid_primary', 0) <= 1:
        parser.error('min-valid-primary must be between 0 and 1')
    if hasattr(args, 'target_count') and args.target_count < 1:
        parser.error('target-count must be at least 1')
    args.func(args)


if __name__ == '__main__':
    main()
