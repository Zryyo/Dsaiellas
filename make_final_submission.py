"""Build the offline KuaiRand-Pure submission from the canonical kept result.

This leaves the root baseline defaults untouched. The validated final settings
from runs/agent-openai-v3 iteration 2 are applied explicitly, then the frozen
submit.py writer and checker preserve the organizer's row-alignment contract.
"""
import argparse
import dataclasses
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

import baseline
import config
from data import encode, load
from evaluate import evaluate
from submit import read_submission, write_submission


REPO_DIR = Path(__file__).resolve().parent
CANONICAL_RUN = REPO_DIR / 'runs' / 'agent-openai-v3' / 'iterations.jsonl'
FINAL_ITERATION = 2
FINAL_OVERRIDES = {'loss': 'pairwise', 'sampler': 'user'}


def _canonical_record():
    records = [json.loads(line) for line in CANONICAL_RUN.read_text().splitlines()
               if line.strip()]
    record = next((row for row in records if row['iteration'] == FINAL_ITERATION), None)
    if record is None or not record.get('beat_previous_best'):
        raise RuntimeError('canonical kept iteration 2 is missing or no longer marked kept')
    for key, expected in FINAL_OVERRIDES.items():
        if record['config'].get(key) != expected:
            raise RuntimeError(
                f'canonical iteration has {key}={record["config"].get(key)!r}, '
                f'expected {expected!r}')
    return record


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=REPO_DIR / 'KuaiRand-Pure' / 'data')
    parser.add_argument('--output', type=Path,
                        default=REPO_DIR / 'submission' / 'kuairand_pure_submission.csv')
    parser.add_argument('--checkpoint', type=Path,
                        default=REPO_DIR / 'submission' / 'kuairand_pure_fm_seed0.npz')
    parser.add_argument('--manifest', type=Path,
                        default=REPO_DIR / 'submission' / 'submission_manifest.json')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    record = _canonical_record()
    final_config = dataclasses.replace(config.DEFAULT, **FINAL_OVERRIDES)
    splits = load(args.data_dir)
    encoded, dimension = encode(splits, feature_fn=final_config.feature_fn)
    model = baseline.train(encoded, dimension, cfg=final_config,
                           seed=args.seed, verbose=True)

    test_rows = splits['test']
    test_scores = model.predict(encoded['test'][0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(args.output, test_rows, test_scores)
    checked_scores = read_submission(args.output, test_rows)
    if len(checked_scores) != len(test_rows):
        raise RuntimeError('submission row count changed during frozen-format validation')

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.checkpoint,
        V=model.V,
        W=model.W,
        b=np.asarray(model.b),
        dimension=np.asarray(dimension),
        seed=np.asarray(args.seed),
        config=np.asarray(json.dumps(dataclasses.asdict(final_config), sort_keys=True)),
    )

    X_valid, y_valid, users_valid = encoded['valid']
    valid_metrics = {
        name: float(value)
        for name, value in evaluate(users_valid, y_valid, model.predict(X_valid)).items()
    }
    manifest = {
        'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_run': 'agent-openai-v3',
        'source_iteration': FINAL_ITERATION,
        'source_iteration_kept': record['beat_previous_best'],
        'source_multiseed_metrics': record['metrics'],
        'config': dataclasses.asdict(final_config),
        'seed': args.seed,
        'single_checkpoint_validation_metrics': valid_metrics,
        'submission': {
            'path': str(args.output.relative_to(REPO_DIR)),
            'rows': len(test_rows),
            'sha256': _sha256(args.output),
            'schema': 'row_id,user_id,video_id,score',
        },
        'checkpoint': {
            'path': str(args.checkpoint.relative_to(REPO_DIR)),
            'sha256': _sha256(args.checkpoint),
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'validated submission: {args.output} ({len(test_rows):,} rows)')
    print(f'checkpoint: {args.checkpoint}')
    print(f'manifest: {args.manifest}')
    print('single-checkpoint valid: ' + ' | '.join(
        f'{name} {valid_metrics[name]:.6f}' for name in ('GAUC', 'nDCG@5', 'primary')))


if __name__ == '__main__':
    main()
