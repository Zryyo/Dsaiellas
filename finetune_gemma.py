"""Run a conservative Gemma 4 E2B QLoRA smoke test or pilot."""
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = 'mlx-community/gemma-4-e2b-it-4bit'


def _count_rows(path):
    with path.open(encoding='utf-8') as fh:
        return sum(1 for line in fh if line.strip())


def _replace_section(text, start_marker, end_marker, replacement):
    start = text.find(start_marker)
    end = text.find(end_marker, start + len(start_marker)) if start >= 0 else -1
    if start < 0 or end < 0:
        raise ValueError(f'could not compact prompt section {start_marker!r}')
    return text[:start] + replacement + text[end:]


def _compact_prompt(prompt, proposal):
    """Remove repeated context that is not needed to learn proposal mechanics.

    Editable source and the exact response contract remain verbatim. The frozen
    evaluator source, accumulated history, and global research facts are already
    represented by the active state and do not need to consume training memory
    in every example.
    """
    excerpts = []
    for change in proposal.get('changes', []):
        if not isinstance(change, dict):
            continue
        excerpts.append(f"=== {change.get('target_file', 'unknown')} exact anchors ===")
        for replacement in change.get('replacements', []):
            if isinstance(replacement, dict) and isinstance(replacement.get('old'), str):
                excerpts.append(replacement['old'])
    source_excerpt = (
        'CURRENT PIPELINE SOURCE (only exact regions changed by this approved example):\n\n'
        + '\n\n'.join(excerpts)
        + '\n\n')
    prompt = _replace_section(
        prompt, 'CURRENT PIPELINE SOURCE:', 'RUN HISTORY SO FAR:', source_excerpt)
    prompt = _replace_section(
        prompt, 'RUN HISTORY SO FAR:', 'KNOWN DEAD ENDS',
        'RUN HISTORY SO FAR:\n[omitted from compact fine-tuning input]\n\n')
    prompt = _replace_section(
        prompt, 'KNOWN DEAD ENDS', 'SAMPLER FACTS',
        'KNOWN DEAD ENDS:\n[omitted from compact fine-tuning input]\n\n')
    prompt = _replace_section(
        prompt, 'SAMPLER FACTS', 'CURRENT BEST',
        'SAMPLER FACTS:\n[omitted from compact fine-tuning input]\n\n')
    return prompt


def _prepare_data(source_dir, prepared_dir):
    prepared_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ('train', 'valid', 'test'):
        source_path = source_dir / f'{split}.jsonl'
        rows = []
        for line in source_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row = copy.deepcopy(row)
            proposal = json.loads(row['messages'][1]['content'])
            row['messages'][0]['content'] = _compact_prompt(
                row['messages'][0]['content'], proposal)
            rows.append(row)
        (prepared_dir / f'{split}.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                    for row in rows),
            encoding='utf-8',
        )
        counts[split] = len(rows)

    source_manifest = json.loads(
        (source_dir / 'manifest.json').read_text(encoding='utf-8'))
    manifest = {
        **source_manifest,
        'counts': counts,
        'source_data': str(source_dir),
        'transformation': (
            'Replaced full pipeline source with the exact old-anchor excerpts changed by '
            'each approved proposal; removed repeated run-history, dead-end, and sampler-fact '
            'sections. Preserved active state, proposal contract, and assistant response '
            'verbatim.'),
    }
    (prepared_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'pilot'), default='smoke')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--data-dir', type=Path, default=REPO_DIR / 'finetune-data')
    parser.add_argument('--prepared-data-dir', type=Path,
                        default=REPO_DIR / 'finetune-data-gemma')
    parser.add_argument('--output-dir', type=Path, default=REPO_DIR / 'finetune-output')
    parser.add_argument('--max-seq-length', type=int, default=2048)
    parser.add_argument('--iters', type=int, default=None,
                        help='override 1 smoke step or the pilot default of two passes')
    parser.add_argument('--print-command', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()

    train_path = args.data_dir / 'train.jsonl'
    manifest_path = args.data_dir / 'manifest.json'
    if not train_path.is_file() or not manifest_path.is_file():
        raise SystemExit(
            f'missing exported data under {args.data_dir}; run example_data.py export first')
    train_rows = _count_rows(train_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if train_rows < 1:
        raise SystemExit('training split is empty')

    counts = _prepare_data(args.data_dir, args.prepared_data_dir)
    if args.prepare_only:
        print(f'prepared compact Gemma data at {args.prepared_data_dir}: {counts}')
        return

    if args.iters is not None:
        iters = args.iters
    elif args.mode == 'smoke':
        iters = 1
    else:
        iters = train_rows * 2

    output_path = args.output_dir / args.mode / 'adapters.safetensors'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, '-m', 'mlx_vlm.lora',
        '--model-path', args.model,
        '--dataset', str(args.prepared_data_dir),
        '--split', 'train',
        '--batch-size', '1',
        '--iters', str(iters),
        '--learning-rate', '1e-5',
        '--max-seq-length', str(args.max_seq_length),
        '--grad-checkpoint',
        '--grad-clip', '1.0',
        '--train-on-completions',
        '--lora-rank', '8',
        '--lora-alpha', '16',
        '--steps-per-report', '1' if args.mode == 'smoke' else '5',
        '--steps-per-save', '1' if args.mode == 'smoke' else '20',
        '--steps-per-eval', str(max(iters + 1, 2)),
        '--output-path', str(output_path),
    ]
    print(
        f"mode={args.mode} model={args.model} approved={sum(manifest['counts'].values())} "
        f'train={train_rows} iters={iters} max_seq_length={args.max_seq_length}')
    print(' '.join(command))
    if not args.print_command:
        env = os.environ.copy()
        env.setdefault('HF_HOME', str(REPO_DIR / '.hf-cache'))
        subprocess.run(command, cwd=REPO_DIR, env=env, check=True)


if __name__ == '__main__':
    main()
