import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import agent
import example_data
import llm


class ExampleCollectionTests(unittest.TestCase):
    def test_collect_review_and_export_compile_only_example(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_dir = root / 'runs'
            agent_obj = agent.Agent(
                'collect-smoke', run_dir=str(runs_dir), dry_run=True,
                provider='ollama', model='fake', collect_examples=True)
            agent_obj.best_files = {
                name: Path(name).read_text(encoding='utf-8') for name in agent.PIPELINE_FILES
            }
            agent_obj.best_metrics = {
                'valid': {'primary': {'mean': 0.6, 'std': 0.0}},
                'test': {'primary': {'mean': 0.59, 'std': 0.0}},
            }
            agent_obj.best_iteration = None
            proposal = {
                'hypothesis': 'collection smoke test',
                'reasoning': 'exercise the artifact path',
                'changes': [{
                    'target_file': 'config.py',
                    'replacements': [{'old': 'k: int = 16', 'new': 'k: int = 17'}],
                }],
                'expected_effect': 'none; test only',
            }

            original = llm.PROVIDERS['ollama']
            llm.PROVIDERS['ollama'] = lambda prompt, model: (json.dumps(proposal), 5, 7)
            try:
                agent_obj._step()
            finally:
                llm.PROVIDERS['ollama'] = original

            example_dir = runs_dir / 'collect-smoke' / 'examples' / '0001'
            for name in (
                    'prompt.txt', 'raw_response.txt', 'responses.json',
                    'repaired_proposal.json', 'verification.json'):
                self.assertTrue((example_dir / name).is_file(), name)
            verification = json.loads((example_dir / 'verification.json').read_text())
            self.assertTrue(verification['automatic']['schema_valid'])
            self.assertTrue(verification['automatic']['replacements_valid'])
            self.assertTrue(verification['automatic']['syntax_valid'])
            self.assertFalse(verification['automatic']['evaluation_completed'])

            example_data.cmd_review(SimpleNamespace(
                example=str(example_dir), label='approved', notes='reviewed test fixture',
                allow_compile_only=True))
            output_dir = root / 'dataset'
            example_data.cmd_export(SimpleNamespace(
                runs_dir=str(runs_dir), run_id=['collect-smoke'],
                output_dir=str(output_dir), seed=0, valid_ratio=0.1, test_ratio=0.1,
                allow_compile_only=True))
            rows = [json.loads(line) for line in (output_dir / 'train.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['messages'][1]['role'], 'assistant')
            exported = json.loads(rows[0]['messages'][1]['content'])
            self.assertEqual(exported['hypothesis'], proposal['hypothesis'])

    def test_hundred_examples_split_eighty_ten_ten(self):
        self.assertEqual(example_data._split_counts(100, 0.1, 0.1), (80, 10, 10))

    def test_split_keeps_each_run_in_one_partition(self):
        items = []
        for run in range(10):
            for iteration in range(10):
                digest = f'{run}-{iteration}'
                items.append((digest, ({'messages': []}, {'run_id': f'run-{run}'})))
        split = example_data._split_items_by_run(items, 0.1, 0.1, seed=0)
        self.assertEqual({name: len(values) for name, values in split.items()}, {
            'train': 80, 'valid': 10, 'test': 10,
        })
        memberships = {}
        for split_name, values in split.items():
            for _, (_, metadata) in values:
                memberships.setdefault(metadata['run_id'], set()).add(split_name)
        self.assertTrue(all(len(value) == 1 for value in memberships.values()))

    def test_triage_rejects_non_improvements_without_auto_approval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            examples = root / 'runs' / 'sample' / 'examples'

            def write_example(iteration, kept, *, syntax=True, evaluated=True):
                path = examples / f'{iteration:04d}'
                path.mkdir(parents=True)
                example_data._write_json(path / 'verification.json', {
                    'automatic': {
                        'schema_valid': True,
                        'allowlist_valid': True,
                        'replacements_valid': True,
                        'syntax_valid': syntax,
                        'evaluation_completed': evaluated,
                        'kept': kept,
                    },
                })
                return path

            kept = write_example(1, True)
            reverted = write_example(2, False)
            invalid = write_example(3, False, syntax=False, evaluated=False)
            output = io.StringIO()
            with redirect_stdout(output):
                example_data.cmd_triage(SimpleNamespace(
                    runs_dir=str(root / 'runs'), run_id=['sample'], apply=True,
                    min_valid_primary=example_data.DEFAULT_MIN_VALID_PRIMARY))
            self.assertIn(f'REVIEW: {kept}', output.getvalue())
            self.assertEqual(
                example_data._load_json(kept / 'verification.json')['triage']['decision'],
                'candidate')
            self.assertEqual(
                example_data._load_json(reverted / 'verification.json')['triage']['decision'],
                'rejected')
            self.assertEqual(
                example_data._load_json(invalid / 'verification.json')['triage']['decision'],
                'rejected')
            self.assertNotIn(
                'human_review', example_data._load_json(kept / 'verification.json'))

    def test_triage_queues_example_at_fixed_primary_floor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / 'runs' / 'sample'
            examples = run_dir / 'examples'
            run_dir.mkdir(parents=True)
            example = examples / '0002'
            example.mkdir(parents=True)
            example_data._write_json(example / 'verification.json', {
                'iteration': 2,
                'automatic': {
                    'schema_valid': True,
                    'allowlist_valid': True,
                    'replacements_valid': True,
                    'syntax_valid': True,
                    'evaluation_completed': True,
                    'kept': False,
                },
                'metrics': {'valid': {'primary': {'mean': 0.603600}}},
                'human_review': {'label': 'pending'},
            })
            report = example_data.triage_run(run_dir, apply=True)
            self.assertEqual(report['counts']['candidate'], 1)
            triage = example_data._load_json(example / 'verification.json')['triage']
            self.assertEqual(triage['decision'], 'candidate')
            self.assertEqual(triage['category'], 'above_primary_floor')

    def test_bulk_approval_creates_non_destructive_approved_view(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_dir = root / 'runs'
            run_dir = runs_dir / 'sample'
            example = run_dir / 'examples' / '0002'
            example.mkdir(parents=True)
            example_data._write_json(example / 'verification.json', {
                'run_id': 'sample',
                'iteration': 2,
                'automatic': {
                    'schema_valid': True,
                    'allowlist_valid': True,
                    'replacements_valid': True,
                    'syntax_valid': True,
                    'evaluation_completed': True,
                    'kept': False,
                },
                'metrics': {'valid': {'primary': {'mean': 0.6035}}},
                'human_review': {'label': 'pending'},
            })
            (example / 'prompt.txt').write_text('test prompt', encoding='utf-8')
            example_data._write_json(example / 'repaired_proposal.json', {
                'hypothesis': 'h', 'reasoning': 'r', 'changes': [], 'expected_effect': 'e',
            })

            example_data.cmd_approve_above(SimpleNamespace(
                runs_dir=str(runs_dir), run_id=['sample'], min_valid_primary=0.6035,
                approved_dir=None, target_count=100, notes='threshold review'))

            verification = example_data._load_json(example / 'verification.json')
            self.assertEqual(verification['human_review']['label'], 'approved')
            link = runs_dir / 'approved' / 'sample-0002'
            self.assertTrue(link.is_symlink())
            manifest = example_data._load_json(runs_dir / 'approved' / 'manifest.json')
            self.assertEqual(manifest['approved_count'], 1)
            self.assertEqual(manifest['unique_exportable_count'], 1)
            summary = example_data._load_json(run_dir / 'summary.json')
            self.assertEqual(summary['manual_interventions'], 1)


if __name__ == '__main__':
    unittest.main()
