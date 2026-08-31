import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import llm


class MLXProviderTests(unittest.TestCase):
    def test_plain_checkpoint_has_no_adapter(self):
        self.assertEqual(
            llm._split_mlx_model_spec('mlx-community/gemma-4-e2b-it-4bit'),
            ('mlx-community/gemma-4-e2b-it-4bit', None),
        )

    def test_adapter_spec_is_sent_to_worker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir)
            (adapter / 'adapter_config.json').write_text('{}', encoding='utf-8')
            (adapter / 'adapters.safetensors').write_bytes(b'test')
            completed = SimpleNamespace(
                returncode=0,
                stdout=json.dumps({
                    'text': '{"hypothesis":"test"}',
                    'prompt_tokens': 123,
                    'generation_tokens': 45,
                }),
                stderr='',
            )
            with patch.object(llm.subprocess, 'run', return_value=completed) as run:
                result = llm._propose_mlx(
                    'prompt', f'mlx-community/gemma-4-e2b-it-4bit::{adapter}')

            self.assertEqual(result, ('{"hypothesis":"test"}', 123, 45))
            request = json.loads(run.call_args.kwargs['input'])
            self.assertEqual(request['model'], 'mlx-community/gemma-4-e2b-it-4bit')
            self.assertEqual(request['adapter_path'], str(adapter.resolve()))
            self.assertEqual(request['prompt'], 'prompt')

    def test_missing_adapter_is_rejected_before_generation(self):
        with self.assertRaises(FileNotFoundError):
            llm._split_mlx_model_spec(
                'mlx-community/gemma-4-e2b-it-4bit::does-not-exist')


if __name__ == '__main__':
    unittest.main()
