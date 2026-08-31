"""Provider-agnostic LLM call for agent.py's proposer.

The interface is `propose(prompt, model) -> (text, tokens_in, tokens_out)`,
implemented once per provider (_propose_anthropic, _propose_openai,
_propose_ollama, _propose_mlx) so
agent.py's loop, _parse_proposal_json, and token accounting never need to
know which provider produced a response — they only see the 3-tuple.

Each SDK is imported lazily inside its own implementation function.  The
Ollama implementation uses only the standard library, so it runs locally
without an API key or any extra Python package.
"""
from dotenv import load_dotenv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Load this repository's local credentials for CLI and IDE runs. Environment
# variables already supplied by the shell take precedence.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'), override=False)

# Generous headroom: a reasoning model (e.g. gpt-5) spends part of this budget
# on internal reasoning before emitting the visible answer, so an 8192 cap
# was observed to truncate mid-reasoning with zero content left for the
# actual JSON response on some calls.
MAX_TOKENS = 16000

DEFAULT_MODELS = {
    'anthropic': 'claude-opus-5',
    # Not pinned by any live catalog this module can check — override with
    # --model if you want a specific/newer OpenAI model.
    'openai': 'gpt-5',
    'ollama': 'gemma4:e4b-mlx',
    'mlx': 'mlx-community/gemma-4-e2b-it-4bit',
}

_ENV_VARS = {
    'anthropic': 'ANTHROPIC_API_KEY',
    'openai': 'OPENAI_API_KEY',
}

_RETRYABLE_EXCEPTION_NAMES = {
    'RateLimitError', 'APIConnectionError', 'APITimeoutError',
    'Timeout', 'ConnectionError', 'InternalServerError',
}


class MissingAPIKeyError(RuntimeError):
    """Raised by check_credentials() at startup — never mid-run — when the
    selected provider's API key environment variable isn't set."""


def check_credentials(provider):
    """Call once before the run loop starts, so a missing key is a clear
    startup error instead of surfacing deep inside the first iteration."""
    if provider in ('ollama', 'mlx'):
        return
    env_var = _ENV_VARS.get(provider)
    if env_var is None:
        raise ValueError(f'unknown provider {provider!r}, must be one of {sorted(_ENV_VARS)}')
    if not os.environ.get(env_var):
        raise MissingAPIKeyError(
            f'{env_var} is not set. Set it before running with --provider {provider}.'
        )


def _propose_anthropic(prompt, model):
    import anthropic
    client = anthropic.Anthropic()   # resolves ANTHROPIC_API_KEY internally
    response = client.messages.create(
        model=model, max_tokens=MAX_TOKENS,
        messages=[{'role': 'user', 'content': prompt}],
    )
    text = next((b.text for b in response.content if b.type == 'text'), '')
    return text, response.usage.input_tokens, response.usage.output_tokens


def _propose_openai(prompt, model):
    import openai
    client = openai.OpenAI()   # resolves OPENAI_API_KEY internally
    response = client.chat.completions.create(
        # gpt-5-family / reasoning models reject max_tokens (400
        # unsupported_parameter) and require max_completion_tokens instead.
        model=model, max_completion_tokens=MAX_TOKENS,
        messages=[{'role': 'user', 'content': prompt}],
    )
    text = response.choices[0].message.content or ''
    return text, response.usage.prompt_tokens, response.usage.completion_tokens


def _propose_ollama(prompt, model):
    """Send a proposal request to a local Ollama server.

    Configuration is deliberately environment-based so no local-machine
    settings need to be committed:
      OLLAMA_BASE_URL    server root (default http://127.0.0.1:11434)
      OLLAMA_NUM_CTX     total input + output context (default 12288)
      OLLAMA_MAX_TOKENS  maximum generated tokens (default 3000)
      OLLAMA_KEEP_ALIVE  how long Ollama retains model weights (default 30m)
      OLLAMA_TIMEOUT_SEC HTTP timeout for a slow local generation (default 300)
      OLLAMA_THINK       enable a separate reasoning trace (default false)
    """
    base_url = os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434').rstrip('/')
    try:
        # These defaults fit the full project prompt while leaving headroom
        # for a 24B Q4 model on a 24 GB unified-memory laptop. Increase them
        # only on a machine with more memory.
        num_ctx = int(os.environ.get('OLLAMA_NUM_CTX', '12288'))
        max_tokens = int(os.environ.get('OLLAMA_MAX_TOKENS', '3000'))
        timeout = int(os.environ.get('OLLAMA_TIMEOUT_SEC', '300'))
    except ValueError as e:
        raise ValueError('OLLAMA_NUM_CTX, OLLAMA_MAX_TOKENS, and OLLAMA_TIMEOUT_SEC must be integers') from e

    payload = {
        'model': model,
        'stream': False,
        # Thinking-capable local models can spend the entire output budget on
        # a hidden trace and return no proposal. Keep it off for a comparable
        # fixed-budget benchmark; it can be explicitly enabled when desired.
        'think': os.environ.get('OLLAMA_THINK', 'false').lower() in ('1', 'true', 'yes'),
        # The agent contract always expects one JSON proposal object. Enforce
        # JSON mode at the local API boundary instead of relying on prompting
        # alone; smaller local models otherwise frequently emit invalid JSON.
        'format': 'json',
        'keep_alive': os.environ.get('OLLAMA_KEEP_ALIVE', '30m'),
        'messages': [{'role': 'user', 'content': prompt}],
        'options': {
            'temperature': 0.1,
            'num_ctx': num_ctx,
            'num_predict': max_tokens,
        },
    }
    request = Request(
        f'{base_url}/api/chat', data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST',
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
    except HTTPError as e:
        detail = e.read().decode('utf-8', 'replace')
        raise RuntimeError(f'Ollama returned HTTP {e.code}: {detail}') from e
    except URLError as e:
        raise ConnectionError(
            f'could not reach Ollama at {base_url}; start it with `ollama serve` and check `ollama ps`'
        ) from e

    text = result.get('message', {}).get('content', '')
    if not text:
        raise RuntimeError(f'Ollama returned no message content: {result!r}')
    return text, int(result.get('prompt_eval_count') or 0), int(result.get('eval_count') or 0)


def _split_mlx_model_spec(model):
    """Return a base checkpoint and optional local adapter directory.

    The CLI model form is ``BASE::ADAPTER_DIR``. Relative adapter directories
    resolve from this repository so the exact tuned artifact is retained in
    each run's model provenance without adding another agent CLI argument.
    """
    base_model, separator, adapter = model.partition('::')
    if not base_model.strip():
        raise ValueError('MLX model checkpoint cannot be empty')
    if not separator:
        return base_model, None
    if not adapter.strip():
        raise ValueError('MLX adapter directory cannot be empty after `::`')
    adapter_path = Path(adapter).expanduser()
    if not adapter_path.is_absolute():
        adapter_path = Path(__file__).resolve().parent / adapter_path
    adapter_path = adapter_path.resolve()
    if not (adapter_path / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'MLX adapter config not found under {adapter_path}')
    if not (adapter_path / 'adapters.safetensors').is_file():
        raise FileNotFoundError(f'MLX adapter weights not found under {adapter_path}')
    return base_model, str(adapter_path)


def _propose_mlx(prompt, model):
    """Generate once with MLX-VLM, then exit the worker to release memory.

    Configuration:
      MLX_MAX_TOKENS   maximum generated tokens (default 3000)
      MLX_TIMEOUT_SEC  whole worker timeout including model load (default 300)
      MLX_SEED         deterministic sampling seed (default 0)
    """
    try:
        max_tokens = int(os.environ.get('MLX_MAX_TOKENS', '3000'))
        timeout = int(os.environ.get('MLX_TIMEOUT_SEC', '300'))
        seed = int(os.environ.get('MLX_SEED', '0'))
    except ValueError as e:
        raise ValueError('MLX_MAX_TOKENS, MLX_TIMEOUT_SEC, and MLX_SEED must be integers') from e
    if max_tokens < 1 or timeout < 1:
        raise ValueError('MLX_MAX_TOKENS and MLX_TIMEOUT_SEC must be positive')

    base_model, adapter_path = _split_mlx_model_spec(model)
    request = {
        'model': base_model,
        'adapter_path': adapter_path,
        'prompt': prompt,
        'max_tokens': max_tokens,
        'temperature': 0.1,
        'top_p': 0.95,
        'seed': seed,
    }
    worker = Path(__file__).resolve().with_name('mlx_generate.py')
    env = os.environ.copy()
    env.setdefault('HF_HOME', str(Path(__file__).resolve().parent / '.hf-cache'))
    try:
        completed = subprocess.run(
            [sys.executable, str(worker)],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            cwd=worker.parent,
            env=env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(
            f'MLX generation exceeded its {timeout}-second timeout for {model!r}'
        ) from e
    if completed.returncode:
        detail = completed.stderr.strip()[-4000:] or completed.stdout.strip()[-4000:]
        raise RuntimeError(f'MLX generation failed with exit {completed.returncode}: {detail}')
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f'MLX worker returned invalid JSON: {completed.stdout[-2000:]!r}'
        ) from e
    text = result.get('text', '')
    if not text:
        raise RuntimeError(f'MLX returned no generated text: {result!r}')
    return text, int(result.get('prompt_tokens') or 0), int(result.get('generation_tokens') or 0)


def _release_ollama(model):
    """Unload local model weights while leaving the Ollama server running.

    This is important on machines with unified memory: the recommender's
    multi-seed evaluation runs in the same memory pool as the local LLM.
    Ollama treats an empty generate request with keep_alive=0 as an unload.
    Releasing is best-effort; a later proposal call will load the model again.
    """
    base_url = os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434').rstrip('/')
    payload = {'model': model, 'prompt': '', 'stream': False, 'keep_alive': 0}
    request = Request(
        f'{base_url}/api/generate', data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST',
    )
    try:
        with urlopen(request, timeout=30):
            pass
    except (HTTPError, URLError, TimeoutError) as e:
        print(f'[ollama-release] could not unload {model!r}: {e}', file=sys.stderr)
        return False

    # The unload endpoint returns before memory is necessarily free. Poll the
    # local process list so a following NumPy/FM job cannot start while Ollama
    # still owns most of a laptop's unified memory.
    deadline = time.monotonic() + 60
    target_name = model.split(':', 1)[0]
    while time.monotonic() < deadline:
        status_request = Request(f'{base_url}/api/ps', method='GET')
        try:
            with urlopen(status_request, timeout=5) as response:
                active = json.loads(response.read().decode('utf-8')).get('models', [])
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f'[ollama-release] could not check unload status: {e}', file=sys.stderr)
            return False
        active_names = [entry.get('name', '').split(':', 1)[0] for entry in active]
        if target_name not in active_names:
            return True
        time.sleep(0.5)
    print(f'[ollama-release] {model!r} is still loaded after 60 seconds', file=sys.stderr)
    return False


PROVIDERS = {
    'anthropic': _propose_anthropic,
    'openai': _propose_openai,
    'ollama': _propose_ollama,
    'mlx': _propose_mlx,
}


def release(model, provider):
    """Free provider resources before a memory-intensive evaluation.

    Remote providers are stateless from this process's perspective. Ollama is
    intentionally the exception: unload its local weights but leave its daemon
    available for the next proposal. MLX generation already runs in a worker
    that exits after every call, so there is nothing left to release here.
    """
    if provider == 'ollama':
        return _release_ollama(model)
    return True


def _is_retryable(e):
    if type(e).__name__ in _RETRYABLE_EXCEPTION_NAMES:
        return True
    status = getattr(e, 'status_code', None)
    return isinstance(status, int) and status >= 500


def propose(prompt, model, provider):
    """The provider-agnostic entrypoint: dispatch to the selected provider's
    implementation. A network error or rate limit is caught, logged to
    stderr, and retried exactly once; any other failure — or a second
    failure after the retry — propagates to the caller, which is expected
    to log it as a failed iteration rather than crash the loop.
    Returns (text, tokens_in, tokens_out)."""
    fn = PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(f'unknown provider {provider!r}, must be one of {sorted(PROVIDERS)}')
    try:
        return fn(prompt, model)
    except Exception as e:
        # A local generation that exhausts its configured timeout has already
        # consumed the full time budget. Retrying it would turn the advertised
        # five-minute cap into ten minutes.
        if provider in ('ollama', 'mlx') and isinstance(e, TimeoutError):
            raise
        if not _is_retryable(e):
            raise
        print(f'[llm-retry] {provider}/{model}: {type(e).__name__}: {e} -- retrying once',
              file=sys.stderr)
        time.sleep(2)
        return fn(prompt, model)   # a second failure propagates to the caller
