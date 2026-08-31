"""One-shot MLX-VLM generation worker used by llm.py.

The request and response are JSON over stdin/stdout. Keeping MLX imports in
this process guarantees model weights are released before agent.py starts its
memory-intensive recommender evaluation.
"""
import json
import sys


def main():
    request = json.load(sys.stdin)

    import mlx.core as mx
    from mlx_vlm import apply_chat_template, generate, load

    mx.random.seed(int(request.get('seed', 0)))
    model, processor = load(
        request['model'],
        adapter_path=request.get('adapter_path'),
        trust_remote_code=True,
    )
    prompt = apply_chat_template(
        processor,
        model.config,
        request['prompt'],
        num_images=0,
        num_audios=0,
        enable_thinking=False,
    )
    result = generate(
        model,
        processor,
        prompt,
        max_tokens=int(request['max_tokens']),
        temperature=float(request.get('temperature', 0.1)),
        top_p=float(request.get('top_p', 0.95)),
        verbose=False,
        enable_thinking=False,
    )
    json.dump({
        'text': result.text,
        'prompt_tokens': result.prompt_tokens,
        'generation_tokens': result.generation_tokens,
        'finish_reason': result.finish_reason,
        'peak_memory_gb': result.peak_memory,
    }, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
