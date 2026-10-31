"""Thin wrapper around mlx-lm for batched solving with thinking-token accounting."""
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(ROOT / 'models' / 'qwen3-8b-4bit')
END_THINK = '</think>'


class LLM:
    def __init__(self, adapter=None, model=MODEL):
        self.model, self.tokenizer = load(model, adapter_path=adapter)

    def prompt(self, system, user, thinking=True):
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=thinking, tokenize=True,
            return_dict=False)

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def solve(self, prompts, max_tokens=1536, batch_size=8, seed=0, temperature=0.6):
        """Return one dict per prompt, in order, plus the wall time in seconds."""
        mx.random.seed(seed)
        sampler = make_sampler(temp=temperature, top_p=0.95, top_k=20)
        start = time.time()
        response = batch_generate(
            self.model, self.tokenizer, prompts, max_tokens=max_tokens, sampler=sampler,
            completion_batch_size=batch_size, prefill_batch_size=batch_size,
            return_token_ids=True)
        elapsed = time.time() - start
        results = []
        for prompt, text, ids in zip(prompts, response.texts, response.token_ids):
            head, found, _ = text.partition(END_THINK)
            results.append({
                'text': text,
                'prompt_tokens': len(prompt),
                'output_tokens': len(ids),
                'thinking_tokens': self.count(head) if found else 0,
                'finished': len(ids) < max_tokens,
            })
        return results, elapsed, response.stats
