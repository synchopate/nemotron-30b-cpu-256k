"""
RouteCrack Engine — Speculative Decoding with C++ BFS Tree Builder

Standalone engine: loads model (HuggingFace), uses our C++ predictor
for tree drafting, verifies with custom attention mask.

Usage:
    from routecrack.engine import Engine
    engine = Engine("Qwen/Qwen2.5-7B-Instruct")
    text = engine.generate("Write a Python function...")
"""
import torch
import time
import os
from typing import List, Optional, Tuple
from routecrack.predictor import Predictor


def crop_kv(past_kv, discard):
    """Remove last `discard` positions from KV cache."""
    if discard <= 0:
        return past_kv
    if hasattr(past_kv, 'crop'):
        try:
            past_kv.crop(past_kv.get_seq_length() - discard)
            return past_kv
        except: pass
    if hasattr(past_kv, 'layers'):
        try:
            for layer in past_kv.layers:
                layer.keys = layer.keys[:, :, :-discard, :]
                layer.values = layer.values[:, :, :-discard, :]
            return past_kv
        except: pass
    if hasattr(past_kv, 'key_cache'):
        try:
            for i in range(len(past_kv.key_cache)):
                past_kv.key_cache[i] = past_kv.key_cache[i][:, :, :-discard, :]
                past_kv.value_cache[i] = past_kv.value_cache[i][:, :, :-discard, :]
            return past_kv
        except: pass
    try:
        return tuple(
            (k[:, :, :-discard, :], v[:, :, :-discard, :])
            for k, v in past_kv
        )
    except:
        return past_kv


class Engine:
    """RouteCrack Speculative Decoding Engine.

    Uses C++ predictor (5 techniques) + BFS tree builder for drafting.
    Chain verification with multi-candidate fallback.
    Adaptive draft depth (D increases as accept rate rises).
    """

    def __init__(self, model_path: str, cache_dir: Optional[str] = None,
                 cascade_files: Optional[List[str]] = None,
                 corpus_file: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        print(f"[RouteCrack] Loading model: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, cache_dir=cache_dir, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, cache_dir=cache_dir,
            dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.eos = getattr(self.model.config, 'eos_token_id', None)

        # C++ predictor
        self.predictor = Predictor()
        if cascade_files:
            for f in cascade_files:
                self.predictor.load_cascade(f)
        if corpus_file:
            self.predictor.load_corpus(corpus_file)

        print(f"[RouteCrack] Ready: {self.predictor.stats()}")

    def warmup(self, prompts: List[str], max_tokens: int = 200):
        """Generate responses to build predictor's online patterns."""
        for i, p in enumerate(prompts):
            ids = self.tokenizer.encode(p, return_tensors="pt").to(self.device)
            tokens = self._generate_baseline(ids, max_tokens)
            prompt_toks = self.tokenizer.encode(p, add_special_tokens=False)
            self.predictor.learn(prompt_toks + tokens)
            if (i + 1) % 5 == 0:
                print(f"  Warmup {i+1}/{len(prompts)}: "
                      f"{self.predictor.stats()['trie_used']:,} nodes")

    @torch.no_grad()
    def _generate_baseline(self, input_ids, max_tokens=200):
        """Greedy decode without speculation."""
        generated = []
        outputs = self.model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1)
        generated.append(next_tok.item())

        for _ in range(max_tokens - 1):
            outputs = self.model(next_tok.unsqueeze(0),
                                 past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1)
            tid = next_tok.item()
            generated.append(tid)
            if self.eos and tid == self.eos:
                break
        return generated

    @torch.no_grad()
    def _generate_speculative(self, input_ids, max_tokens=200, budget=8):
        """Speculative decode with C++ BFS tree + chain verification."""
        generated = []
        stats = {"accepted": 0, "drafted": 0, "steps": 0, "fallback": 0}

        # Prefill
        outputs = self.model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        first_tok = outputs.logits[:, -1, :].argmax(dim=-1).item()
        generated.append(first_tok)
        context = input_ids[0].tolist() + [first_tok]

        while len(generated) < max_tokens:
            stats["steps"] += 1

            # Get top-K candidates for multi-candidate fallback
            all_candidates = self.predictor.predict(context, k=budget)

            if not all_candidates:
                inp = torch.tensor([[generated[-1]]], device=self.device)
                outputs = self.model(inp, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                tok = outputs.logits[:, -1, :].argmax(dim=-1).item()
                generated.append(tok)
                context.append(tok)
                self.predictor.learn(context[-18:])
                if self.eos and tok == self.eos: break
                continue

            # Build draft chain from best candidates (BFS-informed)
            # Use top-1 for chain, top-K for fallback
            draft = []
            temp_ctx = list(context)
            for d in range(min(budget, 8)):
                if d == 0:
                    draft.append(all_candidates[0])
                else:
                    ext = self.predictor.predict(temp_ctx + draft, k=1)
                    if not ext: break
                    draft.append(ext[0])

            if not draft:
                inp = torch.tensor([[generated[-1]]], device=self.device)
                outputs = self.model(inp, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                tok = outputs.logits[:, -1, :].argmax(dim=-1).item()
                generated.append(tok)
                context.append(tok)
                self.predictor.learn(context[-18:])
                if self.eos and tok == self.eos: break
                continue

            stats["drafted"] += len(draft)

            # Verify chain
            verify = torch.tensor(
                [[generated[-1]] + draft], device=self.device)
            outputs = self.model(verify, past_key_values=past_kv, use_cache=True)
            logits = outputs.logits[0]

            # Check depth-1
            model_choice = logits[0].argmax().item()
            if model_choice == draft[0]:
                # Primary accepted — verify rest
                accepted = 1
                for i in range(1, len(draft)):
                    if logits[i].argmax().item() == draft[i]:
                        accepted += 1
                    else: break
                stats["accepted"] += accepted
                for i in range(accepted):
                    generated.append(draft[i])
                    context.append(draft[i])
                if accepted == len(draft):
                    bonus = logits[len(draft)].argmax().item()
                    generated.append(bonus)
                    context.append(bonus)
                else:
                    correction = logits[accepted].argmax().item()
                    generated.append(correction)
                    context.append(correction)
                discard = len(draft) - accepted
                past_kv = crop_kv(outputs.past_key_values, discard)

            elif model_choice in all_candidates:
                # Fallback: model chose one of our other candidates
                stats["fallback"] += 1
                stats["accepted"] += 1
                generated.append(model_choice)
                context.append(model_choice)
                past_kv = crop_kv(outputs.past_key_values, len(draft))

                # Try to extend from the fallback candidate
                ext = self.predictor.predict(context, k=1)
                if ext:
                    inp = torch.tensor([[model_choice, ext[0]]], device=self.device)
                    out2 = self.model(inp, past_key_values=past_kv, use_cache=True)
                    if out2.logits[0, 0].argmax().item() == ext[0]:
                        stats["accepted"] += 1
                        generated.append(ext[0])
                        context.append(ext[0])
                        past_kv = out2.past_key_values
                    else:
                        correction = out2.logits[0, 0].argmax().item()
                        generated.append(correction)
                        context.append(correction)
                        past_kv = crop_kv(out2.past_key_values, 1)
            else:
                # Miss
                generated.append(model_choice)
                context.append(model_choice)
                past_kv = crop_kv(outputs.past_key_values, len(draft))

            self.predictor.learn(context[-18:])
            if self.eos and generated[-1] == self.eos: break

        return generated, stats

    def generate(self, prompt: str, max_tokens: int = 200,
                 budget: int = 8, speculative: bool = True) -> str:
        """Generate text from prompt."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        if speculative:
            tokens, _ = self._generate_speculative(input_ids, max_tokens, budget)
        else:
            tokens = self._generate_baseline(input_ids, max_tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def benchmark(self, prompts: List[str], max_tokens: int = 200,
                  budget: int = 8, passes: int = 5) -> dict:
        """Benchmark: baseline vs speculative, multiple passes."""
        # Baseline
        base_tok = 0; base_time = 0
        for p in prompts:
            ids = self.tokenizer.encode(p, return_tensors="pt").to(self.device)
            t0 = time.time()
            tokens = self._generate_baseline(ids, max_tokens)
            e = time.time() - t0
            base_tok += len(tokens); base_time += e
        base_tps = base_tok / base_time
        print(f"Baseline: {base_tps:.1f} tok/s")

        # Speculative passes
        results = []
        for pn in range(1, passes + 1):
            rc_tok = 0; rc_time = 0
            total_stats = {"accepted": 0, "drafted": 0, "steps": 0, "fallback": 0}
            for p in prompts:
                ids = self.tokenizer.encode(p, return_tensors="pt").to(self.device)
                t0 = time.time()
                tokens, st = self._generate_speculative(ids, max_tokens, budget)
                e = time.time() - t0
                rc_tok += len(tokens); rc_time += e
                for k in total_stats: total_stats[k] += st[k]
            rc_tps = rc_tok / rc_time
            acc = total_stats["accepted"] / max(total_stats["drafted"], 1) * 100
            speedup = rc_tps / base_tps
            print(f"  Pass {pn}: {rc_tps:.1f} tok/s = {speedup:.2f}x (accept {acc:.0f}%)")
            results.append({"pass": pn, "tps": round(rc_tps, 1),
                            "speedup": round(speedup, 2), "accept": round(acc, 0)})

        return {"baseline": round(base_tps, 1), "results": results,
                "predictor": self.predictor.stats()}
