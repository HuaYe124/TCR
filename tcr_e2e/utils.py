"""Shared evaluation utilities for train.py and eval.py."""

import torch
import torch.nn.functional as F
from typing import List


def compute_f1(pred_tokens: List[int], true_tokens: List[int]) -> float:
    """Compute token-level F1 score."""
    pred_set = set(pred_tokens)
    true_set = set(true_tokens)

    if len(pred_set) == 0 or len(true_set) == 0:
        return 0.0

    common = pred_set & true_set
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(true_set)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def generate_text(model, tokenizer, input_ids, attention_mask, sigma_sem, sigma_fact, sigma_ans, device, max_new_tokens=100):
    """Generate text using trained TCR model."""
    model.eval()

    with torch.no_grad():
        aug_emb, aug_mask = model.build_augmented_embeddings(
            input_ids, sigma_sem, sigma_fact, sigma_ans, attention_mask, device, tokenizer
        )
        aug_emb = aug_emb.to(model.llm.dtype)

        eos_token_id = tokenizer.eos_token_id
        generated_ids = model.llm.generate(
            inputs_embeds=aug_emb,
            attention_mask=aug_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or eos_token_id,
            eos_token_id=eos_token_id,
        )

        new_tokens = generated_ids[0].tolist()

        if eos_token_id in new_tokens:
            eos_idx = new_tokens.index(eos_token_id)
            new_tokens = new_tokens[:eos_idx]

        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = text.strip()

    return text


def evaluate_f1(model, dataloader, tokenizer, device, num_samples=10):
    """Evaluate F1 score on a dataloader (train.py style)."""
    model.eval()
    f1_scores = []
    details = []
    samples_evaluated = 0

    for batch in dataloader:
        if samples_evaluated >= num_samples:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sigma_sem = batch["sigma_sem"].to(device)
        sigma_fact = batch["sigma_fact"].to(device)
        sigma_ans = batch["sigma_ans"].to(device)
        prompt_lens = batch["prompt_lens"]
        answers = batch["answers"]

        B = input_ids.size(0)

        for i in range(B):
            if samples_evaluated >= num_samples:
                break

            prompt_len = prompt_lens[i]
            prompt_ids = input_ids[i:i+1, :prompt_len]
            prompt_mask = attention_mask[i:i+1, :prompt_len]

            gen_text = generate_text(
                model, tokenizer,
                prompt_ids,
                prompt_mask,
                sigma_sem[i:i+1],
                sigma_fact[i:i+1],
                sigma_ans[i:i+1],
                device
            )

            true_answer = answers[i]
            pred_tokens = tokenizer.encode(gen_text, add_special_tokens=False)
            true_tokens = tokenizer.encode(true_answer, add_special_tokens=False)

            f1 = compute_f1(pred_tokens, true_tokens)
            f1_scores.append(f1)
            details.append({
                "pred": gen_text[:200],
                "true": true_answer[:200],
                "f1": f1
            })

            if samples_evaluated < 3:
                print(f"\n{'='*80}")
                print(f"[Sample {samples_evaluated + 1}] F1: {f1:.4f}")
                print(f"{'='*80}")
                print(f"Prompt: {repr(tokenizer.decode(prompt_ids[0], skip_special_tokens=True)[:200])}")
                print(f"Pred: {repr(gen_text[:200])}")
                print(f"True: {repr(true_answer[:200])}")

            samples_evaluated += 1

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return avg_f1, f1_scores, details
