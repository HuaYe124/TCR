"""
TCR End-to-End Evaluation Script.

Load checkpoint and evaluate F1 score on test data.

Usage:
    python eval.py --data test_data.jsonl --checkpoint outputs/tcr_final.pt --hf_token hf_xxx
    python eval.py --data test_data.jsonl --checkpoint outputs/tcr_best.pt --hf_token hf_xxx
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

SFR_NAME = "Salesforce/SFR-Embedding-Mistral"
CKPT_DIR = PROJECT_ROOT.parent / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from model import TCRModel
from utils import generate_text, compute_f1


def load_models(hf_token: str, checkpoint_path: Path):
    """Load LLM, encoder, and TCR model."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

    os.environ["HF_TOKEN"] = hf_token
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

    LLM_NAME = "meta-llama/Llama-3.1-8B-Instruct"

    print(f"Device: {DEVICE}")

    print("Loading Llama...")
    llama_tokenizer = AutoTokenizer.from_pretrained(
        LLM_NAME, token=hf_token, trust_remote_code=True
    )
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    llama = AutoModelForCausalLM.from_pretrained(
        LLM_NAME, token=hf_token,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    llama.eval()

    embed_dim = llama.config.hidden_size

    model = TCRModel(llama, embed_dim)

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.signal_projector.load_state_dict(ckpt["signal_projector"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using random init")

    model.to(DEVICE)
    model.eval()

    print("Loading SFR encoder...")
    sfr_tokenizer = AutoTokenizer.from_pretrained(SFR_NAME)
    sfr_encoder = AutoModel.from_pretrained(
        SFR_NAME,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    sfr_encoder.eval()

    print("Loading Stage1 dual encoder...")
    stage1_path = CKPT_DIR / "dual_encoder_v2__Salesforce-SFR-Embedding-Mistral.pt"
    encoder_sem, encoder_fact = None, None
    if stage1_path.exists():
        state = torch.load(stage1_path, map_location="cpu", weights_only=False)
        d_sfr = state.get("d_sfr", 4096)
        sub_dim = state.get("sub_dim", 4096)
        encoder_sem = torch.nn.Linear(d_sfr, sub_dim)
        encoder_fact = torch.nn.Linear(d_sfr, sub_dim)
        encoder_sem.load_state_dict(state["encoder_sem_state_dict"])
        encoder_fact.load_state_dict(state["encoder_fact_state_dict"])
        encoder_sem.to(DEVICE).eval()
        encoder_fact.to(DEVICE).eval()
        print(f"Loaded Stage1 encoder, sub_dim={sub_dim}")

    print("Loading Stage2 MLP...")
    stage2_path = CKPT_DIR / "answerability_mlp__meta-llama-Llama-3.1-8B-Instruct.pt"
    mlp_ans = None
    if stage2_path.exists():
        mlp_state = torch.load(stage2_path, map_location="cpu", weights_only=False)
        hidden_dim = 4096 * 3
        mlp_ans = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 2),
        )
        remapped = {}
        key_map = {
            "fc1.weight": "0.weight", "fc1.bias": "0.bias",
            "fc2.weight": "3.weight", "fc2.bias": "3.bias",
            "fc3.weight": "6.weight", "fc3.bias": "6.bias",
        }
        for old_key, new_key in key_map.items():
            if old_key in mlp_state["mlp_state_dict"]:
                remapped[new_key] = mlp_state["mlp_state_dict"].pop(old_key)
        mlp_state["mlp_state_dict"].update(remapped)
        mlp_ans.load_state_dict(mlp_state["mlp_state_dict"])
        mlp_ans.to(DEVICE).eval()
        print("Loaded Stage2 MLP")

    return {
        "llm": llama, "tokenizer": llama_tokenizer, "embed_dim": embed_dim,
        "sfr_encoder": sfr_encoder, "sfr_tokenizer": sfr_tokenizer,
        "encoder_sem": encoder_sem, "encoder_fact": encoder_fact,
        "mlp_ans": mlp_ans, "model": model
    }


def cosine_similarity(z1, z2):
    z1_norm = F.normalize(z1, p=2, dim=-1)
    z2_norm = F.normalize(z2, p=2, dim=-1)
    return (z1_norm * z2_norm).sum(dim=-1)


def encode_texts(texts, encoder, tokenizer, device):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=2048, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0))
        embeddings = hidden[batch_idx, seq_lengths].float()

    return embeddings


def compute_signals(question, answer, context, models, device):
    """Compute σ_sem, σ_fact, σ_ans for a single record."""
    llm = models["llm"]
    tokenizer = models["tokenizer"]
    sfr_enc = models["sfr_encoder"]
    sfr_tok = models["sfr_tokenizer"]
    enc_sem = models["encoder_sem"]
    enc_fact = models["encoder_fact"]
    mlp = models["mlp_ans"]

    prompt = f"""Question: {question}

Answer the question using your knowledge. Give a specific name, number, or date. Keep answer SHORT (1-3 words maximum).

Answer:"""

    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = llm.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    generated = generated.strip().replace("<|eot_id|>", "").replace("Answer:", "").strip()
    if not generated:
        generated = "Unknown"

    answer_emb = encode_texts([generated], sfr_enc, sfr_tok, device)[0]
    context_emb = encode_texts([context], sfr_enc, sfr_tok, device)[0].to(device)

    if enc_sem is not None and enc_fact is not None:
        sigma_sem = cosine_similarity(
            enc_sem(answer_emb).unsqueeze(0),
            enc_sem(context_emb).unsqueeze(0)
        ).item()

        sigma_fact = cosine_similarity(
            enc_fact(answer_emb).unsqueeze(0),
            enc_fact(context_emb).unsqueeze(0)
        ).item()
    else:
        sigma_sem, sigma_fact = 0.5, 0.5

    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_len = inputs["input_ids"].shape[1]
    last_pos = input_len - 1

    with torch.no_grad():
        outputs = llm(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states
        num_layers_total = len(hidden_states) - 1

        layers_to_extract = [
            int(num_layers_total * (k / 5)) for k in [2, 3, 4]
        ]
        layers_to_extract = [max(1, min(l, num_layers_total)) for l in layers_to_extract]
        layers_to_extract = sorted(set(layers_to_extract))

        pooled_list = []
        for layer_idx in layers_to_extract:
            target_layer = hidden_states[layer_idx]
            last_hidden = target_layer[0, last_pos, :]
            pooled_list.append(last_hidden)

        concat_hidden = torch.cat(pooled_list, dim=0)

    if mlp is not None:
        with torch.no_grad():
            if concat_hidden.dim() == 1:
                concat_hidden = concat_hidden.unsqueeze(0)
            concat_hidden = concat_hidden.to(device)
            logits = mlp(concat_hidden)
            probs = F.softmax(logits, dim=-1)
            sigma_ans = probs[0, 1].item()
    else:
        sigma_ans = 0.5

    return sigma_sem, sigma_fact, sigma_ans


def evaluate(data_file: Path, models, num_samples: int = 200):
    """Evaluate F1 on test data."""
    model = models["model"]
    tokenizer = models["tokenizer"]

    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    print(f"Loaded {len(samples)} samples from {data_file}")

    if num_samples:
        samples = samples[:num_samples]

    f1_scores = []
    details = []

    print(f"\nEvaluating {len(samples)} samples...")

    for i, sample in enumerate(samples):
        question = sample["question"]
        answer = sample["answer"]
        context = sample.get("context") or sample.get("golden") or ""

        if not context:
            continue

        sigma_sem, sigma_fact, sigma_ans = compute_signals(
            question, answer, context, models, DEVICE
        )

        user_header = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        user_content = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        eot = "<|eot_id|>"
        prompt_raw = user_header + user_content + eot + assistant_header

        prompt_tokens = tokenizer(
            prompt_raw,
            padding=False,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
            return_tensors="pt"
        )
        prompt_ids = prompt_tokens["input_ids"].to(DEVICE)
        prompt_mask = prompt_tokens["attention_mask"].to(DEVICE)

        sigma_sem_t = torch.tensor([sigma_sem], dtype=torch.float32, device=DEVICE)
        sigma_fact_t = torch.tensor([sigma_fact], dtype=torch.float32, device=DEVICE)
        sigma_ans_t = torch.tensor([sigma_ans], dtype=torch.float32, device=DEVICE)

        gen_text = generate_text(
            model, tokenizer,
            prompt_ids, prompt_mask,
            sigma_sem_t, sigma_fact_t, sigma_ans_t,
            DEVICE
        )

        pred_tokens = tokenizer.encode(gen_text, add_special_tokens=False)
        true_tokens = tokenizer.encode(answer, add_special_tokens=False)
        f1 = compute_f1(pred_tokens, true_tokens)
        f1_scores.append(f1)
        details.append({
            "question": question[:80],
            "answer": answer,
            "generated": gen_text,
            "f1": f1,
            "sigma_sem": sigma_sem,
            "sigma_fact": sigma_fact,
            "sigma_ans": sigma_ans,
        })

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(samples)}] F1: {f1:.4f} | σ_sem: {sigma_sem:.3f} | σ_fact: {sigma_fact:.3f} | σ_ans: {sigma_ans:.3f}")

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    print("\n" + "=" * 60)
    print(f"Evaluation Results ({len(f1_scores)} samples)")
    print("=" * 60)
    print(f"Average F1: {avg_f1:.4f}")

    if details:
        high_sem = [d for d in details if d["sigma_sem"] > 0.7]
        low_sem = [d for d in details if d["sigma_sem"] <= 0.7]
        print(f"\nBy σ_sem:")
        if high_sem:
            print(f"  High (>0.7): {sum(d['f1'] for d in high_sem)/len(high_sem):.4f} ({len(high_sem)} samples)")
        if low_sem:
            print(f"  Low (<=0.7): {sum(d['f1'] for d in low_sem)/len(low_sem):.4f} ({len(low_sem)} samples)")

    print("\nSample outputs:")
    for j, d in enumerate(details[:3]):
        print(f"\n[{j+1}] F1: {d['f1']:.4f}")
        print(f"  Q: {d['question']}...")
        print(f"  GT: {d['answer']}")
        print(f"  Gen: {d['generated'][:100]}...")

    return avg_f1, f1_scores, details


def main():
    import torch.nn as nn

    parser = argparse.ArgumentParser(description="Evaluate TCR model")
    parser.add_argument("--data", type=str, required=True, help="Test data file (jsonl)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace token")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    args = parser.parse_args()

    data_file = Path(args.data)
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else OUTPUT_DIR / "tcr_final.pt"

    models = load_models(args.hf_token, checkpoint_path)

    avg_f1, f1_scores, details = evaluate(data_file, models, num_samples=args.num_samples)

    if args.output:
        output_path = Path(args.output)
        results = {
            "avg_f1": avg_f1,
            "num_samples": len(f1_scores),
            "f1_scores": f1_scores,
            "details": details,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_path}")

    print("\n[OK] Evaluation complete!")


if __name__ == "__main__":
    main()
