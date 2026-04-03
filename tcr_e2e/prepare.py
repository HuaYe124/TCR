"""
TCR End-to-End Training Data Preparation.

Process:
1. Load context data (golden/conflict/irrelevant)
2. SFR encode question
3. Llama generate answer
4. Compute σ_sem, σ_fact using query + Llama output
5. Compute σ_ans using query Llama hidden states
6. Generate 3 samples per question (one per context type)
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN
if os.environ.get("HTTPS_PROXY"):
    os.environ.setdefault("HTTPS_PROXY", os.environ["HTTPS_PROXY"])
if os.environ.get("HTTP_PROXY"):
    os.environ.setdefault("HTTP_PROXY", os.environ["HTTP_PROXY"])
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
CKPT_DIR = SCRIPT_DIR.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

OUTPUT_FILE = DATA_DIR / "tcr_training_data.jsonl"
TRICK_MODE = os.environ.get("TRICK_MODE", "baseline")  # baseline, v1, v2, v3, v4

# Output file varies by trick mode
if TRICK_MODE != "baseline":
    OUTPUT_FILE = DATA_DIR / f"tcr_training_data_{TRICK_MODE}.jsonl"
    logger.info(f"Trick mode: {TRICK_MODE}, output: {OUTPUT_FILE}")

DEFAULT_N_SAMPLES = 1300

# Model configs
LLM_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SFR_NAME = "Salesforce/SFR-Embedding-Mistral"

# TCR config
NUM_SOFT_TOKENS = 5  # Per signal
LAYER_RATIO = 2 / 3  # Extract middle layer at 2/3 of total layers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "prepare_training_data.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


    # ===== Data Structures =====

@dataclass
class TCRTrainingSample:
    """TCR training sample with all necessary data for end-to-end training."""
    # Question and context
    question: str
    contexts: List[str]  # List of context strings (golden/conflict/irrelevant)
    
    # Golden answer (short answer from dataset)
    golden_answer: str
    
    # Labels
    label: int = 1  # 1 = golden context, 0 = conflict/irrelevant context
    
    # Context type
    context_type: str = "golden"  # "golden", "conflict", "irrelevant"
    
    # Tokenized prompt (question + context + answer)
    prompt_ids: List[int] = field(default_factory=list)
    prompt_mask: List[int] = field(default_factory=list)
    prompt_len: int = 0
    
    # Labels aligned with model output
    labels: List[int] = field(default_factory=list)
    
    # ===== Precomputed signals =====
    sigma_sem: float = 0.0  # Semantic relevance
    sigma_fact: float = 0.0  # Factual consistency
    sigma_ans: float = 0.0   # Answerability
    
    # Llama generated answer
    generated_answer: str = ""
    
    # Metadata
    split: str = "train"
    dataset_name: str = "SQuAD"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["answer"] = d.get("golden_answer", "")
        return d


    # ===== Model Loading =====

def load_llm():
    """Load Llama model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    logger.info(f"Loading LLM: {LLM_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_NAME,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if DEVICE == "cuda" else None
    
    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        quantization_config=quantization_config,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    
    llm_config = model.config
    hidden_size = llm_config.hidden_size
    num_layers = getattr(llm_config, "num_hidden_layers", 32)
    extract_layer = int(num_layers * LAYER_RATIO)
    
    logger.info(f"  hidden_size={hidden_size}, num_layers={num_layers}, extract_layer={extract_layer}")
    
    return model, tokenizer, hidden_size, extract_layer


def load_sfr_encoder():
    """Load SFR embedding model."""
    from transformers import AutoModel, AutoTokenizer
    
    logger.info(f"Loading SFR encoder: {SFR_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(SFR_NAME)
    encoder = AutoModel.from_pretrained(
        SFR_NAME, 
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    encoder.eval()
    
    for param in encoder.parameters():
        param.requires_grad = False
    
    hidden_size = encoder.config.hidden_size
    logger.info(f"  SFR hidden_size={hidden_size}")
    
    return encoder, tokenizer, hidden_size


def load_stage1_dual_encoder():
    """Load pretrained Stage 1 dual encoder projectors."""
    import sys
    logger.info("Loading Stage 1 dual encoder checkpoint...")
    
    ckpt_path = CKPT_DIR / "dual_encoder_v2__Salesforce-SFR-Embedding-Mistral.pt"
    
    if not ckpt_path.exists():
        logger.warning(f"Stage 1 checkpoint not found: {ckpt_path}, using random init")
        return None, None
    
    state = torch.load(ckpt_path, map_location="cpu")
    d_sfr = state.get("d_sfr", 4096)
    sub_dim = state.get("sub_dim", 4096)
    
    encoder_sem = torch.nn.Linear(d_sfr, sub_dim)
    encoder_fact = torch.nn.Linear(d_sfr, sub_dim)
    encoder_sem.load_state_dict(state["encoder_sem_state_dict"])
    encoder_fact.load_state_dict(state["encoder_fact_state_dict"])
    
    encoder_sem.to(DEVICE)
    encoder_fact.to(DEVICE)
    encoder_sem.eval()
    encoder_fact.eval()
    
    logger.info(f"  Loaded dual encoder, sub_dim={sub_dim}")
    
    return encoder_sem, encoder_fact


def load_stage2_answerability_mlp():
    """Load pretrained Stage 2 answerability MLP."""
    
    # Multi-layer concat: 4096 * 3 = 12288
    hidden_dim = 4096 * 3
    
    mlp = torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 2),
    )
    
    ckpt_path = CKPT_DIR / "answerability_mlp__meta-llama-Llama-3.1-8B-Instruct.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Remap old fc1/fc2/fc3 keys to Sequential layer indices 0/3/6
        mlp.load_state_dict(state["mlp_state_dict"])
        logger.info(f"  Loaded Stage 2 MLP from {ckpt_path.name}, val_acc={state.get('val_accuracy', 'N/A'):.4f}, val_f1={state.get('val_f1', 'N/A'):.4f}")
    else:
        logger.warning(f"  Stage 2 checkpoint NOT found: {ckpt_path}")
        logger.warning("  Using randomly initialized MLP — sigma_ans will be meaningless!")
    
    mlp.to(DEVICE)
    mlp.eval()
    
    logger.info(f"  Answerability MLP ready, hidden_dim={hidden_dim}")
    
    return mlp


    # ===== Encoding Functions =====

def encode_texts(texts: List[str], encoder, tokenizer, device) -> torch.Tensor:
    """Encode texts using encoder."""
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=2048, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0))
        embeddings = hidden[batch_idx, seq_lengths].float()
    
    return embeddings


def cosine_similarity(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity."""
    z1_norm = F.normalize(z1, p=2, dim=-1)
    z2_norm = F.normalize(z2, p=2, dim=-1)
    return (z1_norm * z2_norm).sum(dim=-1)


def compute_dual_encoder_signals(
    llama_answer_emb: torch.Tensor,
    context_emb: torch.Tensor,
    encoder_sem: torch.nn.Module,
    encoder_fact: torch.nn.Module,
    llama_answer_text: str = "",
    context_text: str = "",
    sample_idx: int = 0,
) -> Tuple[float, float]:
    """Compute σ_sem and σ_fact."""
    if encoder_sem is None or encoder_fact is None:
        return 0.5, 0.5

    with torch.no_grad():
        proj_answer_sem = encoder_sem(llama_answer_emb)
        proj_context_sem = encoder_sem(context_emb)
        
        # L2 normalization
        answer_sem_normed = F.normalize(proj_answer_sem.unsqueeze(0), p=2, dim=-1)
        context_sem_normed = F.normalize(proj_context_sem.unsqueeze(0), p=2, dim=-1)
        
        # Compute cosine similarity
        sem_sim = cosine_similarity(
            proj_answer_sem.unsqueeze(0), 
            proj_context_sem.unsqueeze(0)
        ).item()
        
        proj_answer_fact = encoder_fact(llama_answer_emb)
        proj_context_fact = encoder_fact(context_emb)
        
        fact_sim = cosine_similarity(
            proj_answer_fact.unsqueeze(0), 
            proj_context_fact.unsqueeze(0)
        ).item()
    
    return sem_sim, fact_sim


def extract_llama_hidden_state(
    model,
    tokenizer,
    text: str,
    device,
) -> torch.Tensor:
    """Extract middle layer hidden states from Llama (query only)."""
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)
    
    input_len = inputs["input_ids"].shape[1]
    last_pos = input_len - 1
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        
        hidden_states = outputs.hidden_states
        num_layers_total = len(hidden_states) - 1
        
        # Extract 3 layers: 2/3, 3/4, 4/5 position
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
        return concat_hidden.float().cpu()


def compute_answerability(
    hidden_state: torch.Tensor,
    mlp: torch.nn.Module,
    device,
) -> float:
    """Compute σ_ans (answerability probability)."""
    if mlp is None:
        return 0.5
    
    with torch.no_grad():
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        
        hidden_state = hidden_state.to(device)
        logits = mlp(hidden_state)
        probs = F.softmax(logits, dim=-1)
        sigma_ans = probs[0, 1].item()
    
    return sigma_ans


    # ===== LLM Generation =====

def llama_generate_answer(
    model,
    tokenizer,
    question: str,
    device,
    sample_idx: int = 0,
) -> str:
    """Llama generates answer using internal knowledge (question only)."""
    prompt = f"""Question: {question}

Answer the question using your knowledge. Give a specific name, number, or date. Keep answer SHORT (1-3 words maximum).

Answer:"""
    
    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    generated = generated.strip()
    generated = generated.replace("<|eot_id|>", "").strip()
    generated = generated.replace("Answer:", "").strip()
    
    return generated if generated else "Unknown"


    # ===== Data Loading =====

def load_context_data() -> List[Dict]:
    """Load context data."""
    input_file = DATA_DIR / "squad_qa_context_complete.json"
    
    if not input_file.exists():
        raise FileNotFoundError(f"Context file not found: {input_file}")
    
    logger.info(f"Loading context data from {input_file}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    
    data = [d for d in data if d.get("golden") and d.get("conflict") and d.get("irrelevant")]
    logger.info(f"Loaded {len(data)} complete records")
    return data


    # ===== Training Sample Construction =====

def construct_training_samples(
    record: Dict,
    llm,
    tokenizer_llm,
    encoder_sfr,
    tokenizer_sfr,
    encoder_sem,
    encoder_fact,
    mlp_ans,
    device,
    signals_cache: Dict,
) -> List[TCRTrainingSample]:
    """Build 3 training samples per question."""
    question = record["question"]
    answer = record["answer"]
    
    samples = []
    context_types = ["golden", "conflict", "irrelevant"]
    
    # Precompute Llama answer and σ_ans (once per query)
    query_key = question[:50]

    if query_key not in signals_cache:
        generated = llama_generate_answer(llm, tokenizer_llm, question, device)
        answer_emb = encode_texts([generated], encoder_sfr, tokenizer_sfr, device)[0]
        llama_hidden = extract_llama_hidden_state(llm, tokenizer_llm, question, device)
        sigma_ans = compute_answerability(llama_hidden, mlp_ans, device)
        signals_cache[query_key] = {
            "answer_emb": answer_emb,
            "sigma_ans": sigma_ans,
            "generated_answer": generated,
            "llama_hidden": llama_hidden,
        }

    cached = signals_cache[query_key]
    answer_emb = cached["answer_emb"].to(device)

    for ctx_type in context_types:
        context = record[ctx_type]
        context_emb = encode_texts([context], encoder_sfr, tokenizer_sfr, device)[0].to(device)
        
        user_header = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        user_content = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        eot = "<|eot_id|>"
        
        prompt_raw = user_header + user_content + eot + assistant_header
        
        prompt_tokens = tokenizer_llm(
            prompt_raw,
            padding=False,
            truncation=True,
            max_length=MAX_PROMPT_LEN,
            add_special_tokens=False,
            return_tensors="pt"
        )
        prompt_ids_only = prompt_tokens["input_ids"][0]
        prompt_len = prompt_ids_only.shape[0]

        # Tokenize answer
        answer_tokens = tokenizer_llm(
            answer,
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt"
        )
        answer_ids_only = answer_tokens["input_ids"][0]

        full_ids = torch.cat([prompt_ids_only, answer_ids_only], dim=0)
        
        if prompt_len < 10:
            continue

        full_ids_clean = full_ids
        labels = torch.full((full_ids_clean.shape[0],), -100, dtype=torch.long)
        labels[prompt_len:] = full_ids_clean[prompt_len:]

        # Label: golden=1, conflict/irrelevant=0
        label = 1 if ctx_type == "golden" else 0

        sample = TCRTrainingSample(
            question=question,
            contexts=[context],
            golden_answer=answer,
            label=label,
            context_type=ctx_type,
            prompt_ids=full_ids_clean.tolist(),
            prompt_mask=[1] * full_ids_clean.shape[0],
            prompt_len=prompt_len,
            labels=labels.tolist(),
            sigma_sem=sigma_sem,
            sigma_fact=sigma_fact,
            sigma_ans=cached["sigma_ans"],
            generated_answer=cached["generated_answer"],
        )
        samples.append(sample)

    return samples


def process_single_record(args_tuple) -> List[TCRTrainingSample]:
    """Process single record for thread pool."""
    record, llm, tokenizer_llm, encoder_sfr, tokenizer_sfr, \
        encoder_sem, encoder_fact, mlp_ans, device, signals_cache, lock, idx = args_tuple
    
    with lock:
        query_key = record["question"][:50]
        if query_key not in signals_cache:
            generated = llama_generate_answer(llm, tokenizer_llm, record["question"], device)
            answer_emb = encode_texts([generated], encoder_sfr, tokenizer_sfr, device)[0]
            llama_hidden = extract_llama_hidden_state(llm, tokenizer_llm, record["question"], device)
            sigma_ans = compute_answerability(llama_hidden, mlp_ans, device)
            signals_cache[query_key] = {
                "answer_emb": answer_emb,
                "sigma_ans": sigma_ans,
                "generated_answer": generated,
            }
    
    cached = signals_cache[query_key]
    answer_emb = cached["answer_emb"].to(device)

    samples = []
    context_types = ["golden", "conflict", "irrelevant"]

    for ctx_type in context_types:
        context = record[ctx_type]
        answer = record["answer"]
        
        context_emb = encode_texts([context], encoder_sfr, tokenizer_sfr, device)[0].to(device)
        sigma_sem, sigma_fact = compute_dual_encoder_signals(
            answer_emb, context_emb, encoder_sem, encoder_fact,
            llama_answer_text=cached["generated_answer"],
            context_text=context,
            sample_idx=idx,
        )
        
        MAX_PROMPT_LEN = 2048
        
        user_header = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        user_content = f"Context: {context}\n\nQuestion: {record['question']}\n\nAnswer:"
        eot = "<|eot_id|>"
        
        prompt_raw = user_header + user_content + eot + assistant_header
        
        prompt_tokens = tokenizer_llm(
            prompt_raw,
            padding=False,
            truncation=True,
            max_length=MAX_PROMPT_LEN,
            add_special_tokens=False,
            return_tensors="pt"
        )
        prompt_ids_only = prompt_tokens["input_ids"][0]
        prompt_len = prompt_ids_only.shape[0]

        answer_tokens = tokenizer_llm(
            answer,
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt"
        )
        answer_ids_only = answer_tokens["input_ids"][0]

        full_ids = torch.cat([prompt_ids_only, answer_ids_only], dim=0)

        if prompt_len < 10:
            continue

        full_ids_clean = full_ids
        labels = torch.full((full_ids_clean.shape[0],), -100, dtype=torch.long)
        labels[prompt_len:] = full_ids_clean[prompt_len:]

        label = 1 if ctx_type == "golden" else 0

        sample = TCRTrainingSample(
            question=record["question"],
            contexts=[context],
            golden_answer=answer,
            label=label,
            context_type=ctx_type,
            prompt_ids=full_ids_clean.tolist(),
            prompt_mask=[1] * full_ids_clean.shape[0],
            prompt_len=prompt_len,
            labels=labels.tolist(),
            sigma_sem=sigma_sem,
            sigma_fact=sigma_fact,
            sigma_ans=cached["sigma_ans"],
            generated_answer=cached["generated_answer"],
        )
        samples.append(sample)
    
    return idx, samples


def prepare_training_data(n: int = DEFAULT_N_SAMPLES, n_workers: int = 4) -> int:
    """Prepare TCR training data with parallel processing."""
    logger.info("=" * 60)
    logger.info("TCR End-to-End Training Data Preparation (Parallel)")
    logger.info(f"  Target samples: {n} questions × 3 contexts = {n * 3} total")
    logger.info(f"  Output: {OUTPUT_FILE}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Workers: {n_workers}")
    logger.info("=" * 60)
    
    # Load data
    records = load_context_data()
    records = records[:n]
    
    # Load models
    llm, tokenizer_llm, _, _ = load_llm()
    encoder_sfr, tokenizer_sfr, _ = load_sfr_encoder()
    encoder_sem, encoder_fact = load_stage1_dual_encoder()
    mlp_ans = load_stage2_answerability_mlp()
    
    # Signal cache and lock
    signals_cache = {}
    cache_lock = threading.Lock()
    
    # Train/val split
    split_idx = int(len(records) * 0.8)
    
    # Clear output file
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
    
    total_written = 0
    
    # Parallel processing
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_single_record,
                (record, llm, tokenizer_llm, encoder_sfr, tokenizer_sfr,
                 encoder_sem, encoder_fact, mlp_ans, DEVICE, signals_cache, cache_lock, i)
            ): i for i, record in enumerate(records)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                idx, samples = future.result()
                split = "train" if idx < split_idx else "val"
                
                for sample in samples:
                    sample.split = split
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                    total_written += 1

                    # Golden sample augmentation
                    if sample.context_type == "golden" and split == "train":
                        # V3/V4: replicate 2x, others replicate 1x
                        num_copies = 2 if TRICK_MODE in ["v3", "v4"] else 1
                        for _ in range(num_copies):
                            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                            total_written += 1
                    
            except Exception as e:
        logger.warning(f"Error processing record: {e}")
                continue
    
    logger.info(f"\nDone! Total samples written: {total_written}")
    
    # Statistics
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            all_samples = [json.loads(line) for line in f]
        
        n_train = sum(1 for s in all_samples if s.get("split") == "train")
        n_val = sum(1 for s in all_samples if s.get("split") == "val")
        
        logger.info(f"\nStatistics:")
        logger.info(f"  Total: {len(all_samples)}")
        logger.info(f"  Train: {n_train}, Val: {n_val}")
        
        # Signal distributions
        sig_sem = [s.get("sigma_sem", 0) for s in all_samples]
        sig_fact = [s.get("sigma_fact", 0) for s in all_samples]
        sig_ans = [s.get("sigma_ans", 0) for s in all_samples]
        
        logger.info(f"\nSignal distributions:")
        logger.info(f"  sigma_sem: mean={sum(sig_sem)/len(sig_sem):.3f}")
        logger.info(f"  sigma_fact: mean={sum(sig_fact)/len(sig_fact):.3f}")
        logger.info(f"  sigma_ans: mean={sum(sig_ans)/len(sig_ans):.3f}")
    
    return total_written


def verify_output():
    """Verify output file."""
    if not OUTPUT_FILE.exists():
        logger.error(f"Output file not found: {OUTPUT_FILE}")
        return
    
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    
    logger.info(f"=== Verification ===")
    logger.info(f"Total samples: {len(samples)}")
    
    if samples:
        sample = samples[0]
        logger.info(f"  Question: {sample.get('question', '')[:60]}...")
        logger.info(f"  Context type: {sample.get('context_type')}")
        logger.info(f"  sigma_sem: {sample.get('sigma_sem'):.4f}")
        logger.info(f"  sigma_fact: {sample.get('sigma_fact'):.4f}")
        logger.info(f"  sigma_ans: {sample.get('sigma_ans'):.4f}")


    # ===== CLI =====

def main():
    parser = argparse.ArgumentParser(description="Prepare TCR end-to-end training data")
    parser.add_argument("--n", type=int, default=DEFAULT_N_SAMPLES, help="Number of questions")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token")
    args = parser.parse_args()
    
    if args.hf_token:
        global HF_TOKEN
        HF_TOKEN = args.hf_token
        os.environ["HF_TOKEN"] = HF_TOKEN
    
    if not HF_TOKEN:
        print("Error: Please set HF_TOKEN via environment or --hf_token")
        sys.exit(1)
    
    prepare_training_data(n=args.n, n_workers=args.workers)
    verify_output()
    logger.info("\n[OK] Data preparation complete!")


if __name__ == "__main__":
    main()