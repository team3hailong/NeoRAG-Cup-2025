"""
Fine-tune Cross-Encoder Reranker for ProPTIT domain using FlagEmbedding.

This script fine-tunes a reranker model (default: namdp-ptit/ViRanker or BAAI/bge-reranker-v2-m3)
on domain-specific data to improve ranking quality for the ProPTIT RAG system.

Data sources:
- CLB_PROPTIT.csv: columns [STT, Văn bản] - corpus of 99 documents
- train_data_proptit.xlsx: columns [Query, Ground truth document, Ground truth answer, Difficulty] - 100 queries

Training approach:
- Positive pairs: (query, ground_truth_doc)
- Hard negatives: Top-k retrieved documents that are NOT ground truth
- Soft negatives: Random sampled documents from corpus
- Loss: Cross-Entropy with pairwise ranking

Memory optimization for limited hardware (12.7GB RAM, 15GB GPU):
- Small batch size (4-8)
- Gradient accumulation (4-8 steps)
- Mixed precision training (FP16)
- Gradient checkpointing
- Optional: LoRA/PEFT for parameter-efficient fine-tuning
- Chunked data loading

Outputs:
- Fine-tuned reranker checkpoint at outputs/reranker-finetuned-<timestamp>

Usage in app:
    reranker = Reranker(model_name="outputs/reranker-finetuned-XXXXXX")
"""

from __future__ import annotations

import argparse
import os
import time
import random
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers import TrainingArguments, Trainer
from tqdm import tqdm

# Reduce tokenizer parallel worker warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def seed_everything(seed: int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_corpus(corpus_csv: str) -> Dict[str, str]:
    """Load corpus documents from CSV file.
    
    Args:
        corpus_csv: Path to CSV with columns [STT, Văn bản]
        
    Returns:
        Dictionary mapping doc_id (str) to document text (str)
    """
    df = pd.read_csv(corpus_csv)
    id_col = "STT"
    text_col = "Văn bản"
    if id_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"CSV {corpus_csv} must contain columns '{id_col}' and '{text_col}'")
    corpus = {str(int(r[id_col])): str(r[text_col]) for _, r in df.iterrows()}
    return corpus


def load_training_data(train_xlsx: str, corpus: Dict[str, str]) -> List[Dict]:
    """Load query-document pairs from training file.
    
    Args:
        train_xlsx: Path to Excel file with columns [Query, Ground truth document, ...]
        corpus: Dictionary of doc_id -> text
        
    Returns:
        List of dicts with keys: query, positive_docs (list of doc_ids)
    """
    df = pd.read_excel(train_xlsx)
    if "Query" not in df.columns or "Ground truth document" not in df.columns:
        raise ValueError("Train file must contain columns 'Query' and 'Ground truth document'")
    
    data = []
    for _, row in df.iterrows():
        query = str(row["Query"]).strip()
        gt = row["Ground truth document"]
        if pd.isna(query) or pd.isna(gt):
            continue
            
        # Parse ground truth doc ids (can be comma-separated)
        gt_ids = []
        if isinstance(gt, str):
            for part in gt.split(","):
                part = part.strip()
                if part and part in corpus:
                    gt_ids.append(part)
        else:
            try:
                doc_id = str(int(gt))
                if doc_id in corpus:
                    gt_ids.append(doc_id)
            except Exception:
                continue
        
        if gt_ids:
            data.append({
                "query": query,
                "positive_docs": gt_ids
            })
    
    return data


def mine_hard_negatives_bm25(
    query: str,
    positive_docs: List[str],
    corpus: Dict[str, str],
    top_k: int = 50
) -> List[str]:
    """Mine hard negatives using simple TF-IDF/BM25-like scoring.
    
    Returns list of doc_ids that are similar to query but NOT in positive_docs.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Build vectorizer on corpus
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]
    
    try:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        corpus_matrix = vectorizer.fit_transform(doc_texts)
        query_vec = vectorizer.transform([query])
        
        # Compute similarity scores
        scores = (corpus_matrix @ query_vec.T).toarray().squeeze()
        
        # Get top-k indices, excluding positives
        sorted_indices = np.argsort(-scores)
        hard_negs = []
        for idx in sorted_indices:
            doc_id = doc_ids[idx]
            if doc_id not in positive_docs:
                hard_negs.append(doc_id)
            if len(hard_negs) >= top_k:
                break
        
        return hard_negs
    except Exception as e:
        print(f"[Warn] Failed to mine hard negatives for query: {e}")
        # Fallback: random negatives
        all_neg_ids = [did for did in corpus.keys() if did not in positive_docs]
        return random.sample(all_neg_ids, min(top_k, len(all_neg_ids)))


class RerankerDataset(Dataset):
    """Dataset for training cross-encoder reranker.
    
    Each sample is a (query, passage, label) triplet:
    - label=1 for positive pairs (query, ground_truth_doc)
    - label=0 for negative pairs (query, hard_negative_doc or random_doc)
    """
    
    def __init__(
        self,
        data: List[Dict],
        corpus: Dict[str, str],
        tokenizer,
        max_length: int = 512,
        num_hard_negatives: int = 2,
        num_random_negatives: int = 1,
        hard_neg_top_k: int = 50,
    ):
        self.data = data
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_hard_negatives = num_hard_negatives
        self.num_random_negatives = num_random_negatives
        self.hard_neg_top_k = hard_neg_top_k
        
        # Pre-build training examples
        self.examples = self._build_examples()
    
    def _build_examples(self) -> List[Tuple[str, str, int]]:
        """Build list of (query, passage, label) tuples."""
        examples = []
        
        for item in tqdm(self.data, desc="Building reranker training examples"):
            query = item["query"]
            pos_docs = item["positive_docs"]
            
            # Add positive pairs
            for pos_id in pos_docs:
                pos_text = self.corpus[pos_id]
                examples.append((query, pos_text, 1))
            
            # Mine hard negatives
            hard_negs = mine_hard_negatives_bm25(
                query, pos_docs, self.corpus, top_k=self.hard_neg_top_k
            )
            
            # Sample hard negatives
            sampled_hard = random.sample(
                hard_negs,
                min(self.num_hard_negatives, len(hard_negs))
            )
            for neg_id in sampled_hard:
                neg_text = self.corpus[neg_id]
                examples.append((query, neg_text, 0))
            
            # Sample random negatives
            all_neg_ids = [did for did in self.corpus.keys() if did not in pos_docs]
            sampled_random = random.sample(
                all_neg_ids,
                min(self.num_random_negatives, len(all_neg_ids))
            )
            for neg_id in sampled_random:
                neg_text = self.corpus[neg_id]
                examples.append((query, neg_text, 0))
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        query, passage, label = self.examples[idx]
        
        # Tokenize query-passage pair
        encoding = self.tokenizer(
            query,
            passage,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float)
        }


def compute_metrics_reranker(eval_dataset, model, tokenizer, device, batch_size=8):
    """Compute evaluation metrics for reranker on dev set.
    
    Metrics:
    - Accuracy: fraction of correct binary classifications
    - Average Positive Score: mean score for positive pairs
    - Average Negative Score: mean score for negative pairs
    - Margin: difference between avg positive and avg negative scores
    """
    model.eval()
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            
            # For binary classification, use sigmoid
            scores = torch.sigmoid(logits).cpu().numpy()
            preds = (scores > 0.5).astype(int)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_scores.extend(scores)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    
    accuracy = (all_preds == all_labels).mean()
    
    pos_mask = all_labels == 1
    neg_mask = all_labels == 0
    
    avg_pos_score = all_scores[pos_mask].mean() if pos_mask.sum() > 0 else 0.0
    avg_neg_score = all_scores[neg_mask].mean() if neg_mask.sum() > 0 else 0.0
    margin = avg_pos_score - avg_neg_score
    
    return {
        "accuracy": float(accuracy),
        "avg_pos_score": float(avg_pos_score),
        "avg_neg_score": float(avg_neg_score),
        "margin": float(margin),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_csv", default="CLB_PROPTIT.csv")
    parser.add_argument("--train_xlsx", default="train_data_proptit.xlsx")
    parser.add_argument("--base_model", default="namdp-ptit/ViRanker", 
                        help="Base reranker model to fine-tune")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Training batch size per device (use small value for limited GPU)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Accumulate gradients over N steps to simulate larger batch")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev_ratio", type=float, default=0.2, 
                        help="Fraction of training data used for validation")
    parser.add_argument("--num_hard_negatives", type=int, default=2, 
                        help="Number of hard negatives per positive")
    parser.add_argument("--num_random_negatives", type=int, default=1,
                        help="Number of random negatives per positive")
    parser.add_argument("--hard_neg_top_k", type=int, default=50,
                        help="Consider top-K retrieved docs as hard negative candidates")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use mixed precision training (recommended for GPU memory)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps. -1 means train for full epochs")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    out_dir = args.output_dir or os.path.join("outputs", "reranker-finetuned")
    os.makedirs(out_dir, exist_ok=True)
    
    print("[Info] Loading corpus and training data...")
    corpus = load_corpus(args.corpus_csv)
    print(f"[Info] Loaded corpus: {len(corpus)} documents")
    
    full_data = load_training_data(args.train_xlsx, corpus)
    print(f"[Info] Loaded training queries: {len(full_data)}")
    
    # Split into train/dev
    if 0.0 < args.dev_ratio < 1.0:
        dev_size = max(1, int(len(full_data) * args.dev_ratio))
        random.shuffle(full_data)
        dev_data = full_data[:dev_size]
        train_data = full_data[dev_size:]
        print(f"[Info] Train/Dev split: {len(train_data)}/{len(dev_data)}")
    else:
        train_data = full_data
        dev_data = []
        print(f"[Info] No dev split, using all {len(train_data)} for training")
    
    # Load model and tokenizer
    print(f"[Info] Loading base reranker model: {args.base_model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,  # Regression task for ranking score
        trust_remote_code=True
    )
    
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            print("[Info] Gradient checkpointing enabled")
        except Exception as e:
            print(f"[Warn] Could not enable gradient checkpointing: {e}")
    
    model.to(device)
    
    # Build datasets
    print("[Info] Building training dataset...")
    train_dataset = RerankerDataset(
        data=train_data,
        corpus=corpus,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        num_hard_negatives=args.num_hard_negatives,
        num_random_negatives=args.num_random_negatives,
        hard_neg_top_k=args.hard_neg_top_k,
    )
    print(f"[Info] Training examples: {len(train_dataset)}")
    
    eval_dataset = None
    if dev_data:
        print("[Info] Building dev dataset...")
        eval_dataset = RerankerDataset(
            data=dev_data,
            corpus=corpus,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            num_hard_negatives=args.num_hard_negatives,
            num_random_negatives=args.num_random_negatives,
            hard_neg_top_k=args.hard_neg_top_k,
        )
        print(f"[Info] Dev examples: {len(eval_dataset)}")
    
    # Training arguments optimized for limited hardware
    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else args.save_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        fp16=args.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
        max_steps=args.max_steps,
        report_to="none",  # Disable wandb/tensorboard
        seed=args.seed,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("[Info] Starting training...")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  FP16: {args.fp16 and torch.cuda.is_available()}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    
    try:
        trainer.train()
        
        # Save final model
        print(f"[Info] Saving final model to {out_dir}")
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        
        # Save training info
        info = {
            "base_model": args.base_model,
            "corpus_size": len(corpus),
            "train_queries": len(train_data),
            "dev_queries": len(dev_data),
            "max_seq_length": args.max_seq_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.lr,
            "num_hard_negatives": args.num_hard_negatives,
            "num_random_negatives": args.num_random_negatives
        }
        with open(os.path.join(out_dir, "training_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        # Evaluate on dev set if available
        if eval_dataset:
            print("[Info] Computing final dev metrics...")
            metrics = compute_metrics_reranker(
                eval_dataset, model, tokenizer, device, batch_size=args.batch_size * 2
            )
            print(f"[Info] Dev metrics: {metrics}")
            with open(os.path.join(out_dir, "dev_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"[Done] Fine-tuned reranker saved to: {out_dir}")
        print(f"[Info] To use in code: Reranker(model_name='{out_dir}')")
        
    except KeyboardInterrupt:
        print("[Warn] Training interrupted by user")
        emergency_dir = os.path.join(out_dir, f"interrupt-{time.strftime('%Y%m%d-%H%M%S')}")
        print(f"[Info] Saving emergency checkpoint to {emergency_dir}")
        trainer.save_model(emergency_dir)
        tokenizer.save_pretrained(emergency_dir)
        print(f"[Info] Emergency checkpoint saved. Resume with --base_model '{emergency_dir}'")
    except Exception as e:
        import traceback
        print(f"[Error] Training failed: {e}")
        traceback.print_exc()
        print("[Hint] Try reducing --batch_size, increasing --gradient_accumulation_steps, or using --fp16")


if __name__ == "__main__":
    main()
