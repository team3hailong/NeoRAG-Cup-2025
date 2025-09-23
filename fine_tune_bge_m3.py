"""
Fine-tune BAAI/bge-m3 for domain retrieval using SentenceTransformers.

Data sources:
- CLB_PROPTIT.csv: columns [STT, Văn bản]
- train_data_proptit.xlsx: columns include [Query, Ground truth document, Ground truth answer]

Training objective:
- MultipleNegativesRankingLoss on (query, positive_passage) pairs.
- In-batch negatives.

Outputs:
- A SentenceTransformers checkpoint at outputs/bge-m3-finetuned-<timestamp>

Note:
- This fine-tunes the dense encoder path of bge-m3. It won’t emit ColBERT multi-vectors.
- To use it in the app, set Embeddings(model_name=<output_dir>, type="sentence_transformers").
"""

from __future__ import annotations

import argparse
import os
import time
import random
import signal
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, models, util
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

# Reduce tokenizer parallel worker warnings and potential Colab deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_corpus(corpus_csv: str) -> Dict[str, str]:
    df = pd.read_csv(corpus_csv)
    # Expect columns: STT, Văn bản
    id_col = "STT"
    text_col = "Văn bản"
    if id_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"CSV {corpus_csv} must contain columns '{id_col}' and '{text_col}'")
    corpus = {str(int(r[id_col])): str(r[text_col]) for _, r in df.iterrows()}
    return corpus


def load_pairs(train_xlsx: str, corpus: Dict[str, str], use_instruction_prefix: bool = True) -> List[InputExample]:
    df = pd.read_excel(train_xlsx)
    if "Query" not in df.columns or "Ground truth document" not in df.columns:
        raise ValueError("Train file must contain columns 'Query' and 'Ground truth document'")

    pairs: List[InputExample] = []
    for _, row in df.iterrows():
        query = str(row["Query"]).strip()
        gt = row["Ground truth document"]
        if pd.isna(query) or pd.isna(gt):
            continue
        # gt can be single id or comma-separated ids
        gt_list = []
        if isinstance(gt, str):
            for part in gt.split(","):
                part = part.strip()
                if part:
                    gt_list.append(part)
        else:
            try:
                gt_list.append(str(int(gt)))
            except Exception:
                continue
        for doc_id in gt_list:
            text = corpus.get(doc_id)
            if not text:
                continue
            if use_instruction_prefix:
                q = f"query: {query}"
                p = f"passage: {text}"
            else:
                q, p = query, text
            pairs.append(InputExample(texts=[q, p]))
    return pairs

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_corpus_matrix(model: SentenceTransformer, corpus: Dict[str, str], use_instruction_prefix: bool = True) -> Tuple[List[str], np.ndarray]:
    """Encode the whole corpus into a matrix for fast similarity and hard-negative mining.
    Returns (doc_ids_ordered, embeddings_matrix [N, D])
    """
    doc_ids = list(corpus.keys())
    texts = [f"passage: {corpus[did]}" if use_instruction_prefix else corpus[did] for did in doc_ids]
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64, show_progress_bar=True)
    return doc_ids, emb.astype(np.float32)


def build_tfidf_index(corpus: Dict[str, str], use_instruction_prefix: bool = True) -> Tuple[List[str], TfidfVectorizer, csr_matrix]:
    """Build a TF-IDF index for the corpus for lexical hard-negative mining.
    Returns (doc_ids, vectorizer, doc_term_matrix)
    """
    doc_ids = list(corpus.keys())
    texts = [f"passage: {corpus[did]}" if use_instruction_prefix else corpus[did] for did in doc_ids]
    vectorizer = TfidfVectorizer(max_features=100_000, ngram_range=(1, 2), lowercase=True)
    dtm = vectorizer.fit_transform(texts)
    return doc_ids, vectorizer, dtm


def mine_hard_negatives(
    model: SentenceTransformer,
    df_train: pd.DataFrame,
    corpus: Dict[str, str],
    corpus_ids_dense: Optional[List[str]] = None,
    corpus_matrix_dense: Optional[np.ndarray] = None,
    tfidf_tuple: Optional[Tuple[List[str], TfidfVectorizer, csr_matrix]] = None,
    max_negatives_per_pos: int = 1,
    mine_top_m: int = 50,
    use_instruction_prefix: bool = True,
    neg_source: str = "dense",
    hybrid_alpha: float = 0.6,
) -> List[InputExample]:
    """Mine hard negatives using dense, tfidf, or hybrid similarities.

    neg_source: one of {"dense", "tfidf", "hybrid"}
    hybrid_alpha: weight for dense score in hybrid (final = alpha*dense + (1-alpha)*tfidf)
    """
    triplets: List[InputExample] = []

    # Prepare indices
    dense_ok = corpus_matrix_dense is not None and corpus_ids_dense is not None
    tfidf_ok = tfidf_tuple is not None

    if neg_source not in {"dense", "tfidf", "hybrid"}:
        neg_source = "dense"

    if neg_source in {"dense", "hybrid"} and not dense_ok:
        raise ValueError("Dense mining requested but no dense corpus matrix provided")
    if neg_source in {"tfidf", "hybrid"} and not tfidf_ok:
        raise ValueError("TFIDF mining requested but no TFIDF index provided")

    if tfidf_ok:
        corpus_ids_tfidf, vectorizer, dtm = tfidf_tuple
        id_to_idx_tfidf = {did: i for i, did in enumerate(corpus_ids_tfidf)}
    else:
        corpus_ids_tfidf, vectorizer, dtm = [], None, None

    if dense_ok:
        id_to_idx_dense = {did: i for i, did in enumerate(corpus_ids_dense)}

    for _, row in df_train.iterrows():
        query = str(row.get("Query", "")).strip()
        gt = row.get("Ground truth document")
        if not query or pd.isna(gt):
            continue
        # Collect gt ids
        gt_ids: List[str] = []
        if isinstance(gt, str):
            for part in gt.split(","):
                p = part.strip()
                if p:
                    gt_ids.append(str(int(p)))
        else:
            try:
                gt_ids.append(str(int(gt)))
            except Exception:
                continue

        # Build query text with instruction prefix
        qtext = f"query: {query}" if use_instruction_prefix else query

        # Compute candidate scores
        dense_scores = None
        tfidf_scores = None

        if neg_source in {"dense", "hybrid"}:
            q_emb = model.encode(qtext, normalize_embeddings=True, convert_to_numpy=True)
            if q_emb is None:
                continue
            q_emb = q_emb.astype(np.float32)
            dense_scores = np.matmul(corpus_matrix_dense, q_emb)  # cosine with normalized vecs

        if neg_source in {"tfidf", "hybrid"} and vectorizer is not None and dtm is not None:
            q_vec = vectorizer.transform([qtext])  # 1 x V
            tfidf_scores = (dtm @ q_vec.T).toarray().squeeze().astype(np.float32)

        # Combine or choose scores
        if neg_source == "dense":
            sims = dense_scores
            doc_ids = corpus_ids_dense
        elif neg_source == "tfidf":
            sims = tfidf_scores
            doc_ids = corpus_ids_tfidf
        else:  # hybrid
            # Align ids by assuming same ordering (build both from same corpus dict)
            # If orders differ, fall back to corpus_ids_dense
            if corpus_ids_dense == corpus_ids_tfidf:
                sims = hybrid_alpha * dense_scores + (1 - hybrid_alpha) * tfidf_scores
                doc_ids = corpus_ids_dense
            else:
                # Fallback: use dense ordering
                sims = dense_scores
                doc_ids = corpus_ids_dense

        top_idx = np.argsort(-sims)[:max(mine_top_m, max_negatives_per_pos + len(gt_ids))]

        # Build triplets for each positive
        for pos_id in gt_ids:
            pos_text = corpus.get(pos_id)
            if not pos_text:
                continue
            # Find negatives among top neighbors not in GT
            negs: List[str] = []
            for idx in top_idx:
                cand_id = doc_ids[idx]
                if cand_id not in gt_ids:
                    negs.append(corpus.get(cand_id))
                if len(negs) >= max_negatives_per_pos:
                    break
            # Fallback: random negatives if needed
            if len(negs) < max_negatives_per_pos:
                for cand_id in random.sample(list(corpus.keys()), k=min(5, len(corpus))):
                    if cand_id not in gt_ids:
                        negs.append(corpus.get(cand_id))
                    if len(negs) >= max_negatives_per_pos:
                        break
            if not negs:
                continue

            for neg_text in negs:
                a = qtext
                p = f"passage: {pos_text}" if use_instruction_prefix else pos_text
                n = f"passage: {neg_text}" if use_instruction_prefix else neg_text
                triplets.append(InputExample(texts=[a, p, n]))
    return triplets


def build_ir_eval(df_train: pd.DataFrame, corpus: Dict[str, str]):
    # Build an IR evaluator from the same training split for quick sanity checks
    queries = {}
    relevant_docs = {}
    for i, row in df_train.iterrows():
        qid = f"q{i}"
        qtext = str(row["Query"]).strip()
        if not qtext:
            continue
        queries[qid] = qtext
        gt = row["Ground truth document"]
        doc_ids = set()
        if isinstance(gt, str):
            for part in gt.split(","):
                part = part.strip()
                if part:
                    doc_ids.add(str(int(part)))
        else:
            try:
                doc_ids.add(str(int(gt)))
            except Exception:
                pass
        if doc_ids:
            relevant_docs[qid] = doc_ids
    return InformationRetrievalEvaluator(queries=queries, corpus=corpus, relevant_docs=relevant_docs, name="train-quick")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_csv", default="CLB_PROPTIT.csv")
    parser.add_argument("--train_xlsx", default="train_data_proptit.xlsx")
    parser.add_argument("--base_model", default="BAAI/bge-m3")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_instruction_prefix", action="store_true", help="Disable 'query:/passage:' prefixes for training inputs")
    parser.add_argument("--max_steps", type=int, default=0, help="If >0, cap dataset to approx max_steps*batch_size for a quick run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev_ratio", type=float, default=0.2, help="Fraction of training rows used as dev for eval")
    parser.add_argument("--num_hard_negatives", type=int, default=1, help="Triplet negatives per (q,pos)")
    parser.add_argument("--mine_top_m", type=int, default=200, help="From top-M nearest neighbors choose negatives")
    parser.add_argument("--use_triplet", action="store_true", help="Add TripletLoss with mined hard negatives")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--neg_source", choices=["dense", "tfidf", "hybrid"], default="hybrid", help="Source for hard negative mining")
    parser.add_argument("--hybrid_alpha", type=float, default=0.6, help="Dense weight in hybrid negative mining")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Freeze bottom N transformer encoder layers for stability on small data")
    parser.add_argument("--scheduler", choices=["WarmupLinear", "WarmupCosine", "WarmupCosineWithRestarts", "WarmupConstant"], default="WarmupLinear")
    parser.add_argument("--evaluation_steps", type=int, default=0, help="Override evaluation steps interval; 0 means auto")
    parser.add_argument("--checkpoint_save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint_save_total_limit", type=int, default=2, help="Max number of checkpoints to keep")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint dir to resume from, or 'auto' to pick latest in output ckpts")
    # Colab/Windows friendly DataLoader options
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers; default auto (0 on Windows/Colab, else 2)")
    parser.add_argument("--pin_memory", type=str, default=None, help="Override pin_memory (auto by default). Use 'true'/'false'")
    parser.add_argument("--persistent_workers", action="store_true", help="Enable persistent_workers when num_workers>0 (off by default)")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision (AMP)")
    args = parser.parse_args()

    seed_everything(args.seed)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.output_dir or os.path.join("outputs", f"bge-m3-finetuned-{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print("[Info] Loading corpus and training pairs...")
    corpus = load_corpus(args.corpus_csv)
    full_df = pd.read_excel(args.train_xlsx)
    # Split dev
    if 0.0 < args.dev_ratio < 0.9:
        dev_size = max(1, int(len(full_df) * args.dev_ratio))
        df_dev = full_df.sample(n=dev_size, random_state=args.seed)
        df_train = full_df.drop(df_dev.index).reset_index(drop=True)
        df_dev = df_dev.reset_index(drop=True)
    else:
        df_train = full_df
        df_dev = None

    train_pairs = []
    # Build positive pairs (for MultipleNegativesRankingLoss)
    train_pairs = load_pairs(args.train_xlsx, corpus, use_instruction_prefix=not args.no_instruction_prefix)
    if len(train_pairs) == 0:
        raise RuntimeError("No training pairs constructed. Check your data files.")
    print(f"[Info] Training pairs: {len(train_pairs)} | Corpus docs: {len(corpus)}")

    def _str2bool(x: Optional[str]) -> Optional[bool]:
        if x is None:
            return None
        return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}

    print(f"[Info] Loading base model: {args.base_model}")
    # SentenceTransformers automatically uses GPU if available
    model = SentenceTransformer(args.base_model, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length

    # Optionally freeze bottom N layers to prevent catastrophic forgetting on small datasets
    def try_freeze_layers(st_model: SentenceTransformer, n_layers: int):
        if n_layers <= 0:
            return
        try:
            transformer = st_model._first_module()
            encoder = None
            if hasattr(transformer, "auto_model"):
                encoder = transformer.auto_model
            elif hasattr(transformer, "model"):
                encoder = transformer.model
            if encoder is None:
                return
            # BERT/RoBERTa style encoders
            if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
                layers = encoder.encoder.layer
                for i, layer in enumerate(layers):
                    if i < n_layers:
                        for p in layer.parameters():
                            p.requires_grad = False
                print(f"[Info] Froze {min(n_layers, len(layers))} bottom encoder layers")
            # Freeze embeddings as well if many layers frozen
            if hasattr(encoder, "embeddings") and n_layers > 0:
                for p in encoder.embeddings.parameters():
                    p.requires_grad = False
        except Exception as e:
            print(f"[Warn] Could not freeze layers: {e}")

    try_freeze_layers(model, args.freeze_layers)

    # If quick run requested, truncate the dataset size
    if args.max_steps and args.max_steps > 0:
        max_examples = max(args.batch_size, args.max_steps * args.batch_size)
        train_pairs = train_pairs[:max_examples]

    # Dataloader settings with sane defaults for Colab/Windows
    is_windows = os.name == "nt"
    # Try to detect Colab environment
    try:
        import google.colab  # type: ignore
        is_colab = True
    except Exception:
        is_colab = False

    auto_num_workers = 0 if (is_windows or is_colab) else 2
    num_workers = auto_num_workers if args.num_workers is None else int(args.num_workers)
    auto_pin = torch.cuda.is_available()
    pin_memory_override = _str2bool(args.pin_memory)
    pin_memory = auto_pin if pin_memory_override is None else bool(pin_memory_override)
    persistent_workers = bool(args.persistent_workers) and num_workers > 0

    train_dataloader = DataLoader(
        train_pairs,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Optional: add TripletLoss with hard-negative mining
    objectives = [(train_dataloader, train_loss)]
    if args.use_triplet:
        print("[Info] Mining hard negatives for TripletLoss...")
        # Precompute indices per chosen neg_source
        dense_ids, dense_matrix = None, None
        tfidf_tuple: Optional[Tuple[List[str], TfidfVectorizer, csr_matrix]] = None
        if args.neg_source in {"dense", "hybrid"}:
            dense_ids, dense_matrix = encode_corpus_matrix(model, corpus, use_instruction_prefix=not args.no_instruction_prefix)
        if args.neg_source in {"tfidf", "hybrid"}:
            tfidf_tuple = build_tfidf_index(corpus, use_instruction_prefix=not args.no_instruction_prefix)

        triplets = mine_hard_negatives(
            model=model,
            df_train=df_train,
            corpus=corpus,
            corpus_ids_dense=dense_ids,
            corpus_matrix_dense=dense_matrix,
            tfidf_tuple=tfidf_tuple,
            max_negatives_per_pos=args.num_hard_negatives,
            mine_top_m=args.mine_top_m,
            use_instruction_prefix=not args.no_instruction_prefix,
            neg_source=args.neg_source,
            hybrid_alpha=args.hybrid_alpha,
        )
        if args.max_steps and args.max_steps > 0:
            triplets = triplets[:max_examples]
        if len(triplets) > 0:
            print(f"[Info] Triplets mined: {len(triplets)}")
            triplet_loader = DataLoader(
                triplets,
                batch_size=max(4, args.batch_size // 2),
                shuffle=True,
                drop_last=True,
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
            )
            triplet_loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE, margin=0.25)
            objectives.append((triplet_loader, triplet_loss))
        else:
            print("[Warn] No triplets mined; skipping TripletLoss objective")

    # Evaluator: use dev split if available; else quick train evaluator
    if df_dev is not None and len(df_dev) > 0:
        evaluator = build_ir_eval(df_dev, corpus)
        print(f"[Info] Dev evaluator set with {len(df_dev)} queries")
    else:
        evaluator = build_ir_eval(df_train, corpus)
        print("[Info] Using train-quick evaluator (no dev split)")
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else max(10, int(len(train_dataloader) * args.epochs * 0.1))
    evaluation_steps = args.evaluation_steps if args.evaluation_steps > 0 else max(50, len(train_dataloader)//5)
    use_amp = False if args.no_amp else torch.cuda.is_available()

    # Optionally resume from latest checkpoint in output_dir/ckpts or provided path
    ckpt_root = os.path.join(out_dir, "ckpts")
    resume_dir: Optional[str] = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "auto":
            if os.path.isdir(ckpt_root):
                # pick latest by step number or mtime
                subdirs = [os.path.join(ckpt_root, d) for d in os.listdir(ckpt_root) if os.path.isdir(os.path.join(ckpt_root, d))]
                if subdirs:
                    # prefer highest numeric step at end of name
                    def _step_of(p: str) -> int:
                        import re
                        m = re.search(r"(\d+)$", os.path.basename(p))
                        return int(m.group(1)) if m else int(os.path.getmtime(p))
                    subdirs.sort(key=_step_of, reverse=True)
                    resume_dir = subdirs[0]
        else:
            if os.path.isdir(args.resume_from_checkpoint):
                resume_dir = args.resume_from_checkpoint
    if resume_dir:
        try:
            print(f"[Info] Resuming weights from checkpoint: {resume_dir}")
            model = SentenceTransformer(resume_dir, trust_remote_code=True)
            model.max_seq_length = args.max_seq_length
        except Exception as e:
            print(f"[Warn] Could not load checkpoint '{resume_dir}': {e}. Continuing from base model.")

    print(f"[Info] Starting training -> epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, scheduler={args.scheduler}, eval_steps={evaluation_steps}, amp={use_amp}")
    print(f"[Info] DataLoader settings: num_workers={num_workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}")
    print(f"[Info] Checkpoints will be saved to: {ckpt_root}")
    
    # Save emergency checkpoint before training starts
    pre_train_dir = os.path.join(out_dir, "pre-train-snapshot")
    try:
        model.save(pre_train_dir)
        print(f"[Info] Pre-training snapshot saved at: {pre_train_dir}")
    except Exception as e:
        print(f"[Warn] Could not save pre-training snapshot: {e}")
    
    try:
        fit_kwargs = dict(
            train_objectives=objectives,
            epochs=args.epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=evaluation_steps,
            use_amp=use_amp,
            output_path=out_dir,
            optimizer_params={'lr': args.lr},
            scheduler=args.scheduler,
            checkpoint_path=ckpt_root,
            checkpoint_save_steps=args.checkpoint_save_steps,
            save_best_model=True,
        )
        # Optional/newer kwargs
        if args.checkpoint_save_total_limit is not None:
            fit_kwargs["checkpoint_save_total_limit"] = args.checkpoint_save_total_limit
        if args.gradient_accumulation_steps is not None:
            fit_kwargs["gradient_accumulation_steps"] = max(1, int(args.gradient_accumulation_steps))

        # Call fit with graceful fallback for older sentence-transformers versions
        import re
        training_successful = False
        for attempt in range(4):
            try:
                print(f"[Info] Training attempt {attempt + 1}/4...")
                model.fit(**fit_kwargs)
                training_successful = True
                print("[Info] Training completed successfully!")
                break
            except TypeError as te:
                msg = str(te)
                m = re.search(r"unexpected keyword argument '([^']+)'", msg)
                if m:
                    bad_kw = m.group(1)
                    if bad_kw in fit_kwargs:
                        print(f"[Warn] sentence-transformers.fit does not support '{bad_kw}' in this environment. Retrying without it.")
                        del fit_kwargs[bad_kw]
                        continue
                # Not a simple unsupported kwarg case; re-raise to outer handler
                raise
        
        if not training_successful:
            print("[Error] All training attempts failed due to unsupported arguments.")
            
    except KeyboardInterrupt:
        print("[Warn] Training interrupted by KeyboardInterrupt (possibly due to Colab timeout/disconnect).")
        print("[Info] This can happen when:")
        print("  - Colab runtime disconnects due to inactivity")
        print("  - Browser tab becomes inactive for too long")  
        print("  - GPU runtime limit exceeded")
        print("  - Network connectivity issues")
        print("[Info] Trying to save an emergency snapshot...")
        try:
            emergency_dir = os.path.join(out_dir, f"interrupt-{time.strftime('%Y%m%d-%H%M%S')}")
            model.save(emergency_dir)
            print(f"[Info] Emergency snapshot saved at: {emergency_dir}")
            print(f"[Info] To resume: --resume_from_checkpoint '{emergency_dir}'")
        except Exception as e:
            print(f"[Warn] Failed to save emergency snapshot: {e}")
            # Try saving in a simpler location
            try:
                simple_dir = f"./emergency-{time.strftime('%Y%m%d-%H%M%S')}"
                model.save(simple_dir)
                print(f"[Info] Emergency snapshot saved at: {simple_dir}")
            except Exception as e2:
                print(f"[Warn] Also failed to save to simple location: {e2}")
    except Exception as e:
        import traceback
        print("[Error] Training crashed with exception:\n", ''.join(traceback.format_exception(type(e), e, e.__traceback__)))
        print("[Hint] Try --no_amp, reduce --batch_size, disable --use_triplet, or add --max_steps 10 for quick test.")

    print(f"[Done] Saved fine-tuned model to: {out_dir}")
    print("How to use in code: Embeddings(model_name=out_dir, type='sentence_transformers')")


if __name__ == "__main__":
    main()
