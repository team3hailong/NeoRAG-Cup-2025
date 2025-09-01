from docx import Document  
from rank_bm25 import BM25Okapi
import numpy as np

# Load and tokenize documents
_doc = Document("CLB_PROPTIT.docx")
_texts = [para.text for para in _doc.paragraphs if para.text.strip()]
_tokenized_texts = [t.split() for t in _texts]
_bmt = BM25Okapi(_tokenized_texts)


def hybrid_retrieve(query, embedding, k_sparse: int = 20, k: int = 5):
    """
    Hybrid retrieval: sparse BM25 to select top-k_sparse, then dense re-rank to top-k.
    Returns list of dicts with 'information' field.
    """
    # Sparse retrieval
    tokens = query.split()
    scores = _bmt.get_scores(tokens)
    top_idxs = np.argsort(scores)[::-1][:k_sparse]
    candidates = [_texts[i] for i in top_idxs]
    # Dense re-ranking
    q_emb = embedding.encode(query)
    cand_embs = [embedding.encode(c) for c in candidates]
    sims = [np.dot(q_emb, ce) / (np.linalg.norm(q_emb) * np.linalg.norm(ce) + 1e-8) for ce in cand_embs]
    top2 = np.argsort(sims)[::-1][:k]
    return [{"information": candidates[i]} for i in top2]
