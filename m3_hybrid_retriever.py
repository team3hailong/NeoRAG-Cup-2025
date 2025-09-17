"""
M3 Hybrid Retrieval System
Combines dense, sparse (lexical), and ColBERT representations for optimal retrieval performance
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import torch
import pickle
import re

class M3HybridRetriever:
    def __init__(self, embedding_model, vector_db, weights=(0.4, 0.2, 0.4), *,
                 per_query_normalization: bool = True,
                 use_mmr: bool = True,
                 mmr_lambda: float = 0.7):
        """
        Initialize M3 Hybrid Retriever
        
        Args:
            embedding_model: M3 embedding model
            vector_db: Vector database instance
            weights: (dense_weight, sparse_weight, colbert_weight) for hybrid scoring
            per_query_normalization: If True, min-max normalize each component across candidate set
            use_mmr: Apply MMR re-ranking to encourage diversity (helps entity coverage & BLEU)
            mmr_lambda: Trade-off between relevance (λ) and diversity (1-λ)
        """
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.dense_weight, self.sparse_weight, self.colbert_weight = weights
        self.per_query_normalization = per_query_normalization
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        
    def encode_query(self, query: str) -> Dict[str, Any]:
        """Encode query with all M3 representations"""
        result = self.embedding_model.encode(query)
        return {
            'dense': result['dense_vecs'],
            'sparse': result['lexical_weights'], 
            'colbert': result['colbert_vecs']
        }
    
    def compute_dense_similarity(self, query_dense: np.ndarray, doc_dense: np.ndarray) -> float:
        """Compute cosine similarity between dense vectors"""
        if isinstance(query_dense, list):
            query_dense = np.array(query_dense)
        if isinstance(doc_dense, list):
            doc_dense = np.array(doc_dense)
            
        # Defensive checks and normalization
        try:
            q_norm = np.linalg.norm(query_dense)
            d_norm = np.linalg.norm(doc_dense)
            if q_norm == 0 or d_norm == 0 or np.isnan(q_norm) or np.isnan(d_norm):
                return 0.0
            query_norm = query_dense / q_norm
            doc_norm = doc_dense / d_norm
            sim = float(np.dot(query_norm, doc_norm))
            # Cosine similarity in [-1,1] -> map to [0,1]
            sim = max(min(sim, 1.0), -1.0)
            return float((sim + 1.0) / 2.0)
        except Exception:
            return 0.0
    
    def compute_sparse_similarity(self, query_sparse: Dict, doc_sparse: Dict) -> float:
        """Compute sparse (lexical) similarity"""
        # Expect dicts mapping token->weight (could be TF-IDF-like)
        if not query_sparse or not doc_sparse:
            return 0.0

        # Build aligned vectors for tokens intersection and compute cosine-like score
        try:
            tokens = set(query_sparse.keys()) & set(doc_sparse.keys())
            if not tokens:
                return 0.0
            q_vals = np.array([float(query_sparse[t]) for t in tokens])
            d_vals = np.array([float(doc_sparse[t]) for t in tokens])

            q_norm = np.linalg.norm(q_vals)
            d_norm = np.linalg.norm(d_vals)
            if q_norm == 0 or d_norm == 0 or np.isnan(q_norm) or np.isnan(d_norm):
                return 0.0

            cos = float(np.dot(q_vals, d_vals) / (q_norm * d_norm))
            # Map from [-1,1] to [0,1]
            cos = max(min(cos, 1.0), -1.0)
            return float((cos + 1.0) / 2.0)
        except Exception:
            return 0.0
    
    def compute_colbert_similarity(self, query_colbert: np.ndarray, doc_colbert: np.ndarray) -> float:
        """Compute ColBERT max-sim similarity"""
        try:
            sim = float(self.embedding_model.compute_colbert_similarity(query_colbert, doc_colbert))
            # ColBERT score often in [-1,1] or [0,1] depending on implementation; clamp and map to [0,1]
            if np.isnan(sim) or np.isinf(sim):
                return 0.0
            sim = max(min(sim, 1.0), -1.0)
            return float((sim + 1.0) / 2.0)
        except Exception as e:
            print(f"ColBERT similarity error: {e}")
            return 0.0
    
    def retrieve_hybrid(self, query: str, k: int = 5, collection_name: str = "information") -> List[Dict]:
        """
        Optimized hybrid retrieval using M3 representations
        """
        print(f"=== M3 Hybrid Retrieval for: '{query[:50]}...' ===")
        
        # Encode query
        query_embeddings = self.encode_query(query)
        
        # Get all documents from vector DB
        all_results = self.vector_db.client.get_or_create_collection(name=collection_name).get(
            include=['documents', 'metadatas', 'embeddings']
        )
        
        results = []
        re_encoding_count = 0
        
        for i in range(len(all_results['ids'])):
            try:
                metadata = all_results['metadatas'][i]
                doc_text = all_results['documents'][i]
                doc_title = all_results['ids'][i]
                
                # Check if we have cached M3 representations in metadata
                if (metadata.get("is_colbert", False) and 
                    "dense_cached" in metadata and 
                    "sparse_cached" in metadata):
                    # Use cached embeddings
                    doc_colbert = pickle.loads(bytes.fromhex(metadata["colbert_embedding"]))
                    doc_dense = pickle.loads(bytes.fromhex(metadata["dense_cached"]))
                    doc_sparse = pickle.loads(bytes.fromhex(metadata["sparse_cached"]))
                else:
                    # Need to re-encode - this is expensive
                    doc_m3 = self.embedding_model.encode(doc_text)
                    doc_dense = doc_m3['dense_vecs']
                    doc_sparse = doc_m3['lexical_weights']
                    if metadata.get("is_colbert", False):
                        doc_colbert = pickle.loads(bytes.fromhex(metadata["colbert_embedding"]))
                    else:
                        doc_colbert = doc_m3['colbert_vecs']
                    re_encoding_count += 1
                
                # Compute individual similarities
                dense_sim = self.compute_dense_similarity(query_embeddings['dense'], doc_dense)
                sparse_sim = self.compute_sparse_similarity(query_embeddings['sparse'], doc_sparse)
                colbert_sim = self.compute_colbert_similarity(query_embeddings['colbert'], doc_colbert)

                # Ensure all component scores are finite and in [0,1]
                for s in (dense_sim, sparse_sim, colbert_sim):
                    if s is None or np.isnan(s) or np.isinf(s):
                        s = 0.0

                # Re-normalize weights to sum to 1 in case they were configured differently
                w_sum = float(self.dense_weight + self.sparse_weight + self.colbert_weight)
                if w_sum == 0:
                    w_sum = 1.0

                hybrid_score = (
                    (self.dense_weight / w_sum) * float(dense_sim) +
                    (self.sparse_weight / w_sum) * float(sparse_sim) +
                    (self.colbert_weight / w_sum) * float(colbert_sim)
                )
                results.append({
                    'title': doc_title,
                    'information': doc_text,
                    'score': hybrid_score,              # preliminary score
                    'dense_score': dense_sim,
                    'sparse_score': sparse_sim,
                    'colbert_score': colbert_sim
                })
            except Exception as e:
                print(f"Error processing document {i}: {e}")
                continue
        
        if not results:
            return []

        # --- Per-query component normalization (min-max) ---
        if self.per_query_normalization:
            def min_max(values):
                v = np.array(values, dtype=float)
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                vmin, vmax = float(v.min()), float(v.max())
                if vmax - vmin < 1e-8:
                    return [0.0] * len(v)
                return [(x - vmin) / (vmax - vmin) for x in v]

            dense_norm = min_max([r['dense_score'] for r in results])
            sparse_norm = min_max([r['sparse_score'] for r in results])
            colbert_norm = min_max([r['colbert_score'] for r in results])

            w_sum = float(self.dense_weight + self.sparse_weight + self.colbert_weight) or 1.0
            dw, sw, cw = self.dense_weight / w_sum, self.sparse_weight / w_sum, self.colbert_weight / w_sum
            for idx, r in enumerate(results):
                r['dense_score_norm'] = dense_norm[idx]
                r['sparse_score_norm'] = sparse_norm[idx]
                r['colbert_score_norm'] = colbert_norm[idx]
                r['score'] = dw * dense_norm[idx] + sw * sparse_norm[idx] + cw * colbert_norm[idx]

        # Sort after normalization
        results.sort(key=lambda x: x['score'], reverse=True)

        # --- Optional MMR diversity re-ranking ---
        if self.use_mmr and k > 1:
            # Light-weight token-based Jaccard similarity to avoid re-encoding
            token_cache = {}
            word_pattern = re.compile(r"\w+", re.UNICODE)
            def tokens(text: str):
                if text not in token_cache:
                    token_cache[text] = set(w.lower() for w in word_pattern.findall(text))
                return token_cache[text]
            def jaccard(a: str, b: str) -> float:
                ta, tb = tokens(a), tokens(b)
                if not ta or not tb:
                    return 0.0
                inter = len(ta & tb)
                union = len(ta | tb)
                return inter / union if union else 0.0

            selected: List[Dict] = []
            candidate_pool = results[: min(len(results), max(k * 4, k + 5))]  # broaden pool for diversity
            while candidate_pool and len(selected) < k:
                if not selected:
                    chosen = candidate_pool.pop(0)
                    chosen['mmr_score'] = chosen['score']
                    selected.append(chosen)
                    continue
                best_doc = None
                best_mmr = -1e9
                for cand in candidate_pool:
                    redundancy = 0.0
                    for s in selected:
                        redundancy = max(redundancy, jaccard(cand['information'], s['information']))
                    mmr_score = self.mmr_lambda * cand['score'] - (1 - self.mmr_lambda) * redundancy
                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best_doc = cand
                if best_doc is None:
                    break
                best_doc['mmr_score'] = best_mmr
                selected.append(best_doc)
                candidate_pool.remove(best_doc)
            results = selected
        
        if results:
            print(f"Top result scores - Hybrid: {results[0]['score']:.4f} (Dense: {results[0]['dense_score']:.4f}, Sparse: {results[0]['sparse_score']:.4f}, ColBERT: {results[0]['colbert_score']:.4f})")
        
        return results[:k]

def create_m3_hybrid_retriever(embedding_model, vector_db):
    """Factory function to create M3 hybrid retriever"""
    return M3HybridRetriever(embedding_model, vector_db)