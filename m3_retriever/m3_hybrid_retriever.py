"""
M3 Hybrid Retrieval System
Combines dense, sparse (lexical), and ColBERT representations for optimal retrieval performance
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import torch
import pickle

class M3HybridRetriever:
    def __init__(self, embedding_model, vector_db, weights=(0.3, 0.2, 0.5)):
        """
        Initialize M3 Hybrid Retriever
        
        Args:
            embedding_model: M3 embedding model
            vector_db: Vector database instance
            weights: (dense_weight, sparse_weight, colbert_weight) for hybrid scoring
        """
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.dense_weight, self.sparse_weight, self.colbert_weight = weights
        
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
    
    def retrieve_hybrid(self, query: str, k: int = 5, collection_name: str = "information", candidate_pool: int = 100) -> List[Dict]:
        """
        Optimized hybrid retrieval using M3 representations
        """
        print(f"=== M3 Hybrid Retrieval for: '{query[:50]}...' ===")
        
        # Encode query
        query_embeddings = self.encode_query(query)

        results: List[Dict] = []

        client = self.vector_db.client
        collection = client.get_or_create_collection(name=collection_name)
        # Build a query vector for ANN candidate retrieval
        if query_embeddings.get('colbert') is not None:
            q_mean = np.mean(query_embeddings['colbert'], axis=0).tolist()
        else:
            q_mean = query_embeddings['dense']
        # Retrieve a small candidate pool using ANN
        cand = collection.query(
            query_embeddings=[q_mean],
            n_results=max(k * 10, candidate_pool),
            include=["documents", "metadatas", "distances"]
        )
        # Flatten returned lists
        ids = cand.get('ids', [[]])[0]
        docs = cand.get('documents', [[]])[0]
        metas = cand.get('metadatas', [[]])[0]
        for i in range(len(ids)):
            try:
                doc_title = ids[i]
                doc_text = docs[i]
                metadata = metas[i] or {}
                doc_dense = None
                doc_sparse = None
                doc_colbert = None
                if metadata.get("dense_cached"):
                    doc_dense = pickle.loads(bytes.fromhex(metadata["dense_cached"]))
                if metadata.get("sparse_cached"):
                    doc_sparse = pickle.loads(bytes.fromhex(metadata["sparse_cached"]))
                if metadata.get("colbert_embedding"):
                    doc_colbert = pickle.loads(bytes.fromhex(metadata["colbert_embedding"]))
                # As a last resort, encode just this candidate (not whole corpus)
                if doc_dense is None or doc_sparse is None or doc_colbert is None:
                    try:
                        enc = self.embedding_model.encode(doc_text)
                        doc_dense = doc_dense or enc.get('dense_vecs')
                        doc_sparse = doc_sparse or enc.get('lexical_weights', enc.get('sparse_weights', {}))
                        doc_colbert = doc_colbert or enc.get('colbert_vecs')
                    except Exception:
                        pass

                # Compute individual similarities (skip components that are missing)
                dense_sim = self.compute_dense_similarity(query_embeddings['dense'], doc_dense) if doc_dense is not None else 0.0
                sparse_sim = self.compute_sparse_similarity(query_embeddings['sparse'], doc_sparse) if doc_sparse else 0.0
                colbert_sim = self.compute_colbert_similarity(query_embeddings['colbert'], doc_colbert) if doc_colbert is not None else 0.0

                hybrid_score = (
                    self.dense_weight * dense_sim +
                    self.sparse_weight * sparse_sim +
                    self.colbert_weight * colbert_sim
                )
                results.append({
                    'title': doc_title,
                    'information': doc_text,
                    'score': hybrid_score,
                    'dense_score': dense_sim,
                    'sparse_score': sparse_sim,
                    'colbert_score': colbert_sim
                })
            except Exception as e:
                print(f"Error scoring candidate {i}: {e}")
                continue

        # Sort by hybrid score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        if results:
            print(f"Top result scores - Hybrid: {results[0]['score']:.4f} (Dense: {results[0]['dense_score']:.4f}, Sparse: {results[0]['sparse_score']:.4f}, ColBERT: {results[0]['colbert_score']:.4f})")
        
        return results[:k]

def create_m3_hybrid_retriever(embedding_model, vector_db, weights=(0.3, 0.2, 0.5)):
    """Factory function to create M3 hybrid retriever"""
    return M3HybridRetriever(embedding_model, vector_db, weights)