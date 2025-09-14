"""
Uses only ColBERT similarity for faster retrieval while maintaining good performance
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import pickle

class FastM3Retriever:
    def __init__(self, embedding_model, vector_db):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        
    def retrieve_colbert_only(self, query: str, k: int = 5, collection_name: str = "information") -> List[Dict]:
        print(f"=== Fast ColBERT Retrieval for: '{query[:50]}...' ===")
        
        query_embeddings = self.embedding_model.encode(query)
        query_colbert = query_embeddings['colbert_vecs']
        
        all_results = self.vector_db.client.get_or_create_collection(name=collection_name).get(
            include=['documents', 'metadatas']
        )
        
        results = []
        processed_count = 0
        
        for i in range(len(all_results['ids'])):
            try:
                metadata = all_results['metadatas'][i]
                doc_text = all_results['documents'][i]
                doc_title = all_results['ids'][i]
                
                # Get ColBERT embedding from metadata
                if metadata.get("is_colbert", False):
                    doc_colbert = pickle.loads(bytes.fromhex(metadata["colbert_embedding"]))
                    
                    colbert_sim = float(self.embedding_model.compute_colbert_similarity(query_colbert, doc_colbert))
                    
                    results.append({
                        'title': doc_title,
                        'information': doc_text,
                        'score': colbert_sim
                    })
                    
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing document {i}: {e}")
                continue
        
        # Sort by ColBERT score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        if results:
            print(f"Top ColBERT score: {results[0]['score']:.4f}")
        
        return results[:k]

def create_fast_m3_retriever(embedding_model, vector_db):
    return FastM3Retriever(embedding_model, vector_db)