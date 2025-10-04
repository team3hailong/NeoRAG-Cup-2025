import torch
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel

load_dotenv()

# Các em có thể tự thêm embedding model mới hoặc dùng các model có sẵn
class Embeddings:
    def __init__(self, model_name, type):
        self.model_name = model_name
        self.type = type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"[Embeddings] Using device: {self.device}")
        self.use_bge_instruction = True
        try:
            self.client = SentenceTransformer(
                model_name,
                device=self.device,
                trust_remote_code=True
            )
            self.use_colbert = False
        except Exception as e:
            print(f"[Error] Failed to load fine-tuned model '{model_name}': {e}")
            print(f"[Warning] Falling back to base BGE-M3 model")
            # Fallback to base model
            self.client = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
            self.use_colbert = True

    def _maybe_prefix(self, text: str, is_query: bool) -> str:
        if not isinstance(text, str):
            text = str(text)
        if not self.use_bge_instruction:
            return text
        return ("query: " + text) if is_query else ("passage: " + text)

    def _l2_normalize(self, vec):
        if isinstance(vec, np.ndarray):
            v = vec
        else:
            v = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return (v / norm).astype(np.float32)

    def encode(self, doc, is_query: bool = None):
        if self.type == "sentence_transformers":
            if self.model_name == "BAAI/bge-m3" and self.use_colbert:
                output = self.client.encode(
                    doc,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True
                )
                return output
            else:
                if is_query is None:
                    if isinstance(doc, str):
                        is_query = doc.strip().endswith('?') or len(doc) < 100
                    else:
                        is_query = False
                text = self._maybe_prefix(doc, is_query=is_query)
                embedding = self.client.encode(text, normalize_embeddings=True)
                if isinstance(embedding, dict):
                    embedding = embedding.get('dense_vecs', embedding)
                import numpy as _np
                if isinstance(embedding, _np.ndarray):
                    embedding = embedding.tolist()
                try:
                    emb_np = np.array(embedding, dtype=np.float32)
                    embedding = self._l2_normalize(emb_np).tolist()
                except Exception:
                    pass
                return embedding
    
    def compute_colbert_similarity(self, query_vecs, doc_vecs):
        if self.model_name == "BAAI/bge-m3" and self.use_colbert:
            return self.client.colbert_score(query_vecs, doc_vecs)
        else:
            if isinstance(query_vecs, np.ndarray):
                query_vecs = torch.from_numpy(query_vecs)
            if isinstance(doc_vecs, np.ndarray):
                doc_vecs = torch.from_numpy(doc_vecs)
            
            query_vecs = torch.nn.functional.normalize(query_vecs, p=2, dim=-1)
            doc_vecs = torch.nn.functional.normalize(doc_vecs, p=2, dim=-1)
            
            similarity_matrix = torch.matmul(query_vecs, doc_vecs.transpose(0, 1))
            max_sim_per_query_token = torch.max(similarity_matrix, dim=1)[0]
            final_score = torch.mean(max_sim_per_query_token).item()
            
            return final_score