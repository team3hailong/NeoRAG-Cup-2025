import torch
import os
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
        if model_name == "BAAI/bge-m3":
            self.client = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
            self.use_colbert = True  
        elif type == "sentence_transformers":
            self.client = SentenceTransformer(
                model_name,
                device=self.device,
                trust_remote_code=True
            )
            self.use_colbert = False

    def encode(self, doc):
        if self.type in ["openai", "ollama"]:
            return self.client.embeddings.create(
                input=doc,
                model=self.model_name
            ).data[0].embedding
        elif self.type == "sentence_transformers":
            if self.model_name == "BAAI/bge-m3" and self.use_colbert:
                output = self.client.encode(
                    doc,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True
                )
                return output
            else:
                embedding = self.client.encode(
                    doc, return_dense=True, return_sparse=False, return_colbert_vecs=False
                )
                if isinstance(embedding, dict):
                    embedding = embedding.get('dense_vecs', embedding)
                return embedding.tolist()
        elif self.type == "gemini":
            return self.client.models.embed_content(
                model=self.model_name,
                contents=doc
            ).embeddings[0].values
    
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