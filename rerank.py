# Alibaba-NLP/gte-multilingual-base > BAAI/bge-m3 
# Nhưng Alibaba-NLP/gte-multilingual-base + Alibaba-NLP/gte-multilingual-reranker-base 
# < BAAI/bge-m3 + BAAI/bge-reranker-v2-m3

# from sentence_transformers import CrossEncoder
# import numpy as np

# class Reranker():
#     def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
#         self.reranker = CrossEncoder(model_name, trust_remote_code=True)

#     def __call__(self, query: str, passages: list[str]) -> tuple[list[float], list[str]]:
#         # Combine query and passages into pairs
#         query_passage_pairs = [[query, passage] for passage in passages]

#         # Get scores from the reranker model
#         scores = self.reranker.predict(query_passage_pairs)

#         # Sort passages based on scores
#         ranked_passages = [passage for _, passage in sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)]
#         ranked_scores = sorted(scores, reverse=True)
        
#         # Convert scores to standard Python floats
#         ranked_scores = [float(score) for score in ranked_scores]
#         # Return just the passages in ranked order
#         return ranked_scores, ranked_passages
    

# Model anh Nam

from FlagEmbedding import FlagReranker
import torch
class Reranker:
    def __init__(self, model_name: str = "namdp-ptit/ViRanker", use_fp16: bool = True, normalize: bool = True):
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)
        self.normalize = normalize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker.model.to(self.device)
        print("Using device:", self.device)

    def __call__(self, query: str, passages: list[str]) -> tuple[list[float], list[str]]:
        # Tạo cặp [query, passage] cho mỗi passage
        query_passage_pairs = [[query, passage] for passage in passages]

        scores = self.reranker.compute_score(query_passage_pairs, normalize=self.normalize)

        # Sắp xếp passage theo điểm số giảm dần
        ranked_data = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
        ranked_scores, ranked_passages = zip(*ranked_data)

        # Đảm bảo đầu ra là list chuẩn
        return list(ranked_scores), list(ranked_passages)