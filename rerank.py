# Alibaba-NLP/gte-multilingual-base > BAAI/bge-m3 
# Nhưng Alibaba-NLP/gte-multilingual-base + Alibaba-NLP/gte-multilingual-reranker-base 
# < BAAI/bge-m3 + BAAI/bge-reranker-v2-m3

from FlagEmbedding import FlagReranker
import torch
import os

class Reranker:
    def __init__(self, model_name: str = "namdp-ptit/ViRanker", use_fp16: bool = True, normalize: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        effective_fp16 = use_fp16 and (self.device == "cuda")
        self.normalize = normalize
        
        # Check if model_name is a local fine-tuned model path
        is_local_model = os.path.isdir(model_name) and os.path.exists(os.path.join(model_name, "config.json"))
        
        if is_local_model:
            print(f"[Reranker] Loading fine-tuned model from: {model_name}")
        else:
            print(f"[Reranker] Loading pretrained model: {model_name}")
        
        try:
            self.reranker = FlagReranker(model_name, use_fp16=effective_fp16)
            self.reranker.model.to(self.device)
            print(f"[Reranker] Using device: {self.device} | FP16: {effective_fp16}")
        except Exception as e:
            print(f"[Error] Failed to load reranker model '{model_name}': {e}")
            print("[Info] Falling back to namdp-ptit/ViRanker")
            self.reranker = FlagReranker("namdp-ptit/ViRanker", use_fp16=effective_fp16)
            self.reranker.model.to(self.device)

    def __call__(self, query: str, passages: list[str]) -> tuple[list[float], list[str]]:
        # Tạo cặp [query, passage] cho mỗi passage
        query_passage_pairs = [[query, passage] for passage in passages]

        scores = self.reranker.compute_score(query_passage_pairs, normalize=self.normalize)

        # Sắp xếp passage theo điểm số giảm dần
        ranked_data = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
        ranked_scores, ranked_passages = zip(*ranked_data)

        # Đảm bảo đầu ra là list chuẩn
        return list(ranked_scores), list(ranked_passages)