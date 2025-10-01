from docx import Document
from embeddings import Embeddings
from vector_db import VectorDatabase
from ingest_utils import build_collection_from_docx
import pandas as pd
import numpy as np
import os
from rerank import Reranker  

# Sửa chỗ FIX_ME để dùng DB mà các em muốn hoặc các em có thể tự sửa code trong lớp VectorDatabase để dùng các DB khác

vector_db = VectorDatabase(db_type="chromadb")

#--------------------Chọn embedding model--------------------
is_finetuned = True
if is_finetuned:
    print("[Embeddings] Using fine-tuned embeddings")
    embedding = Embeddings(model_name="halobiron/bge-m3-embedding-PROPTIT-domain-ft", type="sentence_transformers")
else:
    embedding = Embeddings(model_name="BAAI/bge-m3", type="sentence_transformers")

#--------------------Chọn reranker model--------------------
# Tự động tìm fine-tuned reranker model mới nhất
import glob
finetuned_reranker_dirs = sorted(glob.glob("outputs/reranker-finetuned"), reverse=True)
if finetuned_reranker_dirs and os.path.exists(os.path.join(finetuned_reranker_dirs[0], "config.json")):
    print(f"[Reranker] Using fine-tuned reranker: {finetuned_reranker_dirs[0]}")
    reranker = Reranker(model_name=finetuned_reranker_dirs[0])
else:
    # Fallback to pretrained models
    # Các lựa chọn: "namdp-ptit/ViRanker", "BAAI/bge-reranker-v2-m3"
    print("[Reranker] Using pretrained reranker: namdp-ptit/ViRanker")
    reranker = Reranker(model_name="namdp-ptit/ViRanker")

use_query_expansion = True

#--------------------Build collection using shared ingest utility--------------------
inserted = build_collection_from_docx(
    doc_path="CLB_PROPTIT.docx",
    embedding_model=embedding,
    vector_db=vector_db,
    collection_name="information",
    rebuild=True,
)
current_count = vector_db.count_documents("information")
print(f"Rebuilt collection 'information': inserted={inserted}, total={current_count}")
#-----------------------------------------------------------------------------------

# Các em có thể import từng hàm một để check kết quả, trick là nên chạy trên data nhỏ thôi để xem hàm có chạy đúng hay ko rồi mới chạy trên toàn bộ data

from metrics_rag import  precision_k, groundedness_k, hit_k, bleu_4_k, context_recall_k, rouge_l_k, string_presence_k, context_entities_recall_k, context_precision_k, noise_sensitivity_k, calculate_metrics_retrieval, calculate_metrics_llm_answer, recall_k

print("precision_k@5:", precision_k("CLB_PROPTIT.csv", "test_data_proptit.xlsx", embedding, vector_db, k=5, reranker=reranker, use_query_expansion=use_query_expansion))
print("hit_k@5:", hit_k("CLB_PROPTIT.csv", "test_data_proptit.xlsx", embedding, vector_db, k=5, reranker=reranker, use_query_expansion=use_query_expansion))