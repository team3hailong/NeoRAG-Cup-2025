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

#--------------------Chọn embedding & reranker model--------------------
emb_finetuned_path = 'none'
if emb_finetuned_path and os.path.isdir(emb_finetuned_path):
    print(f"[Embeddings] Using fine-tuned checkpoint at: {emb_finetuned_path}")
    embedding = Embeddings(model_name=emb_finetuned_path, type="sentence_transformers")
else:
    embedding = Embeddings(model_name="halobiron/bge-m3-embedding-PROPTIT-domain-ft", type="sentence_transformers")


rr_finetuned_path = 'none'
if rr_finetuned_path and os.path.isdir(rr_finetuned_path):
    print(f"[Reranker] Using fine-tuned checkpoint at: {rr_finetuned_path}")
    reranker = Reranker(model_name=rr_finetuned_path)
else:
    print("Use base reranker")
    reranker = Reranker(model_name="halobiron/ViRanker-PROPTIT-domain-ft")
#------------------------------------------------------------   

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

from metrics_rag import  precision_k, map_k, hit_k, bleu_4_k, context_recall_k, rouge_l_k, string_presence_k, context_entities_recall_k, context_precision_k, noise_sensitivity_k, calculate_metrics_retrieval, calculate_metrics_llm_answer, recall_k

df_llm_metrics = calculate_metrics_llm_answer("CLB_PROPTIT.csv", "train_data_proptit.xlsx", embedding, vector_db, True, reranker=reranker, use_query_expansion=use_query_expansion) # đặt là True nếu là tập train, False là tập test
print(df_llm_metrics.head())