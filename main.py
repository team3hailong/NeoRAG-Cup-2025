from docx import Document
from embeddings import Embeddings
from vector_db import VectorDatabase
import pandas as pd
import numpy as np
import os
from rerank import Reranker  

doc = Document("CLB_PROPTIT.docx")

# Sửa chỗ FIX_ME để dùng DB mà các em muốn hoặc các em có thể tự sửa code trong lớp VectorDatabase để dùng các DB khác

vector_db = VectorDatabase(db_type="chromadb")

# Có thể dùng:
# - Backtrack: model_name="Alibaba-NLP/gte-multilingual-base", type="sentence_transformers"
embedding = Embeddings(model_name="BAAI/bge-m3", type="sentence_transformers")

# - Backtrack: model_name="BAAI/bge-reranker-v2-m3"
reranker = Reranker(model_name="namdp-ptit/ViRanker") if True else None
use_query_expansion = True

# TODO: Embedding từng document trong file CLB_PROPTIT.docx và lưu vào DB. 
# Code dưới là sử dụng mongodb, các em có thể tự sửa lại cho phù hợp với DB mà mình đang dùng
#--------------------Code Lưu Embedding Document vào DB--------------------------

# Clear existing collection to rebuild with ColBERT embeddings
try:
    vector_db.drop_collection("information")
    print("Dropped existing collection to rebuild with ColBERT embeddings")
except:
    print("Collection doesn't exist or already empty")

cnt = 1
if vector_db.count_documents("information") == 0:
    print("Building database with ColBERT embeddings...")
    for para in doc.paragraphs:
        if para.text.strip():
            # Encode using M3 model: returns dict with 'dense_vecs', 'sparse_weights', 'colbert_vecs'
            passage_embeddings = embedding.encode(para.text)
            print(f"Document {cnt} - M3 embedding keys: {list(passage_embeddings.keys()) if isinstance(passage_embeddings, dict) else 'Not dict'}")
            
            # Prepare embedding for vector DB: use ColBERT multi-vector if available, else dense
            if isinstance(passage_embeddings, dict) and 'colbert_vecs' in passage_embeddings and embedding.use_colbert:
                embedding_for_db = np.array(passage_embeddings['colbert_vecs'])
                print(f"Using ColBERT embeddings - shape: {embedding_for_db.shape}")
            else:
                # dense_vecs for dict or list output
                dense = passage_embeddings.get('dense_vecs') if isinstance(passage_embeddings, dict) else passage_embeddings
                embedding_for_db = np.array(dense)
                print(f"Using dense embeddings - shape: {embedding_for_db.shape}")
            
            # Store into vector DB with cached M3 representations
            vector_db.insert_document(
                collection_name="information",
                document={
                    "title": f"Document {cnt}",
                    "information": para.text,
                    "embedding": embedding_for_db,
                    "m3_dense": passage_embeddings.get('dense_vecs') if isinstance(passage_embeddings, dict) else None,
                    "m3_sparse": passage_embeddings.get('lexical_weights') if isinstance(passage_embeddings, dict) else None
                }
            )
            cnt += 1
    print(f"Successfully inserted {cnt-1} documents with ColBERT embeddings")
else:
    print("Documents already exist in the database. Skipping insertion.")
#------------------------------------------------------------------------------------

# Các em có thể import từng hàm một để check kết quả, trick là nên chạy trên data nhỏ thôi để xem hàm có chạy đúng hay ko rồi mới chạy trên toàn bộ data

from metrics_rag import  ndcg_k, response_relevancy_k, hit_k, bleu_4_k, context_recall_k, rouge_l_k, string_presence_k, context_entities_recall_k, context_precision_k, noise_sensitivity_k, calculate_metrics_retrieval, calculate_metrics_llm_answer, recall_k

# df_retrieval_metrics = calculate_metrics_retrieval("CLB_PROPTIT.csv", "train_data_proptit.xlsx", embedding, vector_db, True) # đặt là True nếu là tập train, False là tập test
# df_llm_metrics = calculate_metrics_llm_answer("CLB_PROPTIT.csv", "train_data_proptit.xlsx", embedding, vector_db, True, reranker) # đặt là True nếu là tập train, False là tập test
# print(df_retrieval_metrics.head())
# print(df_llm_metrics.head())

print("string_presence_k@5:", string_presence_k("CLB_PROPTIT.csv", "test_data_proptit.xlsx", embedding, vector_db, k=5, reranker=reranker, use_query_expansion=use_query_expansion))