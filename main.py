from docx import Document
from embeddings import Embeddings
from vector_db import VectorDatabase
import pandas as pd
import openai
import os

doc = Document("CLB_PROPTIT.docx")

# Sửa chỗ FIX_ME để dùng DB mà các em muốn hoặc các em có thể tự sửa code trong lớp VectorDatabase để dùng các DB khác

vector_db = VectorDatabase(db_type="chromadb")

# Có thể dùng: 
# - Gemini: model_name="text-embedding-004", type="gemini"
# - OpenAI: model_name="text-embedding-3-large", type="openai" 
# - Ollama: model_name="nomic-embed-text", type="ollama"
# - Sentence Transformers: model_name="BAAI/bge-m3", type="sentence_transformers"
embedding = Embeddings(model_name="Alibaba-NLP/gte-multilingual-base", type="sentence_transformers")

# TODO: Embedding từng document trong file CLB_PROPTIT.docx và lưu vào DB. 
# Code dưới là sử dụng mongodb, các em có thể tự sửa lại cho phù hợp với DB mà mình đang dùng
#--------------------Code Lưu Embedding Document vào DB--------------------------
cnt = 1
if vector_db.count_documents("information") == 0:
    for para in doc.paragraphs:
        if para.text.strip():
            embedding_vector = embedding.encode(para.text)
            # Lưu vào cơ sở dữ liệu
            vector_db.insert_document(
                collection_name="information",
                document={
                    "title": f"Document {cnt}",
                    "information": para.text,
                    "embedding": embedding_vector
                }
            )
            cnt += 1
else:
    print("Documents already exist in the database. Skipping insertion.")
#------------------------------------------------------------------------------------

# Các em có thể import từng hàm một để check kết quả, trick là nên chạy trên data nhỏ thôi để xem hàm có chạy đúng hay ko rồi mới chạy trên toàn bộ data

from metrics_rag import calculate_metrics_retrieval, calculate_metrics_llm_answer, hit_k, recall_k, precision_k

# df_retrieval_metrics = calculate_metrics_retrieval("CLB_PROPTIT.csv", "train_data_proptit.xlsx", embedding, vector_db, True) # đặt là True nếu là tập train, False là tập test
# df_llm_metrics = calculate_metrics_llm_answer("CLB_PROPTIT.csv", "train_data_proptit.xlsx", embedding, vector_db, True) # đặt là True nếu là tập train, False là tập test
# print(df_retrieval_metrics.head())
# print(df_llm_metrics.head())

print("Hit@5:", hit_k("CLB_PROPTIT.csv", "train_data_proptit.xlsx", embedding, vector_db, k=5))