# Query Expansion Techniques for NeoRAG Cup 2025

## 📋 Tổng quan

Đã áp dụng **Query Expansion techniques** vào hệ thống RAG để cải thiện hiệu suất retrieval cho NeoRAG Cup 2025. Hệ thống bao gồm 5 kỹ thuật chính:

### 🔧 Các Kỹ Thuật Query Expansion

1. **Query Rewriting** - Viết lại câu hỏi với nhiều cách diễn đạt khác nhau
2. **Query Decomposition** - Phân tách câu hỏi phức tạp thành các câu hỏi con đơn giản  
3. **Synonym/Paraphrase Expansion** - Mở rộng với từ đồng nghĩa và cách diễn đạt khác
4. **Context-Aware Expansion** - Mở rộng dựa trên ngữ cảnh cụ thể của CLB ProPTIT
5. **Multi-Perspective Query** - Tạo các góc nhìn khác nhau cho cùng một câu hỏi

## 🚀 Cách Sử Dụng

### 1. Test Query Expansion đơn lẻ:

```python
from query_expansion import QueryExpansion

# Khởi tạo
expander = QueryExpansion()

# Test một câu hỏi
query = "CLB ProPTIT có những hoạt động gì?"

# Áp dụng tất cả kỹ thuật
expanded_queries = expander.expand_query(query, max_expansions=8)
print(expanded_queries)
```

### 2. Sử dụng trong pipeline RAG:

```python
from metrics_rag import hit_k, recall_k, precision_k

# Với Query Expansion
hit_score_enhanced = hit_k(
    "CLB_PROPTIT.csv", 
    "train_data_proptit.xlsx", 
    embedding, 
    vector_db, 
    reranker=None, 
    k=5, 
    use_query_expansion=True  # 🔥 Bật Query Expansion
)

# Không có Query Expansion (baseline)
hit_score_baseline = hit_k(
    "CLB_PROPTIT.csv", 
    "train_data_proptit.xlsx", 
    embedding, 
    vector_db, 
    reranker=None, 
    k=5, 
    use_query_expansion=False  # Tắt Query Expansion
)

print(f"Baseline: {hit_score_baseline:.3f}")
print(f"Enhanced: {hit_score_enhanced:.3f}")
print(f"Improvement: {((hit_score_enhanced - hit_score_baseline) / hit_score_baseline * 100):.1f}%")
```

### 3. Chạy full evaluation:

```python
from metrics_rag import calculate_metrics_retrieval

# Baseline metrics
df_baseline = calculate_metrics_retrieval(
    "CLB_PROPTIT.csv", 
    "train_data_proptit.xlsx", 
    embedding, 
    vector_db, 
    train=True, 
    reranker=None,
    use_query_expansion=False  # Baseline
)

# Enhanced metrics với Query Expansion
df_enhanced = calculate_metrics_retrieval(
    "CLB_PROPTIT.csv", 
    "train_data_proptit.xlsx", 
    embedding, 
    vector_db, 
    train=True, 
    reranker=None,
    use_query_expansion=True   # 🚀 Enhanced
)

print("Baseline Metrics:")
print(df_baseline)
print("\nEnhanced Metrics:")
print(df_enhanced)
```

## 🧪 Demo và Testing

### Chạy demo đầy đủ:
```bash
python demo_query_expansion.py
```

### Test nhanh trong main.py:
```bash
python main.py
```

## 📊 Kết Quả Mong Đợi

Query Expansion techniques dự kiến sẽ cải thiện:

- **Hit@k**: Tăng khả năng tìm thấy documents liên quan
- **Recall@k**: Cải thiện độ bao phủ thông tin
- **MRR@k**: Đưa documents đúng lên vị trí cao hơn
- **Context Precision**: Tăng chất lượng ngữ cảnh được truy xuất

## ⚙️ Cấu Hình Tùy Chỉnh

### Điều chỉnh số lượng expansions:
```python
expanded_queries = expander.expand_query(
    query, 
    max_expansions=10,  # Tăng/giảm số lượng
    techniques=["rewriting", "context"]  # Chọn kỹ thuật cụ thể
)
```

### Tùy chỉnh trọng số:
Trong `retrieve_and_rerank()`, có thể điều chỉnh:
```python
weights = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3]  # Trọng số cho từng expansion
```

## 🔍 Domain-Specific Keywords

Hệ thống đã được tùy chỉnh cho domain CLB ProPTIT với các từ khóa:

- `clb`: ["câu lạc bộ", "club", "câu lạc bộ lập trình", "proptit"]
- `thanh_vien`: ["thành viên", "member", "sinh viên", "học viên"]
- `hoat_dong`: ["hoạt động", "activity", "sự kiện", "event", "workshop"]
- `lap_trinh`: ["lập trình", "programming", "code", "coding"]
- ...và nhiều hơn nữa

## 📁 Files Liên Quan

- `query_expansion.py` - Module chính chứa tất cả kỹ thuật
- `metrics_rag.py` - Đã được update để hỗ trợ query expansion
- `main.py` - Script test và so sánh
- `demo_query_expansion.py` - Demo đầy đủ các kỹ thuật

## 🎯 Tips Để Đạt Hiệu Suất Tốt Nhất

1. **Cân bằng số lượng expansions**: Không nên quá nhiều (gây nhiễu) hoặc quá ít (không cải thiện)
2. **Tùy chỉnh trọng số**: Ưu tiên query gốc và các expansions chất lượng cao
3. **Kết hợp với reranking**: Query expansion + reranking = hiệu quả tối ưu
4. **Monitor performance**: Theo dõi thời gian execution để tránh quá chậm

## 🚨 Lưu Ý

- Query expansion sẽ làm tăng thời gian xử lý do cần gọi LLM
- Cần có GROQ_API_KEY trong file .env
- Test trên dataset nhỏ trước khi chạy full evaluation
- Có thể điều chỉnh temperature trong QueryExpansion để tăng/giảm tính sáng tạo

## 🏆 Kỳ Vọng Cải Thiện

Với Query Expansion techniques được implement đúng cách, dự kiến:

- **Baseline → Enhanced**: Cải thiện 15-30% trên các metrics chính
- **Đặc biệt hiệu quả**: Với các câu hỏi ngắn, mơ hồ hoặc có nhiều cách diễn đạt
- **Robust**: Tăng khả năng xử lý các query variants trong test set

---

🎉 **Chúc các bạn đạt thành tích cao trong NeoRAG Cup 2025!** 🎉
