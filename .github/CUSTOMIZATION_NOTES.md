# 🔧 CUSTOMIZATION GUIDE - NeoRAG Cup 2025

## 📋 Tổng quan các điểm có thể tùy chỉnh

### 🤖 **1. LLM Models & APIs**

#### 1.1 Thay đổi LLM Provider (file: `metrics_rag.py`)

### 📊 **2. Embedding Models**

#### 2.1 Thay đổi Embedding Model (file: `main.py`, line ~18)
```python
# 🔧 HIỆN TẠI: Ollama
embedding = Embeddings(model_name="nomic-embed-text", type="ollama")
```

---

### 🗃️ **3. Vector Databases**

#### 3.1 Thay đổi Vector DB (file: `main.py`, line ~14)
```python
# 🔧 HIỆN TẠI: MongoDB
vector_db = VectorDatabase(db_type="mongodb")
```

---

### 📝 **4. System Prompts & Instructions**

#### 4.1 Prompt (Xuất hiện nhiều lần trong `metrics_rag.py`)
🔧 **Vị trí**: Tìm `"Sửa prompt nếu muốn"`

---

### ⚙️ **5. Retrieval & Ranking**

#### 5.1 Retrieval Parameters (Tất cả metrics functions)
```python
# 🔧 CÓ THỂ THAY ĐỔI:
results = vector_db.query("information", user_embedding, limit=k)

# Options:
- limit=k: Số documents retrieve
- similarity thresholds
- Reranking algorithms
- Query expansion techniques
```

#### 5.2 Document Processing
🔧 **Vị trí**: `main.py`, document chunking logic

**Có thể optimize:**
- Chunking strategy (sentence vs paragraph vs semantic)
- Overlap between chunks
- Metadata enrichment

---

### 🎯 **7. Performance Optimization**

#### 7.1 Caching Strategy
🔧 **Có thể thêm:**
- Cache embeddings
- Cache LLM responses  
- Batch processing
- Async operations

#### 7.2 Resource Management
🔧 **Có thể optimize:**
- API rate limiting
- Memory usage for large datasets
- Parallel processing

---

### 🔄 **8. Advanced Techniques**

#### 8.1 Query Enhancement
🔧 **Có thể implement:**
```python
# Trong các retrieval functions, trước khi embed query:
# query = query_expansion(query)
# query = query_rewrite(query, context)
# user_embedding = ensemble_embedding([emb1, emb2, emb3])
```

#### 8.2 Post-Processing
🔧 **Có thể thêm:**
- Response filtering
- Fact verification
- Output formatting
- Confidence scoring

---

## 🏆 **Recommendations để vượt Baseline**

### **Immediate wins:**
1. **Embedding Model**: Thử `text-embedding-3-large` hoặc domain-specific models
2. **LLM Model**: Upgrade lên `gemini-1.5-pro` hoặc `gpt-4`
3. **Retrieval**: Tăng k và implement reranking
4. **Prompts**: Fine-tune system prompts cho domain ProPTIT

### **Advanced optimizations:**
1. **Hybrid Search**: Combine vector + keyword search
2. **Query Expansion**: Sử dụng LLM để mở rộng query
3. **Ensemble Methods**: Kết hợp multiple models
4. **Fine-tuning**: Train custom embedding/reranking models

### **Evaluation improvements:**
1. **Custom Metrics**: Thêm domain-specific evaluation
2. **Human Evaluation**: Supplement với manual scoring
3. **A/B Testing**: Compare different configurations

---

## 📝 **Quick Start Customization**

1. **Copy `.env.example` to `.env`** và điền API keys
2. **Thay đổi MODEL_NAME** trong `metrics_rag.py` (line ~16)
3. **Thay đổi embedding model** trong `main.py` (line ~18)
4. **Chạy baseline** để có reference performance
5. **Iterate từng component** một cách systematic

---

## 🐛 **Troubleshooting**

- **API Errors**: Kiểm tra `.env` file và API quotas
- **Memory Issues**: Giảm batch size hoặc k values
- **Slow Performance**: Consider caching và parallel processing
- **Low Scores**: Debug từng metric riêng biệt trước khi optimize tổng thể
