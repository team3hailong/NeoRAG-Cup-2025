# 🚀 NeoRAG Cup 2025 - Implementation Guide

## 📋 Tổng quan nhiệm vụ

**Mục tiêu**: Xây dựng hệ thống RAG (Retrieval-Augmented Generation) với domain CLB ProPTIT để đạt hiệu năng cao hơn baseline model.

**Thời gian**: 4 tuần (đến ngày pitching)

**Tiêu chí chấm điểm**:
- Kiến trúc pipeline: 30%
- Hiệu năng benchmark: 40% 
- Chất lượng demo: 20%
- Kỹ năng thuyết trình: 10%

---

## 🎯 Baseline Performance (cần vượt qua)

### Retrieval Metrics - Train:
| K | hit@k | recall@k | precision@k | f1@k | map@k | mrr@k | ndcg@k | context_precision@k | context_recall@k |
|---|-------|----------|-------------|------|-------|-------|--------|---------------------|------------------|
| 3 | 0.31  | 0.19     | 0.12        | 0.15 | 0.23  | 0.23  | 0.25   | 0.63                | 0.50             |
| 5 | 0.46  | 0.28     | 0.10        | 0.15 | 0.23  | 0.27  | 0.31   | 0.56                | 0.44             |
| 7 | 0.57  | 0.35     | 0.09        | 0.15 | 0.23  | 0.28  | 0.35   | 0.54                | 0.40             |

### LLM Answer Metrics - Train:
| K | string_presence@k | rouge_l@k | bleu_4@k | groundedness@k | response_relevancy@k |
|---|-------------------|-----------|----------|----------------|----------------------|
| 3 | 0.35              | 0.21      | 0.03     | 0.57           | 0.80                 |
| 5 | 0.40              | 0.23      | 0.03     | 0.61           | 0.80                 |
| 7 | 0.41              | 0.22      | 0.04     | 0.64           | 0.80                 |

---

## 🔧 BƯỚC 1: THIẾT LẬP MÔI TRƯỜNG

### 1.1 Tạo file .env
Tạo file `.env` trong thư mục gốc với nội dung:

```env
OPENAI_API_KEY=your_openai_api_key_here

# Vector Databases (chọn một hoặc nhiều)
MONGODB_URI=mongodb://localhost:27017/
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_KEY=your_qdrant_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key

GEMINI_API_KEY=your_gemini_api_key
```

### 1.2 Install dependencies
```bash
pip install -r requirements.txt
```

### 1.3 Kiểm tra các packages cần thiết
```bash
pip install python-dotenv openai pymongo chromadb qdrant-client supabase sentence-transformers
```

---

## 🛠️ BƯỚC 2: SỬA CODE TRONG CÁC FILE

### 2.1 File `main.py` - Cấu hình cơ bản

**Dòng 11**: Chọn vector database
```python
vector_db = VectorDatabase(db_type="mongodb")  # hoặc "chromadb", "qdrant", "supabase"
```

**Dòng 15**: Chọn embedding model
```python
embedding = Embeddings(model_name="text-embedding-3-large", type="openai")
```

**Alternatives để thử nghiệm**:
```python
# OpenAI embeddings (tốt nhưng tốn phí)
embedding = Embeddings(model_name="text-embedding-3-large", type="openai")
embedding = Embeddings(model_name="text-embedding-ada-002", type="openai")

# Sentence Transformers (miễn phí, chạy local)
embedding = Embeddings(model_name="all-mpnet-base-v2", type="sentence_transformers")
embedding = Embeddings(model_name="all-MiniLM-L6-v2", type="sentence_transformers")

# Google Gemini
embedding = Embeddings(model_name="text-embedding-004", type="gemini")
```

### 2.2 File `metrics_rag.py` - Các FIX_ME cần sửa

#### A. Trong các hàm retrieval metrics:

**Pattern chung cho user_embedding**:
```python
user_embedding = embedding.encode(query)
```

**Pattern chung cho results**:
```python
results = vector_db.query("information", user_embedding, limit=k)
```

**Pattern chung cho retrieved_docs**:
```python
retrieved_docs = []
for i, result in enumerate(results):
    # Giả sử title có format "Document X" where X is document ID
    doc_title = result.get('title', '')
    if doc_title.startswith('Document '):
        doc_id = int(doc_title.split(' ')[1])
        retrieved_docs.append(doc_id)
```

#### B. Trong các hàm LLM answer metrics:

**System prompt template**:
```python
system_content = """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống RAG chứa các thông tin chính xác về CLB.

Nguyên tắc trả lời:
1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời
2. Trình bày câu trả lời rõ ràng, dễ hiểu
3. Tuyệt đối không suy đoán hoặc bịa thông tin
4. Giữ phong cách trả lời thân thiện, chuyên nghiệp

Nhiệm vụ: Trả lời các câu hỏi về CLB Lập trình ProPTIT dựa trên context được cung cấp."""
```

**Context building**:
```python
context = "Content từ các tài liệu liên quan:\n"
context += "\n".join([result["information"] for result in results])
```

**User message**:
```python
user_content = context + "\n\nCâu hỏi: " + query
```

**OpenAI API call**:
```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # hoặc "gpt-4"
    messages=messages,
    temperature=0.3,
    max_tokens=500
)
reply = response.choices[0].message.content
```

#### C. Specific fixes cho một số hàm:

**ndcg_k function - similarity score**:
```python
# Lấy embedding của document từ CLB_PROPTIT.csv
doc_info = df_clb.iloc[doc-1]['Văn bản']  # doc-1 vì index bắt đầu từ 0
doc_embedding = embedding.encode(doc_info)
similarity_score = similarity(user_embedding, doc_embedding)
```

---

## 🚀 BƯỚC 3: CHIẾN LƯỢC TỐI ƯU HÓA

### 3.1 Cải thiện Retrieval (40% điểm số)

#### A. Embedding Model Optimization
```python
# Test multiple models và so sánh hiệu năng
models_to_test = [
    ("text-embedding-3-large", "openai"),
    ("text-embedding-ada-002", "openai"),
    ("all-mpnet-base-v2", "sentence_transformers"),
    ("all-MiniLM-L6-v2", "sentence_transformers"),
]
```

#### B. Document Chunking Strategy
```python
# Thay vì lưu từng paragraph, chia documents thành chunks tối ưu
def chunk_document(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
```

#### C. Query Enhancement
```python
def enhance_query(original_query, embedding_model):
    """Generate multiple variations of the query"""
    
    # Query expansion
    expanded_query = f"{original_query} CLB ProPTIT lập trình"
    
    # Query rewriting using LLM
    rewrite_prompt = f"""
    Hãy viết lại câu hỏi sau để tìm kiếm thông tin về CLB ProPTIT hiệu quả hơn:
    Câu hỏi gốc: {original_query}
    Câu hỏi mới:
    """
    
    # Return multiple query variations
    return [original_query, expanded_query, rewritten_query]
```

#### D. Hybrid Search Implementation
```python
def hybrid_search(query, embedding, vector_db, k=5):
    """Combine dense and sparse retrieval"""
    
    # Dense retrieval (semantic)
    query_embedding = embedding.encode(query)
    dense_results = vector_db.query("information", query_embedding, limit=k*2)
    
    # Sparse retrieval (keyword matching)
    sparse_results = keyword_search(query, k*2)
    
    # Combine and rerank results
    combined_results = combine_and_rerank(dense_results, sparse_results, k)
    
    return combined_results
```

### 3.2 Cải thiện Generation

#### A. Advanced Prompt Engineering
```python
def create_advanced_system_prompt():
    return """Bạn là chuyên gia về Câu lạc bộ Lập trình ProPTIT với kiến thức sâu rộng.

NHIỆM VỤ:
- Trả lời câu hỏi dựa CHÍNH XÁC trên context được cung cấp
- Sử dụng thông tin từ nhiều nguồn trong context nếu có
- Trình bày theo cấu trúc logic, rõ ràng

QUY TẮC:
1. Nếu context có thông tin đầy đủ → Trả lời chi tiết
2. Nếu context có thông tin một phần → Trả lời phần có và nói rõ giới hạn
3. Nếu context không có thông tin → "Dựa trên thông tin hiện có, tôi không thể..."

ĐỊNH DẠNG TRẢ LỜI:
- Sử dụng bullet points cho danh sách
- Highlight thông tin quan trọng
- Cung cấp context khi cần thiết"""
```

#### B. Few-shot Learning
```python
def add_few_shot_examples():
    examples = """
EXAMPLE 1:
Context: "CLB được thành lập ngày 9/10/2011..."
Question: CLB được thành lập khi nào?
Answer: Câu lạc bộ Lập trình ProPTIT được thành lập ngày 9/10/2011.

EXAMPLE 2:
Context: "Team AI, Team Mobile, Team Data..."
Question: CLB có những team nào?
Answer: CLB ProPTIT có các team dự án sau: Team AI, Team Mobile, Team Data, Team Game, Team Web, Team Backend.
"""
    return examples
```

### 3.3 Reranking Implementation
```python
def rerank_results(query, initial_results, top_k=5):
    """Rerank retrieved results using cross-encoder"""
    
    scores = []
    for result in initial_results:
        # Calculate relevance score
        score = calculate_relevance(query, result['information'])
        scores.append((result, score))
    
    # Sort by score and return top k
    sorted_results = sorted(scores, key=lambda x: x[1], reverse=True)
    return [result[0] for result in sorted_results[:top_k]]
```

---

## 📊 BƯỚC 4: MONITORING VÀ EVALUATION

### 4.1 Tạo script đánh giá tự động
```python
# create file: evaluate_model.py
def evaluate_all_configurations():
    """Test different configurations and log results"""
    
    configurations = [
        {"db": "mongodb", "embedding": ("text-embedding-3-large", "openai")},
        {"db": "chromadb", "embedding": ("all-mpnet-base-v2", "sentence_transformers")},
        # Add more configurations
    ]
    
    results = {}
    for config in configurations:
        print(f"Testing config: {config}")
        metrics = run_evaluation(config)
        results[str(config)] = metrics
        
    # Save results for comparison
    save_results(results)
```

### 4.2 Tracking improvements
```python
# Tạo file log để theo dõi cải thiện
def log_improvement(baseline_metrics, current_metrics):
    improvements = {}
    for metric in baseline_metrics:
        if metric in current_metrics:
            improvement = current_metrics[metric] - baseline_metrics[metric]
            improvements[metric] = {
                'baseline': baseline_metrics[metric],
                'current': current_metrics[metric],
                'improvement': improvement,
                'percentage': (improvement / baseline_metrics[metric]) * 100
            }
    return improvements
```

---

## 📈 BƯỚC 5: ADVANCED TECHNIQUES

### 5.1 Multi-stage Retrieval
```python
def multi_stage_retrieval(query, embedding, vector_db):
    # Stage 1: Broad retrieval
    initial_results = vector_db.query("information", embedding.encode(query), limit=20)
    
    # Stage 2: Query expansion based on initial results
    expanded_queries = generate_related_queries(query, initial_results[:3])
    
    # Stage 3: Retrieve with expanded queries
    final_results = []
    for expanded_query in expanded_queries:
        results = vector_db.query("information", embedding.encode(expanded_query), limit=5)
        final_results.extend(results)
    
    # Stage 4: Deduplicate and rerank
    return deduplicate_and_rerank(final_results, query, top_k=5)
```

### 5.2 Context Optimization
```python
def optimize_context(retrieved_docs, query, max_tokens=2000):
    """Select most relevant parts of retrieved documents"""
    
    # Score each sentence in each document
    scored_sentences = []
    for doc in retrieved_docs:
        sentences = doc['information'].split('. ')
        for sentence in sentences:
            score = calculate_sentence_relevance(sentence, query)
            scored_sentences.append((sentence, score, doc['title']))
    
    # Select top sentences within token limit
    sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
    
    context = ""
    token_count = 0
    for sentence, score, doc_title in sorted_sentences:
        sentence_tokens = len(sentence.split())
        if token_count + sentence_tokens <= max_tokens:
            context += f"{sentence}. "
            token_count += sentence_tokens
        else:
            break
    
    return context
```

---

## 🎯 BƯỚC 6: PLAN THỰC HIỆN 4 TUẦN

### Tuần 1 (Hiện tại): Foundation Setup
- [ ] Setup môi trường, tạo .env file
- [ ] Sửa tất cả FIX_ME để code chạy được
- [ ] Chạy baseline model và xác nhận kết quả
- [ ] Test các embedding models khác nhau

### Tuần 2: Core Optimization
- [ ] Implement chunking strategies
- [ ] Test different vector databases
- [ ] Implement query enhancement
- [ ] Optimize system prompts

### Tuần 3: Advanced Techniques
- [ ] Implement hybrid search
- [ ] Add reranking mechanism
- [ ] Implement multi-stage retrieval
- [ ] Fine-tune hyperparameters

### Tuần 4: Demo & Presentation
- [ ] Create Streamlit demo app
- [ ] Prepare presentation slides
- [ ] Practice demo và Q&A
- [ ] Final testing và bug fixes

---

## 🚨 TIPS QUAN TRỌNG

### Để đạt hiệu suất cao:

1. **Focus on metrics quan trọng nhất**:
   - `hit@k` và `recall@k` cho retrieval
   - `groundedness@k` để giảm hallucination
   - `context_precision@k` để tăng chất lượng context

2. **Thử nghiệm systematic**:
   - Test từng component riêng biệt
   - A/B test different approaches
   - Keep track of what works

3. **Optimization order**:
   - Embedding model (biggest impact)
   - Chunking strategy
   - System prompt
   - Reranking
   - Advanced techniques

4. **Common pitfalls to avoid**:
   - Over-engineering without testing
   - Ignoring computational cost
   - Not validating on test data
   - Poor error handling

---

## 📝 CHECKLIST TRƯỚC KHI NỘP

- [ ] Code chạy không lỗi trên toàn bộ dataset
- [ ] Metrics cao hơn baseline trên ít nhất 80% các chỉ số
- [ ] Demo app hoạt động mượt mà
- [ ] Slides thuyết trình hoàn chỉnh (kiến trúc + kết quả + demo)
- [ ] Đã practice thuyết trình trong 30 phút
- [ ] Code được comment và organize tốt

---

**Chúc bạn thành công với NeoRAG Cup 2025! 🏆**

*Lưu ý: File này sẽ được cập nhật khi có thêm insights từ quá trình thử nghiệm.*
