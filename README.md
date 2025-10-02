# NeoRAG: Advanced Retrieval-Augmented Generation System
Hệ thống NeoRAG được thiết kế với kiến trúc RAG tiên tiến, tích hợp nhiều kỹ thuật tối ưu hóa để nâng cao hiệu suất retrieval và generation.

## 📊 Sơ đồ kiến trúc

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NeoRAG Architecture                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌─────────────────────────────────────────┐   │
│  │  User Query  │    │          Query Expansion                │   │
│  └──────┬───────┘    │  • Synonym Expansion                    │   │
│         │            │  • Context-Aware Expansion              │   │
│         ▼            │  • LLM-Based Expansion                  │   │
│  ┌──────────────┐    │  • Weighted Query Ranking               │   │
│  │Expanded      │◄───┤  • Domain-Specific Keywords             │   │
│  │Queries       │    └─────────────────────────────────────────┘   │
│  └──────┬───────┘                                                  │
│         │                                                          │
│    ┌────▼────┐    ┌─────────────────────────────────────────┐     │
│    │  For    │    │         Embedding Model                 │     │
│    │ Each    │◄───┤   BAAI/bge-m3 (Multilingual + ColBERT) │     │
│    │ Query   │    │   Dense + Sparse + ColBERT Vectors     │     │
│    └────┬───┘    └─────────────────────────────────────────┘     │
│         │                                                          │
│    ┌────▼────┐    ┌─────────────────────────────────────────┐     │
│    │  For    │    │         Vector Database                 │     │
│    │ Each    │◄───┤  • ChromaDB (Primary)                  │     │
│    │Embedding│    │  • MongoDB Atlas (Alternative)         │     │
│    └────┬───┘    │  • Qdrant (Alternative)                │     │
│         │        └─────────────────────────────────────────┘     │
│    ┌────▼────┐                                                  │
│    │  Merge  │                                                  │
│    │Results  │                                                  │
│    └────┬───┘                                                  │
│         │                                                          │
│  ┌──────▼──────┐    ┌─────────────────────────────────────────┐   │
│  │  Re-ranking  │◄───┤         Reranking Model                │   │
│  │              │    │   namdp-ptit/ViRanker (Vietnamese)     │   │
│  └──────┬───────┘    │   Optimized for Vietnamese Context     │   │
│         │            └─────────────────────────────────────────┘   │
│         ▼                                                          │
│  ┌──────────────┐    ┌─────────────────────────────────────────┐   │
│  │Context Fusion│    │      Response Generation               │   │
│  │& Generation  │◄───┤  • NVIDIA Models (Primary)             │   │
│  │              │    │  • GROQ Models (Alternative)           │   │
│  └──────┬───────┘    │  • Optimized Prompts for ProPTIT       │   │
│         │            └─────────────────────────────────────────┘   │
│         ▼                                                          │
│  ┌──────────────┐                                                  │
│  │Final Answer  │                                                  │
│  └──────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## 🎯 Giới thiệu về kiến trúc

**NeoRAG** là hệ thống RAG (Retrieval-Augmented Generation) được tối ưu hóa đặc biệt cho domain câu lạc bộ lập trình ProPTIT. Hệ thống kết hợp nhiều kỹ thuật tiên tiến:

### 🔧 Các thành phần chính:

1. **Query Expansion Module**: Mở rộng câu truy vấn với 3 kỹ thuật chính
2. **Multilingual ColBERT Embedding**: BAAI/bge-m3 với hỗ trợ dense, sparse và ColBERT vectors
3. **Multi-Database Support**: ChromaDB, MongoDB, Qdrant
4. **Vietnamese-Optimized Reranking**: namdp-ptit/ViRanker được fine-tune cho tiếng Việt
5. **Multi-LLM Generation**: NVIDIA và GROQ với prompts tối ưu cho domain ProPTIT️

### 🚀 Kỹ thuật Query Expansion:

- **Synonym Expansion**: Mở rộng với từ đồng nghĩa và cách diễn đạt khác từ domain-specific keywords
- **Context-Aware Expansion**: Mở rộng dựa trên ngữ cảnh CLB ProPTIT với template-based patterns
- **LLM-Based Expansion**: Sử dụng LLM để tạo các cách hỏi khác nhau với prompt tối ưu

## 📈 Benchmark Results

### 🏋️ Retrieval Metrics - Train Data (100 queries)

| Metric | k=3 | k=5 | k=7 |
|--------|-----|-----|-----|
| **Hit@k** | 0.94 | 0.97 | 0.99 |
| **Recall@k** | 0.79 | 0.91 | 0.95 |
| **Precision@k** | 0.51 | 0.36 | 0.27 |
| **F1@k** | 0.62 | 0.52 | 0.43 |
| **MAP@k** | 0.74 | 0.69 | 0.66 |
| **MRR@k** | 0.75 | 0.72 | 0.71 |
| **NDCG@k** | 0.79 | 0.77 | 0.76 |
| **Context Precision@k** | 0.84 | 0.62 | 0.67 |
| **Context Recall@k** | 0.63 | 0.53 | 0.49 |
| **Context Entities Recall@k** | 0.77 | 0.83 | 0.84 |

### 🤖 LLM Answer Metrics - Train Data

| Metric | k=3 | k=5 | k=7 |
|--------|-----|-----|-----|
| **String Presence@k** | 0.75 | 0.73 | 0.73 |
| **ROUGE-L@k** | 0.24 | 0.24 | 0.25 |
| **BLEU-4@k** | 0.05 | 0.05 | 0.05 |
| **Groundedness@k** | 0.93 | 0.96 | 0.96 |
| **Response Relevancy@k** | 0.82 | 0.82 | 0.82 |
| **Noise Sensitivity@k** | 0.18 | 0.19 | 0.17 |

### 🎯 Retrieval Metrics - Test Data (30 queries)

| Metric | k=3 | k=5 | k=7 |
|--------|-----|-----|-----|
| **Hit@k** | 0.97 | 0.97 | 0.97 |
| **Recall@k** | 0.82 | 0.89 | 0.90 |
| **Precision@k** | 0.53 | 0.36 | 0.27 |
| **F1@k** | 0.65 | 0.51 | 0.41 |
| **MAP@k** | 0.87 | 0.82 | 0.79 |
| **MRR@k** | 0.88 | 0.86 | 0.86 |
| **NDCG@k** | 0.90 | 0.87 | 0.86 |
| **Context Precision@k** | 0.94 | 0.76 | 0.75 |
| **Context Recall@k** | 0.88 | 0.73 | 0.72 |
| **Context Entities Recall@k** | 0.94 | 0.96 | 0.96 |

### 🤖 LLM Answer Metrics - Test Data

| Metric | k=3 | k=5 | k=7 |
|--------|-----|-----|-----|
| **String Presence@k** | 0.84 | 0.86 | 0.87 |
| **ROUGE-L@k** | 0.54 | 0.54 | 0.58 |
| **BLEU-4@k** | 0.31 | 0.33 | 0.37 |
| **Groundedness@k** | 0.99 | 1.00 | 1.00 |
| **Response Relevancy@k** | 0.82 | 0.82 | 0.81 |
| **Noise Sensitivity@k** | 0.02 | 0.02 | 0.00 |

## ✨ Điểm nổi bật (Điểm mạnh)

### 🔥 Kỹ thuật tiên tiến:
- **ColBERT Multi-Vector Retrieval**: Sử dụng BAAI/bge-m3 với dense, sparse và ColBERT vectors
- **Domain-specific Query Expansion**: 3 kỹ thuật mở rộng query với từ khóa chuyên biệt cho CLB ProPTIT
- **Vietnamese-Optimized Reranking**: namdp-ptit/ViRanker được fine-tune đặc biệt cho tiếng Việt
- **Hybrid Retrieval Pipeline**: Kết hợp retrieval ban đầu + reranking với scoring tối ưu

### 🌟 Hiệu suất vượt trội:
*NeoRAG đạt hiệu suất xuất sắc trên cả tập train và test với các kỹ thuật tối ưu hóa tiên tiến*

### 🔧 Tính linh hoạt:
- **Multi-Database Support**: ChromaDB, MongoDB Atlas, Qdrant với khả năng mở rộng
- **Multi-LLM Integration**: NVIDIA và GROQ với cấu hình linh hoạt
- **ColBERT Vector Support**: Hỗ trợ dense, sparse và ColBERT embeddings
- **Configurable Components**: Dễ dàng thay đổi embedding model, reranker, LLM
- **Comprehensive Metrics**: 15+ metrics đánh giá toàn diện hiệu suất hệ thống

### 🇻🇳 Tối ưu tiếng Việt:
- **Vietnamese Reranker**: namdp-ptit/ViRanker được fine-tune đặc biệt cho tiếng Việt
- **Domain-Specific Keywords**: Bộ từ khóa chuyên biệt cho lĩnh vực giáo dục và CLB ProPTIT
- **Context-Aware Expansion**: Hiểu ngữ cảnh văn hóa và thuật ngữ Việt Nam
- **Optimized Prompts**: Prompts được thiết kế phù hợp với phong cách giao tiếp tiếng Việt

## ⚠️ Hạn chế

### 🐌 Hiệu suất:
- **Độ trễ cao với Query Expansion**: Việc mở rộng query làm tăng thời gian xử lý 2-3 lần
- **Resource intensive ColBERT**: Cần GPU để chạy BAAI/bge-m3 với ColBERT vectors
- **Memory usage**: Yêu cầu RAM cao để load multiple models và ColBERT embeddings
- **API Rate Limits**: Giới hạn rate của NVIDIA/GROQ API có thể ảnh hưởng hiệu suất

### 🎯 Độ chính xác:
- **ColBERT Complexity**: Việc xử lý multi-vector có thể dẫn đến false positives
- **Domain Dependency**: Hiệu suất giảm khi áp dụng cho domain khác ngoài ProPTIT

### 🔧 Kỹ thuật:
- **Model Dependency**: Phụ thuộc vào chất lượng của external models (BAAI, ViRanker)
- **ColBERT Storage**: Yêu cầu lưu trữ phức tạp cho multi-vector embeddings
- **Scalability**: Khó scale với lượng dữ liệu lớn do ColBERT overhead
- **Debugging Complexity**: Kiến trúc phức tạp làm khó debug và maintain

### 📊 Evaluation:
- **Subjective evaluation**: Thiếu đánh giá human evaluation

## 🛠️ Technical Stack

### 📦 Core Dependencies:
- **Embedding**: `sentence-transformers`, `BAAI/bge-m3`, `FlagEmbedding` (ColBERT support)
- **Vector DB**: `chromadb`, `qdrant-client`, `pymongo` (Multi-database support)
- **Reranking**: `FlagEmbedding`, `namdp-ptit/ViRanker` (Vietnamese-optimized)
- **LLM Integration**: `requests` (NVIDIA API), `groq` (GROQ API)
- **Query Expansion**: Custom implementation với domain-specific keywords
- **Metrics**: Comprehensive evaluation với 15+ metrics
- **Document Processing**: `python-docx`, `pandas`, `numpy`