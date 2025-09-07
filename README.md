# NeoRAG: Advanced Retrieval-Augmented Generation System

## 🏗️ Kiến trúc hệ thống: Enhanced RAG with Multi-Stage Optimization

Hệ thống NeoRAG được thiết kế với kiến trúc RAG tiên tiến, tích hợp nhiều kỹ thuật tối ưu hóa để nâng cao hiệu suất retrieval và generation.

### 📊 Sơ đồ kiến trúc

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NeoRAG Architecture                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌─────────────────────────────────────────┐   │
│  │  User Query  │    │          Query Expansion                │   │
│  └──────┬───────┘    │  • Query Rewriting                     │   │
│         │            │  • Query Decomposition                 │   │
│         ▼            │  • Synonym/Paraphrase Expansion        │   │
│  ┌──────────────┐    │  • Context-Aware Expansion             │   │
│  │Query Analysis│◄───┤  • Multi-Perspective Queries           │   │
│  └──────┬───────┘    │  • Document Structure Awareness        │   │
│         │            └─────────────────────────────────────────┘   │
│         ▼                                                          │
│  ┌──────────────┐    ┌─────────────────────────────────────────┐   │
│  │  Embedding   │    │         Embedding Model                 │   │
│  │  Generation  │◄───┤   BAAI/bge-m3 (Multilingual)           │   │
│  └──────┬───────┘    │   Alternative: Alibaba-NLP/gte-base    │   │
│         │            └─────────────────────────────────────────┘   │
│         ▼                                                          │
│  ┌──────────────┐    ┌─────────────────────────────────────────┐   │
│  │Vector Search │◄───┤         Vector Database                 │   │
│  │              │    │  • ChromaDB (Primary)                  │   │
│  └──────┬───────┘    │  • MongoDB Atlas (Alternative)         │   │
│         │            │  • Qdrant (Alternative)                │   │
│         ▼            └─────────────────────────────────────────┘   │
│  ┌──────────────┐    ┌─────────────────────────────────────────┐   │
│  │  Re-ranking  │◄───┤         Reranking Model                │   │
│  │              │    │   namdp-ptit/ViRanker (Vietnamese)     │   │
│  └──────┬───────┘    │   Alternative: BAAI/bge-reranker-v2-m3 │   │
│         │            └─────────────────────────────────────────┘   │
│         ▼                                                          │
│  ┌──────────────┐    ┌─────────────────────────────────────────┐   │
│  │Context Fusion│    │      Response Generation               │   │
│  │& Generation  │◄───┤  • Gemini Models                   │   │
│  │              │    │  • Grok Models                       │   │
│  └──────┬───────┘    │  • Local Ollama Models                 │   │
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

1. **Query Expansion Module**: Mở rộng câu truy vấn với 6 kỹ thuật khác nhau để tăng recall
2. **Multilingual Embedding**: Hỗ trợ nhiều model embedding đa ngôn ngữ
4. **Advanced Reranking**: Sử dụng model reranking được fine-tune cho tiếng Việt
5. **Multi-LLM Support**: Hỗ trợ nhiều LLM khác nhau cho generation

### 🚀 Kỹ thuật Query Expansion:

- **Query Rewriting**: Viết lại câu hỏi với nhiều cách diễn đạt
- **Query Decomposition**: Phân tách câu hỏi phức tạp thành câu hỏi con
- **Synonym Expansion**: Mở rộng với từ đồng nghĩa và cách diễn đạt khác
- **Context-Aware Expansion**: Mở rộng dựa trên ngữ cảnh CLB ProPTIT
- **Multi-Perspective Query**: Tạo các góc nhìn khác nhau
- **Document Structure Awareness**: Hiểu cấu trúc tài liệu để tối ưu truy vấn

## 📈 Benchmark Results

### 🏋️ Retrieval Metrics - Train Data (100 queries)

| Metric | k=3 | k=5 | k=7 | Baseline k=3 | Baseline k=5 | Baseline k=7 |
|--------|-----|-----|-----|--------------|--------------|--------------|
| **Hit@k** | 0.59 | 0.72 | 0.76 | 0.31 | 0.46 | 0.57 |
| **Recall@k** | 0.41 | 0.49 | 0.54 | 0.19 | 0.28 | 0.35 |
| **Precision@k** | 0.21 | 0.16 | 0.13 | 0.12 | 0.10 | 0.09 |
| **F1@k** | 0.28 | 0.25 | 0.21 | 0.15 | 0.15 | 0.15 |
| **MAP@k** | 0.52 | 0.55 | 0.54 | 0.23 | 0.23 | 0.23 |
| **MRR@k** | 0.52 | 0.55 | 0.56 | 0.23 | 0.27 | 0.28 |
| **NDCG@k** | 0.54 | 0.59 | 0.60 | 0.25 | 0.31 | 0.35 |
| **Context Precision@k** | 0.78 | 0.66 | 0.57 | 0.63 | 0.56 | 0.54 |
| **Context Recall@k** | 0.54 | 0.45 | 0.42 | 0.50 | 0.44 | 0.40 |
| **Context Entities Recall@k** | 0.47 | 0.45 | 0.47 | 0.32 | 0.37 | 0.38 |

### 🤖 LLM Answer Metrics - Train Data

| Metric | k=3 | k=5 | k=7 | Baseline k=3 | Baseline k=5 | Baseline k=7 |
|--------|-----|-----|-----|--------------|--------------|--------------|
| **String Presence@k** | 0.47 | 0.50 | 0.48 | 0.35 | 0.40 | 0.41 |
| **ROUGE-L@k** | 0.22 | 0.21 | 0.21 | 0.21 | 0.23 | 0.22 |
| **BLEU-4@k** | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 |
| **Groundedness@k** | 0.57 | 0.61 | 0.64 | 0.57 | 0.61 | 0.64 |
| **Response Relevancy@k** | 0.85 | 0.85 | 0.85 | 0.80 | 0.80 | 0.80 |
| **Noise Sensitivity@k** | 0.51 | 0.53 | 0.51 | 0.51 | 0.53 | 0.51 |

### 🎯 Retrieval Metrics - Test Data (30 queries)

| Metric | k=3 | k=5 | k=7 | Baseline k=3 | Baseline k=5 | Baseline k=7 |
|--------|-----|-----|-----|--------------|--------------|--------------|
| **Hit@k** | 0.93 | 0.93 | 0.97 | 0.23 | 0.40 | 0.47 |
| **Recall@k** | 0.73 | 0.76 | 0.82 | 0.06 | 0.10 | 0.13 |
| **Precision@k** | 0.47 | 0.30 | 0.24 | 0.08 | 0.08 | 0.08 |
| **F1@k** | 0.57 | 0.43 | 0.37 | 0.07 | 0.09 | 0.10 |
| **MAP@k** | 0.86 | 0.84 | 0.85 | 0.12 | 0.16 | 0.17 |
| **MRR@k** | 0.87 | 0.87 | 0.89 | 0.12 | 0.16 | 0.17 |
| **NDCG@k** | 0.88 | 0.87 | 0.89 | 0.15 | 0.22 | 0.24 |
| **Context Precision@k** | 0.88 | 0.74 | 0.57 | 0.34 | 0.35 | 0.31 |
| **Context Recall@k** | 0.66 | 0.53 | 0.45 | 0.32 | 0.29 | 0.27 |
| **Context Entities Recall@k** | 0.61 | 0.62 | 0.67 | 0.11 | 0.15 | 0.16 |

### 🤖 LLM Answer Metrics - Test Data

| Metric | k=3 | k=5 | k=7 | Baseline k=3 | Baseline k=5 | Baseline k=7 |
|--------|-----|-----|-----|--------------|--------------|--------------|
| **String Presence@k** | 0.53 | 0.58 | 0.57 | 0.18 | 0.16 | 0.21 |
| **ROUGE-L@k** | 0.42 | 0.43 | 0.40 | 0.14 | 0.15 | 0.15 |
| **BLEU-4@k** | 0.16 | 0.18 | 0.20 | 0.01 | 0.01 | 0.02 |
| **Groundedness@k** | 0.91 | 0.87 | 0.90 | 0.33 | 0.30 | 0.39 |
| **Response Relevancy@k** | 0.86 | 0.86 | 0.86 | 0.79 | 0.79 | 0.80 |
| **Noise Sensitivity@k** | 0.00 | 0.00 | 0.00 | 0.68 | 0.71 | 0.71 |

## ✨ Điểm nổi bật (Điểm mạnh)

### 🔥 Kỹ thuật tiên tiến:
- **Query Expansion đa chiều**: 6 kỹ thuật mở rộng câu truy vấn khác nhau
- **Domain-specific optimization**: Tối ưu hóa riêng cho domain ProPTIT với từ khóa chuyên biệt
- **Weighted query ranking**: Xếp hạng các query mở rộng theo độ tương đồng
- **Multi-stage retrieval**: Kết hợp initial retrieval + reranking

### 🌟 Hiệu suất cao:
*So sánh với baseline*

**Train Data Performance:**
- **Hit@k cải thiện đáng kể**: k=3: +90% (0.31→0.59), k=5: +24% (0.46→0.57), k=7: +33% (0.57→0.76)
- **Recall@k tăng mạnh**: k=3: +116% (0.19→0.41), k=5: +75% (0.28→0.49), k=7: +54% (0.35→0.54)
- **MAP@k vượt trội**: Tăng hơn 2x so với baseline ở tất cả các k
- **NDCG@k cải thiện**: k=3: +116% (0.25→0.54), k=5: +90% (0.31→0.59), k=7: +71% (0.35→0.60)
- **String Presence@k**: k=3: +34% (0.35→0.47), k=5: +25% (0.40→0.50)
- **Response Relevancy tăng**: Từ 0.80 lên 0.85 (+6.25%)

**Test Data Performance:**
- **Hit@k xuất sắc**: k=3: +304% (0.23→0.93), k=5: +133% (0.40→0.93), k=7: +106% (0.47→0.97)
- **Recall@k vượt trội**: k=3: +1117% (0.06→0.73), k=5: +660% (0.10→0.76), k=7: +531% (0.13→0.82)
- **Precision@k tăng mạnh**: k=3: +488% (0.08→0.47), k=5: +275% (0.08→0.30), k=7: +200% (0.08→0.24)
- **BLEU-4@k cải thiện**: k=3: +1500% (0.01→0.16), k=5: +1700% (0.01→0.18), k=7: +900% (0.02→0.20)
- **Groundedness@k vượt trội**: k=3: +176% (0.33→0.91), k=5: +190% (0.30→0.87), k=7: +131% (0.39→0.90)
- **ROUGE-L@k tăng**: k=3: +200% (0.14→0.42), k=5: +187% (0.15→0.43), k=7: +167% (0.15→0.40)

### 🔧 Tính linh hoạt:
- **Multi-database support**: ChromaDB, MongoDB, Qdrant
- **Multi-LLM compatibility**: Grok
- **Configurable components**: Dễ dàng thay đổi embedding model, reranker
- **Comprehensive metrics**: 15+ metrics đánh giá toàn diện
- **No-cost system**: Tất cả embedding model và llm model đều được sử dụng miễn phí số lượng request và token lớn

### 🇻🇳 Tối ưu tiếng Việt:
- **Vietnamese reranker**: namdp-ptit/ViRanker được fine-tune cho tiếng Việt
- **Context-aware expansion**: Hiểu ngữ cảnh văn hóa và thuật ngữ Việt Nam
- **Domain keywords**: Tập từ khóa chuyên biệt cho lĩnh vực giáo dục Việt Nam

## ⚠️ Hạn chế

### 🐌 Hiệu suất:
- **Độ trễ cao**: Query expansion tăng thời gian xử lý lên 3-5 lần
- **Resource intensive**: Cần GPU để chạy embedding và reranking models
- **Free LLM**: khó cạnh tranh được với các LLM thương mại

### 🎯 Độ chính xác:
- **Query expansion noise**: Một số câu hỏi mở rộng có thể không liên quan
- **Context window limitation**: Giới hạn độ dài context với LLM
- **Domain dependency**: Hiệu suất giảm khi áp dụng cho domain khác

### 🔧 Kỹ thuật:
- **Model dependency**: Phụ thuộc vào chất lượng của external models
- **Complexity**: Kiến trúc phức tạp, khó maintain và debug
- **Memory usage**: Yêu cầu RAM cao để load multiple models

### 📊 Evaluation:
- **Subjective evaluation**: Thiếu đánh giá human evaluation

## 🛠️ Technical Stack

### 📦 Core Dependencies:
- **Embedding**: `sentence-transformers`, `BAAI/bge-m3`
- **Vector DB**: `chromadb`, `qdrant-client`, `pymongo`
- **Reranking**: `FlagEmbedding`, `namdp-ptit/ViRanker`
- **LLM Integration**: `openai`, `google-genai`, `groq`
- **Query Expansion**: Custom implementation with LLM-based techniques

