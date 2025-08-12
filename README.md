# 🚀 NeoRAG Cup 2025

## 1. Giới thiệu
NeoRAG Cup 2025 là cuộc thi học thuật – kỹ thuật do **Team AI – CLB Lập trình ProPTIT** tổ chức, dành cho các bạn đam mê **Trí tuệ nhân tạo (AI)**, **Xử lý ngôn ngữ tự nhiên (NLP)** và **Kỹ thuật hệ thống**.  

Người tham gia sẽ:
- Tự thiết kế, hiện thực hóa và trình bày một **pipeline RAG** (Retrieval-Augmented Generation) với domain là thông tin của CLB ProPTIT.
- Trải nghiệm toàn bộ quy trình phát triển sản phẩm AI từ **ý tưởng → triển khai → demo**.

---

## 2. Thể lệ & Yêu cầu
**Domain:** Thông tin liên quan đến CLB ProPTIT (lịch sử, thành viên, hoạt động, dự án, tài liệu nội bộ, v.v.)

**Nhiệm vụ:**
1. Thiết kế pipeline RAG hoàn chỉnh (kiến trúc, công nghệ, chiến lược index, retrieval, reranking, generation…).
2. Triển khai code hiện thực pipeline.
3. Chuẩn bị slide thuyết trình mô tả kiến trúc, giải pháp và kết quả.
4. Chạy demo hệ thống trong buổi pitching.

**Tài nguyên BTC cung cấp:**
- Bộ dataset chuẩn về CLB PROPTIT.
- Metrics benchmark: Context Recall, Context Precision, MRR, Hit@k, …

**Hình thức dự thi:** Cá nhân.

---

## 3. Mốc thời gian
- **Tuần 0:** Phát động cuộc thi, gửi dataset & benchmark metrics.
- **Tuần 1–3:** Hoàn thiện pipeline, code và slide.
- **Ngày Pitching:**
  - Tối đa **30 phút** thuyết trình + **10 phút** Q&A.
  - Chạy demo code trực tiếp (có thể dùng Streamlit).

---

## 4. Tiêu chí chấm điểm
| Tiêu chí                  | Trọng số |
|---------------------------|----------|
| Kiến trúc pipeline        | 30%      |
| Hiệu năng benchmark       | 40%      |
| Chất lượng demo           | 20%      |
| Kỹ năng thuyết trình      | 10%      |

---

## 5. Giải thưởng
🥇 **Giải Nhất:** 200.000 VNĐ + Giấy chứng nhận  
🥈 **Giải Nhì:** 150.000 VNĐ + Giấy chứng nhận  
🥉 **Giải Ba:** 100.000 VNĐ + Giấy chứng nhận  
🌟 **Giải Tiềm Năng:** 50.000 VNĐ + Giấy chứng nhận  

---

## 6. Đối tượng tham gia
- Thành viên thuộc **Team AI – CLB Lập trình ProPTIT**

📌 Hãy sẵn sàng **sáng tạo & bứt phá** cùng NeoRAG Cup 2025!  
💬 Mọi thắc mắc vui lòng comment hoặc inbox BTC để được giải đáp.

---

## 📊 Benchmark

- Trong suốt cuộc thi, các bạn sẽ chỉ được cung cấp bộ dữ liệu train. Bộ dữ liệu test sẽ được BTC công bố vào ngày thi cuối cùng. Dưới đây là benchmark của baseline model — mục tiêu của bạn là xây dựng mô hình có hiệu năng vượt qua được baseline model. 

### **Retrieval – Train (100 query)** 
| K  | hit@k | recall@k | precision@k | f1@k | map@k | mrr@k | ndcg@k | context_precision@k | context_recall@k | context_entities_recall@k |
|----|-------|----------|-------------|------|-------|-------|--------|----------------------|------------------|---------------------------|
| 3  | 0.31  | 0.19     | 0.12        | 0.15 | 0.23  | 0.23  | 0.25   | 0.63                 | 0.50             | 0.32                      |
| 5  | 0.46  | 0.28     | 0.10        | 0.15 | 0.23  | 0.27  | 0.31   | 0.56                 | 0.44             | 0.37                      |
| 7  | 0.57  | 0.35     | 0.09        | 0.15 | 0.23  | 0.28  | 0.35   | 0.54                 | 0.40             | 0.38                      |

### **LLM Answer – Train (100 query)**
| K  | string_presence@k | rouge_l@k | bleu_4@k | groundedness@k | response_relevancy@k | noise_sensitivity@k |
|----|-------------------|-----------|----------|----------------|----------------------|---------------------|
| 3  | 0.35              | 0.21      | 0.03     | 0.57           | 0.80                 | 0.51                |
| 5  | 0.40              | 0.23      | 0.03     | 0.61           | 0.80                 | 0.53                |
| 7  | 0.41              | 0.22      | 0.04     | 0.64           | 0.80                 | 0.51                |

---

### **Retrieval – Test (30 query)**
| K  | hit@k | recall@k | precision@k | f1@k | map@k | mrr@k | ndcg@k | context_precision@k | context_recall@k | context_entities_recall@k |
|----|-------|----------|-------------|------|-------|-------|--------|----------------------|------------------|---------------------------|
| 3  | 0.23  | 0.06     | 0.08        | 0.07 | 0.12  | 0.12  | 0.15   | 0.34                 | 0.32             | 0.11                      |
| 5  | 0.40  | 0.10     | 0.08        | 0.09 | 0.16  | 0.16  | 0.22   | 0.35                 | 0.29             | 0.15                      |
| 7  | 0.47  | 0.13     | 0.08        | 0.10 | 0.17  | 0.17  | 0.24   | 0.31                 | 0.27             | 0.16                      |

### **LLM Answer – Test (30 query)**
| K  | string_presence@k | rouge_l@k | bleu_4@k | groundedness@k | response_relevancy@k | noise_sensitivity@k |
|----|-------------------|-----------|----------|----------------|----------------------|---------------------|
| 3  | 0.18              | 0.14      | 0.01     | 0.33           | 0.79                 | 0.68                |
| 5  | 0.16              | 0.15      | 0.01     | 0.30           | 0.79                 | 0.71                |
| 7  | 0.21              | 0.15      | 0.02     | 0.39           | 0.80                 | 0.71                |
