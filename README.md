# 🚀 NeoRAG Cup 2025  
**Thiết kế & Hiện thực hóa Pipeline RAG tối ưu cho CLB ProPTIT**  

---

## 1. Giới thiệu  
**NeoRAG Cup 2025** là cuộc thi **học thuật – kỹ thuật** do **Team AI – CLB Lập trình ProPTIT** tổ chức, dành cho các bạn đam mê **Trí tuệ nhân tạo (AI)**, **Xử lý ngôn ngữ tự nhiên (NLP)** và **Kỹ thuật hệ thống**.  

Người tham gia sẽ:  
- Tự thiết kế, hiện thực hóa và trình bày một **pipeline RAG** (*Retrieval-Augmented Generation*) với domain là thông tin của CLB ProPTIT.  
- Trải nghiệm toàn bộ quy trình phát triển sản phẩm AI từ **ý tưởng → triển khai → demo**.  

---

## 2. Thể lệ & Yêu cầu  

**Domain:** Thông tin liên quan đến CLB ProPTIT (*lịch sử, thành viên, hoạt động, dự án, tài liệu nội bộ, v.v.*)  

**Nhiệm vụ:**  
- Thiết kế pipeline RAG hoàn chỉnh (*kiến trúc, công nghệ, chiến lược index, retrieval, reranking, generation…*).  
- Triển khai code hiện thực pipeline.  
- Chuẩn bị slide thuyết trình mô tả kiến trúc, giải pháp và kết quả.  
- Chạy demo hệ thống trong buổi pitching.  

**Tài nguyên BTC cung cấp:**  
- Bộ dataset chuẩn về CLB PROPTIT.  
- Metrics benchmark: *Context Recall, Context Precision, MRR, Hit@k, …*  

**Hình thức dự thi:** Cá nhân.  

---

## 3. Mốc thời gian  

- **Tuần 0:** Phát động cuộc thi, gửi dataset & benchmark metrics.  
- **Tuần 1–3:** Hoàn thiện pipeline, code và slide.  
- **Ngày Pitching:**  
  - Tối đa **30 phút thuyết trình** + **10 phút Q&A**.  
  - Chạy demo code trực tiếp (*có thể làm bằng Streamlit*).  

---

## 4. Tiêu chí chấm điểm  

| Hạng mục               | Tỉ lệ |
|------------------------|-------|
| Kiến trúc pipeline     | 30%   |
| Hiệu năng benchmark    | 40%   |
| Chất lượng demo        | 20%   |
| Kỹ năng thuyết trình   | 10%   |

---

## 5. Giải thưởng  

🥇 **Giải Nhất** – 200.000 VNĐ + Giấy chứng nhận  
🥈 **Giải Nhì** – 150.000 VNĐ + Giấy chứng nhận  
🥉 **Giải Ba** – 100.000 VNĐ + Giấy chứng nhận  
🌟 **Giải Tiềm Năng** – 50.000 VNĐ + Giấy chứng nhận  

---

## 6. Đối tượng tham gia  
Thành viên thuộc **Team AI – CLB Lập trình ProPTIT**.  

---

📌 **Hãy sẵn sàng sáng tạo & bứt phá cùng NeoRAG Cup 2025!**  
💬 Mọi thắc mắc vui lòng comment hoặc inbox BTC để được giải đáp.  

---

## 📊 Benchmark Kết Quả Trên Tập Train  

### 1. Retrieval Benchmark (Base Model)

| K  | hit@k | recall@k | precision@k | f1@k | map@k | mrr@k | ndcg@k | context_precision@k | context_recall@k | context_entities_recall@k |
|----|-------|----------|-------------|------|-------|-------|--------|---------------------|------------------|---------------------------|
| 3  | 0.31  | 0.19     | 0.12        | 0.15 | 0.23  | 0.23  | 0.25   | 0.63                | 0.50             | 0.32                      |
| 5  | 0.46  | 0.28     | 0.10        | 0.15 | 0.23  | 0.27  | 0.31   | 0.56                | 0.44             | 0.37                      |
| 7  | 0.57  | 0.35     | 0.09        | 0.15 | 0.23  | 0.28  | 0.35   | 0.54                | 0.40             | 0.38                      |


---

### 2. LLM Answer Benchmark (Base Model)

| K  | string_presence@k | rouge_l@k | bleu_4@k | groundedness@k | response_relevancy@k | noise_sensitivity@k |
|----|-------------------|-----------|----------|----------------|----------------------|---------------------|
| 3  | 0.35              | 0.21      | 0.03     | 0.57           | 0.80                 | 0.51                |
| 5  | 0.40              | 0.23      | 0.03     | 0.61           | 0.80                 | 0.53                |
| 7  | 0.41              | 0.22      | 0.04     | 0.64           | 0.80                 | 0.51                |

---
