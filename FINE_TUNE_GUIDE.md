# 🎯 Hướng Dẫn Fine-tune Toàn Diện cho NeoRAG Cup 2025

## 📋 Mục Lục
1. [Giới thiệu](#giới-thiệu)
2. [Fine-tune Embedding Model (BGE-M3)](#fine-tune-embedding-model-bge-m3)
3. [Fine-tune Reranker](#fine-tune-reranker)
4. [Tối ưu hóa và Troubleshooting](#tối-ưu-hóa-và-troubleshooting)
5. [Best Practices](#best-practices)

---

## 🎓 Giới thiệu

Fine-tuning là bước quan trọng để tối ưu hóa RAG system cho domain cụ thể của CLB ProPTIT. Trong NeoRAG Cup 2025, việc fine-tune cả embedding model và reranker giúp cải thiện đáng kể các metrics benchmark như context_recall, context_precision, và response_relevancy.

### Quy trình Fine-tune
1. Chuẩn bị dữ liệu (train/dev sets)
2. Cấu hình hyperparameters tối ưu
3. Training với monitoring
4. Evaluation và iteration

### Hyperparameters Chính

*Lưu ý: Các template dưới đây đã chỉ định giá trị tối ưu cụ thể cho từng model. Bảng này chỉ hướng dẫn điều chỉnh nếu điểm số benchmark thấp (tuyến tính).*

| Tham số/Model | Hướng điều chỉnh (nếu điểm số thấp) |
|---------------|-------------------------------------|
| `epochs` | Tăng để convergence tốt hơn |
| `batch_size` | Giảm nếu OOM, tăng nếu gradients unstable |
| `max_seq_length` | Giữ nguyên (cố định) |
| `learning_rate` | Giảm nếu không convergence, tăng nhẹ nếu quá chậm |
| `gradient_accumulation_steps` | Tăng để effective batch lớn hơn |
| `freeze_layers` | Giảm để học nhiều hơn (chỉ cho embedding) |
| `num_hard_negatives` | Tăng để discrimination tốt hơn (chỉ cho reranker) |
| `scheduler` | Giữ nguyên |
| `dev_ratio` | Giảm để nhiều data train

---

## 🔧 Templete Fine-tune

### Template Chạy Fine-tune Embedding

```bash
python fine_tune_bge_m3.py \
  --epochs 5 \
  --batch_size 8 \
  --max_seq_length 384 \
  --evaluation_steps 15 \
  --lr 1e-5 \
  --checkpoint_save_steps 30 \
  --freeze_layers 4 \
  --warmup_steps 100 \
  --resume_from_checkpoint auto
```

### Template Chạy Fine-tune Reranker

```bash
python fine_tune_reranker.py \
  --batch_size 3 \
  --gradient_accumulation_steps 8 \
  --epochs 4 \
  --lr 2e-5 \
  --fp16 \
  --gradient_checkpointing \
  --dev_ratio 0.2 \
  --num_hard_negatives 2 \
  --num_random_negatives 1
```

