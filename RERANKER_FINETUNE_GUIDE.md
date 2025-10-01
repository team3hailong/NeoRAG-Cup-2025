# 🎯 Hướng Dẫn Fine-tune Reranker cho ProPTIT RAG System

## 📋 Mục Lục
1. [Giới thiệu](#giới-thiệu)
2. [Chuẩn bị](#chuẩn-bị)
3. [Chạy Fine-tune](#chạy-fine-tune)
4. [Tối ưu cho Hardware hạn chế](#tối-ưu-cho-hardware-hạn-chế)
5. [Sử dụng Model đã Fine-tune](#sử-dụng-model-đã-fine-tune)
6. [Troubleshooting](#troubleshooting)

---

## 🎓 Giới thiệu

Script `fine_tune_reranker.py` cho phép bạn fine-tune một cross-encoder reranker model trên domain-specific data của CLB ProPTIT. Reranker được train để:

- **Positive pairs**: Xếp hạng cao các document có liên quan đến query (ground truth)
- **Hard negatives**: Phân biệt được các document tương tự nhưng không phải đáp án đúng
- **Random negatives**: Reject các document không liên quan

### Kiến trúc Training

```
Query + Document → Cross-Encoder → Score (0-1)
                                    ↓
                              BCE Loss với labels:
                              - 1 cho positive pairs
                              - 0 cho negative pairs
```

**Hard Negative Mining**: Sử dụng TF-IDF/BM25 để tìm các document "gần" với query nhưng không phải ground truth, giúp model học phân biệt tốt hơn.

---

## 🛠️ Chuẩn bị

### 1. Kiểm tra Hardware

Với giới hạn hệ thống:
- **RAM**: 12.7 GB
- **GPU RAM**: 15 GB  
- **Disk**: 112.6 GB

Script đã được optimize với:
- Batch size nhỏ (4-8)
- Gradient accumulation (8 steps)
- Mixed precision (FP16)
- Gradient checkpointing
- Checkpoint saving để recovery

### 2. Cài đặt Dependencies

Đảm bảo đã cài đặt các packages cần thiết:

```bash
pip install -r requirements.txt
```

Các package chính:
- `transformers` - Hugging Face transformers
- `FlagEmbedding` - Cross-encoder reranker
- `torch` - PyTorch
- `scikit-learn` - TF-IDF cho hard negative mining
- `pandas`, `openpyxl` - Data loading

### 3. Kiểm tra Data

Đảm bảo có các file:
- `CLB_PROPTIT.csv`: Corpus (99 documents)
- `train_data_proptit.xlsx`: Training queries (100 queries)

---

## 🚀 Chạy Fine-tune

### Cấu hình Cơ bản (Khuyến nghị cho GPU 15GB)

```bash
python fine_tune_reranker.py \
  --corpus_csv CLB_PROPTIT.csv \
  --train_xlsx train_data_proptit.xlsx \
  --base_model namdp-ptit/ViRanker \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --epochs 5 \
  --lr 2e-5 \
  --fp16 \
  --gradient_checkpointing \
  --dev_ratio 0.2 \
  --num_hard_negatives 2 \
  --num_random_negatives 1
```

### Giải thích Parameters

| Parameter | Mô tả | Giá trị khuyến nghị |
|-----------|-------|---------------------|
| `--base_model` | Model reranker pretrained | `namdp-ptit/ViRanker` hoặc `BAAI/bge-reranker-v2-m3` |
| `--batch_size` | Số samples/batch (per device) | 4-8 |
| `--gradient_accumulation_steps` | Accumulate N batches trước khi update | 8 (effective batch=32) |
| `--epochs` | Số epoch training | 5-10 |
| `--lr` | Learning rate | 2e-5 (standard cho fine-tune) |
| `--fp16` | Mixed precision FP16 | Bật cho GPU |
| `--gradient_checkpointing` | Tiết kiệm GPU memory | Bật nếu OOM |
| `--dev_ratio` | % data cho validation | 0.2 (20%) |
| `--num_hard_negatives` | Hard negatives/positive | 2-3 |
| `--num_random_negatives` | Random negatives/positive | 1 |
| `--max_seq_length` | Max tokens cho query+doc | 512 |
| `--save_steps` | Lưu checkpoint mỗi N steps | 100 |
| `--eval_steps` | Evaluate mỗi N steps | 100 |

---

## ⚙️ Tối ưu cho Hardware hạn chế

### Chiến lược 1: Giảm Memory Usage

**Nếu gặp OOM (Out of Memory):**

```bash
python fine_tune_reranker.py \
  --batch_size 2 \
  --gradient_accumulation_steps 16 \
  --fp16 \
  --gradient_checkpointing \
  --max_seq_length 384
```

### Chiến lược 2: Quick Test Run

**Chạy nhanh để test setup:**

```bash
python fine_tune_reranker.py \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --epochs 1 \
  --max_steps 50 \
  --dev_ratio 0.1 \
  --num_hard_negatives 1
```

### Chiến lược 3: Maximize Quality (nếu có đủ resources)

```bash
python fine_tune_reranker.py \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --epochs 10 \
  --lr 1e-5 \
  --dev_ratio 0.2 \
  --num_hard_negatives 3 \
  --num_random_negatives 2 \
  --hard_neg_top_k 100
```

### Tính toán Effective Batch Size

```
Effective Batch Size = batch_size × gradient_accumulation_steps
```

**Ví dụ:**
- `batch_size=4` + `gradient_accumulation_steps=8` → Effective = 32
- Tương đương training với batch size 32 nhưng chỉ dùng memory cho 4 samples

---

## 📦 Sử dụng Model đã Fine-tune

### 1. Tự động Load trong main.py

File `main.py` đã được cập nhật để tự động tìm và load fine-tuned reranker mới nhất:

```python
# Tự động tìm fine-tuned reranker model mới nhất
import glob
finetuned_reranker_dirs = sorted(glob.glob("outputs/reranker-finetuned-*"), reverse=True)
if finetuned_reranker_dirs and os.path.exists(os.path.join(finetuned_reranker_dirs[0], "config.json")):
    print(f"[Reranker] Using fine-tuned reranker: {finetuned_reranker_dirs[0]}")
    reranker = Reranker(model_name=finetuned_reranker_dirs[0])
else:
    print("[Reranker] Using pretrained reranker: namdp-ptit/ViRanker")
    reranker = Reranker(model_name="namdp-ptit/ViRanker")
```

### 2. Manual Load

```python
from rerank import Reranker

# Thay thế path bằng output directory của bạn
reranker = Reranker(model_name="outputs/reranker-finetuned-20250101-120000")

# Sử dụng
query = "CLB PROPTIT hoạt động như thế nào?"
passages = [doc1, doc2, doc3, ...]

scores, ranked_passages = reranker(query, passages)
```

### 3. Kiểm tra Quality

Sau khi fine-tune, chạy metrics để đánh giá:

```bash
python main.py
```

So sánh các metrics với baseline (xem `README.md` hoặc `.github/copilot-instructions.md`):
- **context_precision@k**: Tăng → ít noise hơn
- **context_recall@k**: Tăng → tìm được nhiều relevant docs hơn
- **hit@k**: Tăng → ground truth xuất hiện trong top-k
- **mrr@k**: Tăng → ground truth xuất hiện ở vị trí cao hơn

---

## 🔧 Troubleshooting

### Problem 1: CUDA Out of Memory

**Triệu chứng:**
```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
1. Giảm `--batch_size` (thử 2 hoặc 1)
2. Bật `--gradient_checkpointing`
3. Giảm `--max_seq_length` (thử 384 hoặc 256)
4. Tăng `--gradient_accumulation_steps` để giữ effective batch size

```bash
python fine_tune_reranker.py --batch_size 2 --gradient_accumulation_steps 16 --gradient_checkpointing
```

### Problem 2: Training quá chậm

**Giải pháp:**
1. Đảm bảo đã bật `--fp16`
2. Kiểm tra GPU đang được sử dụng: `torch.cuda.is_available()`
3. Giảm `--num_hard_negatives` và `--num_random_negatives`
4. Tăng `--batch_size` nếu có đủ memory

### Problem 3: Model không converge / Loss không giảm

**Nguyên nhân có thể:**
- Learning rate quá cao hoặc quá thấp
- Dữ liệu không đủ đa dạng
- Hard negatives quá dễ

**Giải pháp:**
1. Thử learning rate khác (1e-5, 5e-6)
2. Tăng số epoch
3. Tăng `--num_hard_negatives` và `--hard_neg_top_k`
4. Kiểm tra dev metrics để xác nhận overfitting

```bash
# Thử với learning rate thấp hơn
python fine_tune_reranker.py --lr 1e-5 --epochs 10
```

### Problem 4: Checkpoint bị corrupted

**Giải pháp:**
Script tự động lưu checkpoint mỗi `--save_steps`. Nếu training bị gián đoạn:

```bash
# Resume từ checkpoint mới nhất
python fine_tune_reranker.py --base_model outputs/reranker-finetuned-XXXXXX/checkpoint-500
```

### Problem 5: Windows DataLoader multiprocessing issues

**Triệu chứng:**
```
RuntimeError: DataLoader worker (pid XXXX) is killed
```

**Giải pháp:**
Script đã set `dataloader_num_workers=0` mặc định. Nếu vẫn gặp lỗi, đảm bảo:
- Chạy script trong `if __name__ == "__main__":` block
- Không sử dụng Jupyter Notebook (chạy từ terminal)

---

## 📊 Monitoring Training

### 1. Training Logs

Script sẽ in ra:
- Loss mỗi `--logging_steps`
- Dev metrics mỗi `--eval_steps`
- Progress bar với tqdm

### 2. Output Files

Sau khi training xong, directory `outputs/reranker-finetuned-XXXXXX/` chứa:
- `pytorch_model.bin`: Model weights
- `config.json`: Model config
- `tokenizer_config.json`: Tokenizer config
- `training_info.json`: Training hyperparameters
- `dev_metrics.json`: Final dev set metrics
- `checkpoint-XXX/`: Intermediate checkpoints

### 3. Dev Metrics

Kiểm tra `dev_metrics.json`:
```json
{
  "accuracy": 0.85,
  "avg_pos_score": 0.78,
  "avg_neg_score": 0.32,
  "margin": 0.46
}
```

**Giải thích:**
- `accuracy`: Tỷ lệ phân loại đúng (positive vs negative)
- `avg_pos_score`: Điểm trung bình cho positive pairs (càng cao càng tốt)
- `avg_neg_score`: Điểm trung bình cho negative pairs (càng thấp càng tốt)
- `margin`: Khoảng cách giữa pos và neg scores (càng lớn càng tốt)

**Target:**
- Accuracy > 0.80
- Margin > 0.40

---

## 🎯 Best Practices

### 1. Data Quality
- Đảm bảo ground truth documents chính xác
- Review và clean data trước khi train
- Xem xét augment thêm queries nếu có thể

### 2. Hyperparameter Tuning
- Bắt đầu với default parameters
- Adjust dựa trên dev metrics
- Grid search trên learning rate và epochs

### 3. Model Selection
- `namdp-ptit/ViRanker`: Tốt cho tiếng Việt
- `BAAI/bge-reranker-v2-m3`: Multilingual, performant

### 4. Training Strategy
- Train với full data sau khi verify với small test
- Save best model based on dev metrics
- Keep checkpoints để rollback nếu overfit

### 5. Evaluation
- Luôn evaluate trên test set sau khi train
- So sánh với baseline pretrained model
- Test trên real queries để xác nhận quality

---

## 📝 Example: Full Pipeline

```bash
# 1. Test setup nhanh (5-10 phút)
python fine_tune_reranker.py \
  --batch_size 4 \
  --epochs 1 \
  --max_steps 20

# 2. Nếu OK, chạy full training (30-60 phút)
python fine_tune_reranker.py \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --epochs 5 \
  --fp16 \
  --gradient_checkpointing

# 3. Evaluate
python main.py

# 4. So sánh metrics với baseline trong README.md
```

---

## 🤝 Contributing & Support

Nếu gặp vấn đề:
1. Kiểm tra [Troubleshooting](#troubleshooting)
2. Review training logs
3. Thử adjust hyperparameters
4. Liên hệ team AI - CLB ProPTIT

**Good luck với fine-tuning! 🚀**
