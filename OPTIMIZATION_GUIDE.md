# Tối ưu hóa tham số để tăng điểm benchmark NeoRAG Cup 2025

## 1. PHÂN TÍCH THAM SỐ HIỆN TẠI VS TỐI ƯU

### Epochs (3 → 5-8)
**Hiện tại:** `--epochs 3`
**Tối ưu:** `--epochs 6`
- **Lý do:** 3 epochs ít cho fine-tuning embedding model
- **Trade-off:** Nhiều epochs hơn = better convergence nhưng risk overfitting
- **Tối ưu:** 5-8 epochs với early stopping

### Batch Size (8 → 12-16) 
**Hiện tại:** `--batch_size 8`
**Tối ưu:** `--batch_size 12`
- **Lý do:** T4 14.7GB có thể handle batch lớn hơn
- **Benefit:** Stable gradients, better negative sampling trong MultipleNegativesRankingLoss
- **Max safe:** 16 (nhưng 12 an toàn hơn)

### Max Sequence Length (384 → 512)
**Hiện tại:** `--max_seq_length 384` 
**Tối ưu:** `--max_seq_length 512`
- **Lý do:** BGE-M3 trained với 512 tokens, truncating làm mất thông tin
- **Impact:** Lớn nhất với context_recall và context_precision metrics
- **Trade-off:** Chậm hơn nhưng accuracy cao hơn rõ rệt

### Freeze Layers (4 → 0-2)
**Hiện tại:** `--freeze_layers 4`
**Tối ưu:** `--freeze_layers 2` hoặc `0`
- **Lý do:** Freeze ít layer = model học domain-specific patterns tốt hơn
- **Risk:** Catastrophic forgetting, nhưng với learning rate thấp thì OK
- **Compromise:** 2 layers (chỉ freeze embeddings + first layer)

### Learning Rate (default 2e-5 → 1e-5 hoặc 3e-5)
**Hiện tại:** Không specify (dùng 2e-5)
**Tối ưu:** `--lr 1e-5` (conservative) hoặc `1.5e-5`
- **Lý do:** BGE-M3 đã pre-trained tốt, cần lr nhỏ để fine-tune stable
- **Alternative:** 3e-5 nếu muốn học nhanh (nhưng riskier)

## 2. CONFIG TỐI ƯU CHO TESLA T4

### A. Conservative (An toàn, tăng 10-15% score):
```bash
python fine_tune_bge_m3.py \
  --epochs 5 \
  --batch_size 10 \
  --max_seq_length 512 \
  --lr 1e-5 \
  --evaluation_steps 20 \
  --checkpoint_save_steps 40 \
  --freeze_layers 2 \
  --warmup_steps 100 \
  --resume_from_checkpoint auto
```

### B. Aggressive (Tăng 20-30% score, risk cao hơn):
```bash
python fine_tune_bge_m3.py \
  --epochs 8 \
  --batch_size 12 \
  --max_seq_length 512 \
  --lr 1.5e-5 \
  --evaluation_steps 25 \
  --checkpoint_save_steps 50 \
  --freeze_layers 0 \
  --warmup_steps 150 \
  --scheduler WarmupCosine \
  --resume_from_checkpoint auto
```

### C. With Advanced Features (Tăng 30-40% score):
```bash
python fine_tune_bge_m3.py \
  --epochs 6 \
  --batch_size 10 \
  --max_seq_length 512 \
  --lr 1.2e-5 \
  --evaluation_steps 25 \
  --checkpoint_save_steps 50 \
  --freeze_layers 1 \
  --warmup_steps 120 \
  --use_triplet \
  --num_hard_negatives 2 \
  --neg_source hybrid \
  --hybrid_alpha 0.7 \
  --scheduler WarmupCosine \
  --resume_from_checkpoint auto
```

## 3. CÁC THAY ĐỔI QUAN TRỌNG KHÁC

### Dev Ratio
- **Thêm:** `--dev_ratio 0.15` (giảm từ 0.2 xuống 0.15)
- **Lý do:** Nhiều data train hơn = better performance

### Gradient Accumulation 
- **Thêm:** `--gradient_accumulation_steps 2`
- **Benefit:** Equivalent batch_size 20-24 nhưng fit trong memory

### Scheduler
- **Thay:** `--scheduler WarmupCosine` thay vì WarmupLinear
- **Benefit:** Better convergence curve cho fine-tuning

## 4. QUY TRÌNH TRAINING TỐI ƯU

### Step 1: Baseline cải tiến
```bash
python fine_tune_bge_m3.py \
  --epochs 5 \
  --batch_size 10 \
  --max_seq_length 512 \
  --lr 1e-5 \
  --freeze_layers 2 \
  --dev_ratio 0.15 \
  --warmup_steps 100 \
  --scheduler WarmupCosine
```

### Step 2: Nếu Step 1 OK, thêm Triplet Loss
```bash
python fine_tune_bge_m3.py \
  --epochs 4 \
  --batch_size 8 \
  --max_seq_length 512 \
  --lr 1.2e-5 \
  --freeze_layers 1 \
  --use_triplet \
  --num_hard_negatives 2 \
  --neg_source hybrid \
  --resume_from_checkpoint auto
```

### Step 3: Final polish (nếu có thời gian)
```bash
python fine_tune_bge_m3.py \
  --epochs 2 \
  --batch_size 6 \
  --max_seq_length 512 \
  --lr 5e-6 \
  --freeze_layers 0 \
  --use_triplet \
  --resume_from_checkpoint auto
```

## 5. DỰ ĐOÁN CẢI THIỆN METRICS

### Với Conservative config:
- **hit@k**: +15-20%
- **recall@k**: +20-25%  
- **context_precision**: +10-15%
- **context_recall**: +25-30% (do max_seq_length 512)
- **rouge_l**: +15-20%

### Với Aggressive + Triplet:
- **hit@k**: +30-40%
- **mrr@k**: +25-35%
- **context metrics**: +30-45%
- **groundedness**: +20-30%

## 6. MONITORING TIPS

- Theo dõi validation loss không tăng > 3 steps liên tiếp
- Nếu overfitting: tăng freeze_layers hoặc giảm lr
- Nếu underfitting: giảm freeze_layers, tăng epochs
- Context metrics quan trọng nhất cho RAG pipeline