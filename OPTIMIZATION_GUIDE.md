# Tối ưu hóa tham số để tăng điểm benchmark NeoRAG Cup 2025

## 1. PHÂN TÍCH THAM SỐ HIỆN TẠI VS TỐI ƯU

### Epochs (3 → 5-8)
**Tối ưu:** `--epochs 6`
- **Lý do:** 3 epochs ít cho fine-tuning embedding model
- **Trade-off:** Nhiều epochs hơn = better convergence nhưng risk overfitting
- **Tối ưu:** 5-8 epochs với early stopping

### Batch Size (8 → 12-16) 
**Tối ưu:** `--batch_size 12`
- **Lý do:** T4 14.7GB có thể handle batch lớn hơn
- **Benefit:** Stable gradients, better negative sampling trong MultipleNegativesRankingLoss
- **Max safe:** 16 (nhưng 12 an toàn hơn)

### Max Sequence Length (384 → 512)
**Tối ưu:** `--max_seq_length 512`
- **Lý do:** BGE-M3 trained với 512 tokens, truncating làm mất thông tin
- **Impact:** Lớn nhất với context_recall và context_precision metrics
- **Trade-off:** Chậm hơn nhưng accuracy cao hơn rõ rệt

### Freeze Layers (4 → 0-2)
**Tối ưu:** `--freeze_layers 2` hoặc `0`
- **Lý do:** Freeze ít layer = model học domain-specific patterns tốt hơn
- **Risk:** Catastrophic forgetting, nhưng với learning rate thấp thì OK
- **Compromise:** 2 layers (chỉ freeze embeddings + first layer)

### Learning Rate (default 2e-5 → 1e-5 hoặc 3e-5)
**Tối ưu:** `--lr 1e-5` (conservative) hoặc `1.5e-5`
- **Lý do:** BGE-M3 đã pre-trained tốt, cần lr nhỏ để fine-tune stable
- **Alternative:** 3e-5 nếu muốn học nhanh (nhưng riskier)

### Dev Ratio
- **Thêm:** `--dev_ratio 0.15` (giảm từ 0.2 xuống 0.15)
- **Lý do:** Nhiều data train hơn = better performance

### Gradient Accumulation 
- **Thêm:** `--gradient_accumulation_steps 2`
- **Benefit:** Equivalent batch_size 20-24 nhưng fit trong memory

### Scheduler
- **Thay:** `--scheduler WarmupCosine` thay vì WarmupLinear
- **Benefit:** Better convergence curve cho fine-tuning

## 2. Template:
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
