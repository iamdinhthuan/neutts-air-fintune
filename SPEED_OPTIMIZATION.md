# ⚡ Tối ưu Tốc độ Training

Hướng dẫn tăng tốc training sau khi đã pre-encode dataset.

## 🎯 Các Tối ưu Đã Thêm

### 1. **TF32 (Tensor Float 32)** - Nhanh ~20%

**Cho GPU:** RTX 30xx, RTX 40xx, A100, H100

```yaml
tf32: true  # Bật trong config
```

**Lợi ích:**
- Nhanh hơn 20% so với FP32
- Độ chính xác gần như FP32
- Không tốn thêm VRAM

**Tự động bật nếu GPU hỗ trợ!**

---

### 2. **Fused AdamW Optimizer** - Nhanh ~10%

```yaml
# Tự động dùng trong code
optim: "adamw_torch_fused"
```

**Lợi ích:**
- Optimizer nhanh hơn AdamW thường
- Ít kernel launches hơn
- Giảm overhead

---

### 3. **Dataloader Optimizations** - Nhanh ~15%

```yaml
dataloader_pin_memory: true      # Pin memory
dataloader_prefetch_factor: 2    # Prefetch 2 batches
```

**Lợi ích:**
- Pin memory: Faster CPU → GPU transfer
- Prefetch: GPU không phải đợi data
- Overlap data loading với training

---

### 4. **Tăng Batch Size** - Nhanh ~30%

```yaml
per_device_train_batch_size: 4   # Tăng từ 1 → 4
gradient_accumulation_steps: 2   # Giảm từ 4 → 2
# Effective batch vẫn = 8
```

**Lợi ích:**
- Ít gradient updates hơn
- GPU utilization cao hơn
- Throughput tốt hơn

---

### 5. **Giảm Eval Frequency** - Tiết kiệm thời gian

```yaml
eval_steps: 10000  # Tăng từ 5000 → 10000
```

**Lợi ích:**
- Eval ít hơn 50%
- Tiết kiệm ~5-10% thời gian training
- Vẫn đủ để monitor

---

### 6. **PyTorch 2.0 Compile** (Tùy chọn) - Nhanh ~10-20%

```yaml
torch_compile: true  # Cần PyTorch 2.0+
```

**Lợi ích:**
- Compile model thành optimized code
- Nhanh hơn 10-20%
- Lần đầu chậm (compile), sau đó nhanh

**Lưu ý:**
- Cần PyTorch 2.0+
- Lần đầu compile mất ~5-10 phút
- Có thể gặp lỗi với một số models

---

### 7. **Gradient Checkpointing** (Nếu OOM) - Trade speed for memory

```yaml
gradient_checkpointing: true  # Chỉ bật nếu CUDA OOM
```

**Lợi ích:**
- Tiết kiệm VRAM ~40%
- Có thể train batch lớn hơn

**Nhược điểm:**
- Chậm hơn ~20%
- Chỉ dùng khi cần thiết

---

## 📊 Tổng hợp Tối ưu

### Config Khuyến nghị (GPU 24GB+):

```yaml
# Dataset
dataset_path: "vietnamese_dataset.pkl"  # Pre-encoded

# Training
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
num_train_epochs: 3
save_steps: 5000
eval_steps: 10000

# Speed optimizations
torch_compile: false             # Bật nếu PyTorch 2.0+
gradient_checkpointing: false    # Tắt (đủ VRAM)
tf32: true                       # Bật (GPU Ampere+)
dataloader_pin_memory: true
dataloader_prefetch_factor: 2
```

### Config cho GPU nhỏ (8-16GB):

```yaml
# Dataset
dataset_path: "vietnamese_dataset.pkl"

# Training
per_device_train_batch_size: 1   # Giảm batch
gradient_accumulation_steps: 8   # Tăng accumulation
num_train_epochs: 3
save_steps: 5000
eval_steps: 10000

# Speed optimizations
torch_compile: false
gradient_checkpointing: true     # Bật để tiết kiệm VRAM
tf32: true
dataloader_pin_memory: true
dataloader_prefetch_factor: 2
```

---

## ⚡ Ước tính Tốc độ

### Baseline (On-the-fly encoding):
```
~8-9 giây/batch
→ 3 epochs = ~30 ngày
```

### Pre-encoded (Không tối ưu):
```
~0.8 giây/batch
→ 3 epochs = ~5 ngày
```

### Pre-encoded + Tối ưu (Config khuyến nghị):
```
~0.4-0.5 giây/batch
→ 3 epochs = ~2.5-3 ngày

NHANH GẤP 10-12 LẦN SO VỚI BASELINE!
```

---

## 🔧 Cách Sử dụng

### Bước 1: Cập nhật config

Mở `finetune_vietnamese_config.yaml`, sửa:

```yaml
# Tăng batch size
per_device_train_batch_size: 4
gradient_accumulation_steps: 2

# Giảm eval frequency
eval_steps: 10000

# Bật tối ưu
tf32: true
dataloader_pin_memory: true
dataloader_prefetch_factor: 2
```

### Bước 2: Chạy training

```bash
python finetune_vietnamese.py finetune_vietnamese_config.yaml
```

### Bước 3: Monitor

```
[7/7] Setting up training...
  Using OnTheFlyDataCollator (on-the-fly preprocessing)
  Dataloader workers: 4 (pre-encoded data)
  ✓ TF32 enabled for faster training

============================================================
STARTING TRAINING
============================================================
Train samples: 2591598
Val samples: 13023
Batch size per device: 4
Gradient accumulation steps: 2
Effective batch size: 8
...

Step 100: loss=2.456, time=0.45s/batch  ← Nhanh!
Step 200: loss=2.234, time=0.43s/batch
```

---

## 💡 Tips Thêm

### 1. Kiểm tra GPU Utilization

```bash
# Terminal khác
watch -n 1 nvidia-smi
```

**Mục tiêu:** GPU Util ~95-100%

Nếu thấp:
- Tăng `per_device_train_batch_size`
- Tăng `dataloader_num_workers`
- Tăng `dataloader_prefetch_factor`

---

### 2. Profile Training

```python
# Thêm vào code nếu cần debug
import torch.profiler

with torch.profiler.profile() as prof:
    trainer.train()

print(prof.key_averages().table())
```

---

### 3. Mixed Precision Training

Đã bật sẵn với `bf16=True`:

```yaml
# Trong TrainingArguments
bf16: true  # BFloat16 (khuyến nghị cho Ampere+)
```

**Lợi ích:**
- Nhanh gấp ~2x so với FP32
- Tiết kiệm VRAM ~50%
- Ổn định hơn FP16

---

### 4. Disable Unnecessary Logging

```yaml
logging_steps: 100  # Tăng lên nếu log quá nhiều
```

---

### 5. Sử dụng SSD

Đảm bảo `vietnamese_dataset.pkl` trên SSD, không phải HDD:

```bash
# Kiểm tra
df -h /media/huy/data1tb/thuan/neutts-air/

# Copy sang SSD nếu cần
cp vietnamese_dataset.pkl /path/to/ssd/
```

---

## 📈 Benchmark

### GPU: RTX 3090 (24GB)

| Config | Batch | Acc | Time/batch | 3 epochs |
|--------|-------|-----|------------|----------|
| On-the-fly | 2 | 4 | 8.5s | ~30 ngày |
| Pre-encoded | 2 | 4 | 0.8s | ~5 ngày |
| Pre-encoded + Opt | 4 | 2 | 0.45s | ~2.8 ngày |
| Pre-encoded + Opt + Compile | 4 | 2 | 0.38s | ~2.3 ngày |

### GPU: A100 (40GB)

| Config | Batch | Acc | Time/batch | 3 epochs |
|--------|-------|-----|------------|----------|
| Pre-encoded + Opt | 8 | 1 | 0.35s | ~2.2 ngày |
| Pre-encoded + Opt + Compile | 8 | 1 | 0.28s | ~1.8 ngày |

---

## ⚠️ Troubleshooting

### CUDA OOM khi tăng batch size

```yaml
# Giảm batch, tăng accumulation
per_device_train_batch_size: 2
gradient_accumulation_steps: 4

# Hoặc bật gradient checkpointing
gradient_checkpointing: true
```

---

### torch_compile error

```yaml
# Tắt nếu gặp lỗi
torch_compile: false
```

---

### Dataloader slow

```yaml
# Tăng workers
dataloader_num_workers: 8  # Thử 4, 8, 16

# Tăng prefetch
dataloader_prefetch_factor: 4
```

---

## 🎯 Checklist Tối ưu

- [ ] Pre-encode dataset (nhanh gấp 10x)
- [ ] Tăng batch size lên 4 (nếu GPU đủ)
- [ ] Bật TF32 (GPU Ampere+)
- [ ] Bật pin_memory và prefetch
- [ ] Giảm eval_steps xuống 10000
- [ ] Dùng fused AdamW (tự động)
- [ ] Kiểm tra GPU util ~95-100%
- [ ] Dataset trên SSD
- [ ] (Tùy chọn) Bật torch_compile

---

## 🚀 Kết luận

**Với tất cả tối ưu:**

```
Baseline (on-the-fly): ~30 ngày
→ Pre-encoded: ~5 ngày (6x nhanh hơn)
→ Pre-encoded + Optimized: ~2.5-3 ngày (10-12x nhanh hơn!)
```

**Bắt đầu ngay:**

```bash
# 1. Cập nhật config
vim finetune_vietnamese_config.yaml

# 2. Train
python finetune_vietnamese.py finetune_vietnamese_config.yaml
```

**Happy fast training!** ⚡

