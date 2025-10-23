# 🚀 Hướng dẫn Training Nhanh với Pre-encoded Dataset

## Tại sao Pre-encode?

### ❌ On-the-fly encoding (Hiện tại):
```
Mỗi batch (8 samples):
- Load 8 audio files: ~2 giây
- Encode với NeuCodec: ~6 giây
- Phonemize text: ~0.5 giây
→ Total: ~8-9 giây/batch

Với 2.6M samples, 3 epochs:
→ ~30 ngày training
```

### ✅ Pre-encoded dataset (Khuyến nghị):
```
Pre-encode 1 lần (36-48 giờ):
- Encode toàn bộ 2.6M samples
- Lưu vào file .pkl

Training (chỉ load data):
- Load 8 samples từ RAM: ~0.1 giây
- Phonemize text: ~0.5 giây
→ Total: ~0.6 giây/batch

Với 2.6M samples, 3 epochs:
→ ~3-5 ngày training (nhanh gấp 6-10 lần!)
```

---

## 📋 BƯỚC 1: Pre-encode Dataset

### Chuẩn bị

Đảm bảo bạn có:
- ✅ `metadata.csv` với format: `audio|transcript`
- ✅ Thư mục `wavs/` chứa audio files
- ✅ Đủ disk space: ~10-20GB cho 2.6M samples

### Chạy Pre-encoding

```bash
# Cách 1: Dùng đường dẫn đầy đủ (Khuyến nghị)
python prepare_vietnamese_dataset.py \
    --metadata "/media/huy/data1tb/thuan/dataset/metadata.csv" \
    --audio_dir "/media/huy/data1tb/thuan/dataset/wavs" \
    --output "vietnamese_dataset.pkl" \
    --device "cuda"

# Cách 2: Nếu metadata.csv và wavs/ ở cùng thư mục với script
python prepare_vietnamese_dataset.py \
    --metadata "metadata.csv" \
    --audio_dir "wavs" \
    --output "vietnamese_dataset.pkl" \
    --device "cuda"
```

### Output mẫu:

```
============================================================
PREPARING VIETNAMESE DATASET FOR NEUTTS-AIR
============================================================

[1/4] Loading NeuCodec model on cuda...
✓ NeuCodec loaded successfully!

[2/4] Reading metadata from /media/huy/data1tb/thuan/dataset/metadata.csv...
✓ Found 2,604,621 samples

[3/4] Encoding audio files with NeuCodec...
Encoding: 100%|████████████| 2604621/2604621 [36:24:15<00:00, 19.87it/s]
✓ Successfully encoded 2,604,621 samples

[4/4] Saving encoded dataset to vietnamese_dataset.pkl...
✓ Dataset saved successfully!

============================================================
DATASET STATISTICS
============================================================
Total samples: 2,604,621
Average codes length: 245.3
Average text length: 67.2 characters

Sample data:
  Audio: vivoice_0.wav
  Text: Xin chào Việt Nam
  Codes length: 234
  First 10 codes: [12345, 23456, 34567, ...]

✅ Dataset preparation complete!
============================================================
```

### Thời gian ước tính:

| Samples | GPU | Thời gian |
|---------|-----|-----------|
| 10,000 | RTX 3090 | ~8 phút |
| 100,000 | RTX 3090 | ~1.5 giờ |
| 1,000,000 | RTX 3090 | ~15 giờ |
| 2,604,621 | RTX 3090 | ~36-40 giờ |

**💡 Tip:** Chạy qua đêm hoặc cuối tuần!

---

## 📋 BƯỚC 2: Cấu hình Training

Sửa `finetune_vietnamese_config.yaml`:

```yaml
# Vietnamese Finetuning Configuration for NeuTTS-Air

# Model settings
restore_from: "neuphonic/neutts-air"
codebook_size: 65536
max_seq_len: 2048

# Dataset Configuration
# ✅ SỬ DỤNG PRE-ENCODED DATASET
dataset_path: "vietnamese_dataset.pkl"  # File vừa tạo ở Bước 1

# ❌ KHÔNG CẦN audio_dir nữa (đã encode rồi)
# audio_dir: "/media/huy/data1tb/thuan/dataset/wavs"

# KHÔNG CẦN max_samples (dùng toàn bộ)
max_samples: null

# Training hyperparameters
lr: 0.00004
lr_scheduler_type: "cosine"
warmup_ratio: 0.05

# Training configuration
per_device_train_batch_size: 4  # Tăng lên 4 (nhanh hơn on-the-fly)
gradient_accumulation_steps: 2   # Giảm xuống 2 (effective batch vẫn = 8)
num_train_epochs: 3
logging_steps: 100
save_steps: 5000
eval_steps: 5000

# Output
save_root: "./checkpoints"
run_name: "neutts-vietnamese"

# Other
seed: 1337
```

**Thay đổi quan trọng:**
- ✅ `dataset_path: "vietnamese_dataset.pkl"` (thay vì CSV)
- ✅ `per_device_train_batch_size: 4` (tăng từ 2 → 4)
- ✅ `gradient_accumulation_steps: 2` (giảm từ 4 → 2)
- ✅ Xóa `audio_dir` (không cần nữa)

---

## 📋 BƯỚC 3: Training

```bash
python finetune_vietnamese.py finetune_vietnamese_config.yaml
```

### Output mẫu:

```
============================================================
FINETUNING NEUTTS-AIR FOR VIETNAMESE
============================================================

[1/6] Loading config from finetune_vietnamese_config.yaml
✓ Checkpoints will be saved to: ./checkpoints/neutts-vietnamese

[2/6] Loading model and tokenizer from neuphonic/neutts-air
✓ Tokenizer loaded
✓ Model loaded: 494,033,920 parameters

[3/6] Initializing Vietnamese phonemizer...
✓ Phonemizer initialized successfully!
  Test: 'Xin chào' → s i n tɕ a ː w

[4/6] Loading Vietnamese dataset...
  Loading pre-encoded dataset from vietnamese_dataset.pkl
✓ Loaded 2,604,621 pre-encoded samples
  Dataset columns: ['audio_file', 'text', 'codes']

[5/6] Preprocessing dataset...
  📦 USING PRE-ENCODED DATA
  Preprocessing all samples...
Preprocessing: 100%|████████| 2604621/2604621 [00:45:23<00:00, 956.12it/s]
✓ Dataset ready: 2,604,621 samples

[6/7] Splitting dataset into train/val...
✓ Train: 2,591,598 samples (99.5%)
✓ Val: 13,023 samples (0.5%)

[7/7] Setting up training...
  Using default data collator

============================================================
STARTING TRAINING
============================================================
Train samples: 2,591,598
Val samples: 13,023
Batch size per device: 4
Gradient accumulation steps: 2
Effective batch size: 8
Training epochs: 3
Estimated total steps: ~971,847
Learning rate: 4e-05
Save every: 5000 steps
Eval every: 5000 steps
============================================================

Step 100: loss=2.456, lr=4.2e-05
Step 200: loss=2.234, lr=4.3e-05
Step 5000: loss=1.987, eval_loss=2.012
Checkpoint saved: ./checkpoints/neutts-vietnamese/checkpoint-5000
...
```

---

## ⏱️ So sánh Thời gian

### On-the-fly encoding (Cũ):

```
Pre-processing: 0 giờ (không cần)
Training 3 epochs: ~30 ngày
→ Total: ~30 ngày
```

### Pre-encoded dataset (Mới):

```
Pre-encoding: ~36-40 giờ (1 lần duy nhất)
Training 3 epochs: ~3-5 ngày
→ Total: ~5-6 ngày

Nhanh gấp 5-6 lần!
```

**Nếu train nhiều lần (thử nghiệm hyperparameters):**

```
On-the-fly:
- Lần 1: 30 ngày
- Lần 2: 30 ngày
- Lần 3: 30 ngày
→ Total: 90 ngày

Pre-encoded:
- Pre-encode: 40 giờ (1 lần)
- Lần 1: 5 ngày
- Lần 2: 5 ngày
- Lần 3: 5 ngày
→ Total: ~17 ngày (nhanh gấp 5 lần!)
```

---

## 💾 Disk Space

### File sizes ước tính:

```
metadata.csv: ~200 MB
wavs/ (2.6M files): ~50-100 GB
vietnamese_dataset.pkl: ~10-20 GB

Total: ~60-120 GB
```

**💡 Tip:** Sau khi pre-encode xong, bạn có thể:
- Giữ `vietnamese_dataset.pkl` để training
- Backup `wavs/` ra ổ cứng ngoài (tiết kiệm space)
- Xóa `wavs/` nếu cần (nhưng không khuyến nghị)

---

## 🔧 Troubleshooting

### Lỗi: "CUDA out of memory" khi pre-encode

```bash
# Giải pháp 1: Dùng CPU (chậm hơn nhưng ổn định)
python prepare_vietnamese_dataset.py \
    --metadata "metadata.csv" \
    --audio_dir "wavs" \
    --output "vietnamese_dataset.pkl" \
    --device "cpu"

# Giải pháp 2: Pre-encode từng phần
# Sửa prepare_vietnamese_dataset.py để xử lý từng 100k samples
```

### Lỗi: "File too large" khi save pickle

```python
# Nếu file > 4GB, dùng protocol 4
# Sửa dòng 103 trong prepare_vietnamese_dataset.py:
pickle.dump(encoded_dataset, f, protocol=4)
```

### Pre-encoding bị gián đoạn

```bash
# Thêm checkpoint để resume
# TODO: Tôi có thể thêm tính năng này nếu bạn cần
```

---

## 📊 Workflow Hoàn chỉnh

```
┌─────────────────────────────────────────────────────────────┐
│              FAST TRAINING WORKFLOW                         │
└─────────────────────────────────────────────────────────────┘

1. Chuẩn bị data
   - metadata.csv
   - wavs/
   ↓

2. Pre-encode (1 lần duy nhất - 36-40 giờ)
   python prepare_vietnamese_dataset.py \
       --metadata "metadata.csv" \
       --audio_dir "wavs" \
       --output "vietnamese_dataset.pkl"
   ↓

3. Sửa config
   dataset_path: "vietnamese_dataset.pkl"
   per_device_train_batch_size: 4
   gradient_accumulation_steps: 2
   ↓

4. Training (3-5 ngày)
   python finetune_vietnamese.py finetune_vietnamese_config.yaml
   ↓

5. Inference
   python infer_vietnamese.py --text "..." --ref_audio "..." --ref_text "..."
```

---

## ✅ Checklist

### Pre-encoding:
- [ ] Có đủ disk space (~20GB)
- [ ] metadata.csv và wavs/ sẵn sàng
- [ ] Chạy `prepare_vietnamese_dataset.py`
- [ ] Đợi 36-40 giờ (hoặc qua đêm)
- [ ] Kiểm tra file `vietnamese_dataset.pkl` đã tạo

### Training:
- [ ] Sửa config: `dataset_path: "vietnamese_dataset.pkl"`
- [ ] Tăng batch size: `per_device_train_batch_size: 4`
- [ ] Giảm accumulation: `gradient_accumulation_steps: 2`
- [ ] Chạy training
- [ ] Đợi 3-5 ngày

### Inference:
- [ ] Checkpoint đã save
- [ ] Test với `quick_infer.py`
- [ ] So sánh các checkpoints khác nhau

---

## 🎯 Kết luận

**Pre-encode dataset = Đầu tư 1 lần, lợi ích mãi mãi!**

- ✅ Nhanh gấp 5-6 lần
- ✅ Ổn định hơn (không lo lỗi encoding giữa chừng)
- ✅ Dễ debug (data đã chuẩn)
- ✅ Tiết kiệm thời gian khi train nhiều lần

**Bắt đầu ngay:**

```bash
# Bước 1: Pre-encode
python prepare_vietnamese_dataset.py \
    --metadata "/media/huy/data1tb/thuan/dataset/metadata.csv" \
    --audio_dir "/media/huy/data1tb/thuan/dataset/wavs" \
    --output "vietnamese_dataset.pkl"

# Bước 2: Sửa config
# dataset_path: "vietnamese_dataset.pkl"

# Bước 3: Train
python finetune_vietnamese.py finetune_vietnamese_config.yaml
```

**Happy fast training!** 🚀

