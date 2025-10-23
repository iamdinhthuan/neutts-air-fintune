# NeuTTS-Air Vietnamese Finetuning Guide

Hướng dẫn đầy đủ từ training đến inference cho tiếng Việt.

## 📋 Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/iamdinhthuan/neutts-air-fintune.git
cd neutts-air-fintune

# 2. Install Python packages
pip install torch transformers datasets neucodec phonemizer librosa soundfile fire omegaconf loguru pandas

# 3. Install espeak-ng (Vietnamese phonemizer)
# Ubuntu/Debian:
sudo apt-get install espeak-ng

# macOS:
brew install espeak-ng

# Windows: Download từ https://github.com/espeak-ng/espeak-ng/releases
```

---

## 🎓 TRAINING

### Bước 1: Chuẩn bị dữ liệu

Tạo file `metadata.csv`:

```csv
audio|transcript
wavs/sample1.wav|Xin chào Việt Nam
wavs/sample2.wav|Hôm nay trời đẹp quá
wavs/sample3.wav|Tôi yêu tiếng Việt
```

Đặt audio files trong thư mục `wavs/`.

### Bước 2: Cấu hình

Sửa `finetune_vietnamese_config.yaml`:

```yaml
# Dataset
dataset_path: "metadata.csv"
audio_dir: "wavs"
max_samples: null  # null = dùng toàn bộ, hoặc số như 10000

# Training
per_device_train_batch_size: 2  # Batch size per GPU
gradient_accumulation_steps: 4   # Effective batch = 2*4 = 8
num_train_epochs: 3              # Số epochs
save_steps: 5000                 # Save checkpoint mỗi 5000 steps
eval_steps: 5000                 # Eval mỗi 5000 steps
```

**Điều chỉnh theo GPU:**

| GPU VRAM | Config |
|----------|--------|
| 8GB | `batch=1, acc=8` |
| 16GB | `batch=2, acc=4` |
| 24GB+ | `batch=4, acc=2` |

### Bước 3: Chạy training

```bash
python finetune_vietnamese.py finetune_vietnamese_config.yaml
```

**Output:**
```
Train samples: 2591598
Val samples: 13023
Batch size per device: 2
Gradient accumulation steps: 4
Effective batch size: 8
Training epochs: 3
Estimated total steps: ~242,961
```

Checkpoints sẽ được lưu trong `./checkpoints/neutts-vietnamese/checkpoint-XXXX/`.

---

## 🎤 INFERENCE

### Cách 1: Quick Test

Sửa `quick_infer.py`:

```python
CHECKPOINT_DIR = "./checkpoints/neutts-vietnamese"
REF_AUDIO = "wavs/vivoice_0.wav"
REF_TEXT = "Xin chào"
TEXT = "Hôm nay trời đẹp quá"
OUTPUT = "output_vietnamese.wav"
```

Chạy:

```bash
python quick_infer.py
```

### Cách 2: Command Line

```bash
# Tự động tìm checkpoint mới nhất
python infer_vietnamese.py \
    --text "Xin chào Việt Nam" \
    --ref_audio "wavs/vivoice_0.wav" \
    --ref_text "Xin chào" \
    --output "output.wav"

# Chỉ định checkpoint cụ thể
python infer_vietnamese.py \
    --text "Hôm nay trời đẹp quá" \
    --ref_audio "wavs/vivoice_0.wav" \
    --ref_text "Xin chào" \
    --checkpoint "./checkpoints/neutts-vietnamese/checkpoint-5000" \
    --output "output.wav"

# Tùy chỉnh sampling
python infer_vietnamese.py \
    --text "Tôi yêu Việt Nam" \
    --ref_audio "wavs/vivoice_0.wav" \
    --ref_text "Xin chào" \
    --temperature 0.8 \
    --top_k 30 \
    --output "output.wav"
```

### Tham số inference

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--text` | Text tiếng Việt cần tổng hợp | **Bắt buộc** |
| `--ref_audio` | Audio tham chiếu | **Bắt buộc** |
| `--ref_text` | Text của audio tham chiếu | **Bắt buộc** |
| `--checkpoint` | Checkpoint cụ thể | Auto (mới nhất) |
| `--output` | File output | `output.wav` |
| `--temperature` | Sampling temperature (0.5-1.5) | `1.0` |
| `--top_k` | Top-k sampling (10-100) | `50` |
| `--device` | Device (cuda/cpu) | `cuda` |

---

## 💡 Tips & Tricks

### Training

**1. Dataset size vs Training time**

| Samples | Time (3 epochs) |
|---------|-----------------|
| 10,000 | ~3 giờ |
| 100,000 | ~30 giờ |
| 1,000,000 | ~12 ngày |
| 2,600,000 | ~30 ngày |

**2. Giảm training time**

```yaml
# Test với ít data trước
max_samples: 10000
num_train_epochs: 1

# Hoặc tăng batch size (nếu GPU đủ RAM)
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
```

**3. CUDA out of memory**

```yaml
# Giảm batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

### Inference

**1. Chọn reference audio tốt**

- Độ dài: 3-10 giây
- Chất lượng: Rõ ràng, ít nhiễu
- Nội dung: Càng giống text cần tổng hợp càng tốt

**2. Điều chỉnh giọng nói**

```bash
# Giọng ổn định
--temperature 0.7 --top_k 20

# Giọng cân bằng (khuyến nghị)
--temperature 1.0 --top_k 50

# Giọng đa dạng
--temperature 1.3 --top_k 80
```

**3. So sánh checkpoints**

```bash
# Test nhiều checkpoints để tìm tốt nhất
for step in 5000 10000 15000 20000; do
    python infer_vietnamese.py \
        --text "Test" \
        --ref_audio "ref.wav" \
        --ref_text "Test" \
        --checkpoint "./checkpoints/neutts-vietnamese/checkpoint-$step" \
        --output "test_$step.wav"
done
```

---

## 🔧 Troubleshooting

### Lỗi: "No checkpoints found"

```bash
# Kiểm tra thư mục
ls -la ./checkpoints/neutts-vietnamese/

# Chỉ định checkpoint thủ công
python infer_vietnamese.py --checkpoint "./checkpoints/neutts-vietnamese/checkpoint-5000" ...
```

### Lỗi: CUDA out of memory

```yaml
# Training: Giảm batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 8

# Inference: Dùng CPU
python infer_vietnamese.py --device "cpu" ...
```

### Lỗi: "Failed to phonemize"

```bash
# Kiểm tra espeak-ng
espeak-ng --version
espeak-ng -v vi "Xin chào"

# Cài đặt lại nếu cần
sudo apt-get install --reinstall espeak-ng
```

### Audio output bị lỗi/nhiễu

- Thử giảm temperature: `--temperature 0.8`
- Thử giảm top_k: `--top_k 30`
- Dùng reference audio chất lượng cao hơn
- Thử checkpoint khác

---

## 📊 Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING WORKFLOW                        │
└─────────────────────────────────────────────────────────────┘

1. Chuẩn bị metadata.csv + wavs/
   ↓
2. Cấu hình finetune_vietnamese_config.yaml
   ↓
3. python finetune_vietnamese.py config.yaml
   ↓
4. Model tự động:
   - Load dataset từ CSV
   - Split 99.5% train / 0.5% val
   - Encode audio on-the-fly (không cần pre-processing)
   - Phonemize Vietnamese text
   - Train với gradient accumulation
   - Save checkpoints mỗi 5000 steps
   - Evaluate mỗi 5000 steps
   ↓
5. Checkpoints → ./checkpoints/neutts-vietnamese/

┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE WORKFLOW                        │
└─────────────────────────────────────────────────────────────┘

1. python infer_vietnamese.py --text "..." --ref_audio "..." --ref_text "..."
   ↓
2. Tự động tìm checkpoint mới nhất
   ↓
3. Load model + tokenizer + codec
   ↓
4. Encode reference audio → codes
   ↓
5. Phonemize Vietnamese text → IPA
   ↓
6. Generate speech codes
   ↓
7. Decode codes → audio waveform (24kHz)
   ↓
8. Save to WAV file
```

---

## 📁 Files

```
neutts-air-fintune/
├── finetune_vietnamese.py              # Training script
├── finetune_vietnamese_config.yaml     # Training config
├── infer_vietnamese.py                 # Inference script
├── quick_infer.py                      # Quick test
├── prepare_vietnamese_dataset.py       # Optional pre-encoding
├── VIETNAMESE_GUIDE.md                 # This file
├── metadata.csv                        # Your data (gitignored)
├── wavs/                               # Your audio (gitignored)
└── checkpoints/                        # Checkpoints (gitignored)
```

---

## 🎯 Quick Reference

### Training

```bash
python finetune_vietnamese.py finetune_vietnamese_config.yaml
```

### Inference

```bash
python infer_vietnamese.py --text "Xin chào" --ref_audio "ref.wav" --ref_text "Xin chào"
```

### Config

```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 4  # Effective batch = 8
num_train_epochs: 3
```

---

**Happy training!** 🎉

