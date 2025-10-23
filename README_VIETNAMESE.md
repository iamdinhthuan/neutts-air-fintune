# NeuTTS-Air Vietnamese Finetuning

Vietnamese TTS finetuning cho [NeuTTS-Air](https://github.com/neuphonic/neutts-air) - một mô hình text-to-speech hiện đại dựa trên Qwen2.5 0.5B.

## 🎯 Tính năng

✅ **On-the-fly encoding** - Không cần pre-processing, encode audio trong quá trình training  
✅ **Vietnamese phonemizer** - Hỗ trợ tiếng Việt với espeak-ng  
✅ **Gradient accumulation** - Train với batch size lớn trên GPU nhỏ  
✅ **Train/Val split** - Tự động chia dataset 99.5%/0.5%  
✅ **Auto checkpoint detection** - Tự động tìm checkpoint mới nhất khi inference  
✅ **Large dataset support** - Hỗ trợ 2.6M+ samples  

## 📋 Yêu cầu

### Phần mềm

```bash
# Python 3.10+
python --version

# espeak-ng (cho Vietnamese phonemizer)
# Ubuntu/Debian:
sudo apt-get install espeak-ng

# macOS:
brew install espeak-ng

# Windows:
# Download từ: https://github.com/espeak-ng/espeak-ng/releases
```

### Python packages

```bash
pip install torch transformers datasets
pip install neucodec phonemizer librosa soundfile
pip install fire omegaconf loguru pandas
```

## 🚀 Quick Start

### 1. Chuẩn bị dữ liệu

Tạo file `metadata.csv` với format:

```csv
audio|transcript
wavs/sample1.wav|Xin chào Việt Nam
wavs/sample2.wav|Hôm nay trời đẹp quá
wavs/sample3.wav|Tôi yêu tiếng Việt
```

Đặt audio files trong thư mục `wavs/`.

### 2. Cấu hình training

Sửa `finetune_vietnamese_config.yaml`:

```yaml
# Dataset
dataset_path: "metadata.csv"
audio_dir: "wavs"
max_samples: null  # null = dùng toàn bộ dataset

# Training
per_device_train_batch_size: 2
gradient_accumulation_steps: 4  # Effective batch = 2 * 4 = 8
num_train_epochs: 3
save_steps: 5000
eval_steps: 5000
```

### 3. Training

```bash
python finetune_vietnamese.py finetune_vietnamese_config.yaml
```

### 4. Inference

```bash
# Quick test
python quick_infer.py

# Hoặc command line
python infer_vietnamese.py \
    --text "Xin chào Việt Nam" \
    --ref_audio "wavs/sample.wav" \
    --ref_text "Xin chào" \
    --output "output.wav"
```

## 📚 Tài liệu

- **[TRAINING_VIETNAMESE.md](TRAINING_VIETNAMESE.md)** - Hướng dẫn training chi tiết
- **[INFERENCE_VIETNAMESE.md](INFERENCE_VIETNAMESE.md)** - Hướng dẫn inference chi tiết

## 📁 Cấu trúc files

```
neutts-air-fintune/
├── finetune_vietnamese.py              # Main training script
├── finetune_vietnamese_config.yaml     # Training config
├── prepare_vietnamese_dataset.py       # Optional pre-encoding script
├── infer_vietnamese.py                 # Full inference script
├── quick_infer.py                      # Quick inference script
├── TRAINING_VIETNAMESE.md              # Training guide
├── INFERENCE_VIETNAMESE.md             # Inference guide
├── metadata.csv                        # Your dataset (gitignored)
├── wavs/                               # Your audio files (gitignored)
└── checkpoints/                        # Training checkpoints (gitignored)
```

## ⚙️ Cấu hình

### Training Parameters

| Parameter | Mô tả | Mặc định |
|-----------|-------|----------|
| `per_device_train_batch_size` | Batch size per GPU | 2 |
| `gradient_accumulation_steps` | Gradient accumulation | 4 |
| `num_train_epochs` | Số epochs | 3 |
| `lr` | Learning rate | 4e-5 |
| `save_steps` | Save checkpoint mỗi N steps | 5000 |
| `eval_steps` | Evaluate mỗi N steps | 5000 |

### Effective Batch Size

```
Effective batch = per_device_train_batch_size × gradient_accumulation_steps
```

Ví dụ:
- `batch=2, acc=4` → Effective batch = 8
- `batch=1, acc=8` → Effective batch = 8 (ít RAM hơn)
- `batch=4, acc=2` → Effective batch = 8 (nhanh hơn)

## 💡 Tips

### GPU Memory

| GPU VRAM | Recommended Config |
|----------|-------------------|
| 8GB | `batch=1, acc=8` |
| 16GB | `batch=2, acc=4` |
| 24GB+ | `batch=4, acc=2` |

### Dataset Size

| Samples | Training Time (3 epochs) |
|---------|-------------------------|
| 10,000 | ~3 giờ |
| 100,000 | ~30 giờ |
| 1,000,000 | ~12 ngày |
| 2,600,000 | ~30 ngày |

### Quality Tips

1. **Audio chất lượng cao**
   - Sample rate: 16kHz hoặc cao hơn
   - Format: WAV, MP3
   - Ít nhiễu, rõ ràng

2. **Transcript chính xác**
   - Khớp 100% với audio
   - Dấu câu đúng
   - Không có lỗi chính tả

3. **Reference audio tốt**
   - Độ dài: 3-10 giây
   - Giọng rõ ràng
   - Nội dung giống text cần tổng hợp

## 🔧 Troubleshooting

### CUDA out of memory

```yaml
# Giảm batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

### Training quá chậm

```yaml
# Tăng batch size (nếu GPU đủ RAM)
per_device_train_batch_size: 4
gradient_accumulation_steps: 2

# Hoặc giảm dataset
max_samples: 100000
```

### espeak-ng error

```bash
# Kiểm tra cài đặt
espeak-ng --version

# Test Vietnamese
espeak-ng -v vi "Xin chào"
```

## 📊 Workflow

```
1. Chuẩn bị metadata.csv + wavs/
   ↓
2. Cấu hình finetune_vietnamese_config.yaml
   ↓
3. Chạy training: python finetune_vietnamese.py config.yaml
   ↓
4. Model tự động:
   - Encode audio on-the-fly
   - Phonemize Vietnamese text
   - Train với gradient accumulation
   - Save checkpoints mỗi 5000 steps
   - Evaluate mỗi 5000 steps
   ↓
5. Inference: python infer_vietnamese.py --text "..."
```

## 🎓 Ví dụ

### Training với 10k samples

```yaml
# config.yaml
max_samples: 10000
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
num_train_epochs: 3
```

```bash
python finetune_vietnamese.py finetune_vietnamese_config.yaml
```

### Inference với checkpoint cụ thể

```bash
python infer_vietnamese.py \
    --text "Chào buổi sáng" \
    --ref_audio "wavs/sample.wav" \
    --ref_text "Xin chào" \
    --checkpoint "./checkpoints/neutts-vietnamese/checkpoint-5000" \
    --output "morning.wav"
```

## 🤝 Contributing

Fork repo này và tạo pull request nếu bạn có cải tiến!

## 📄 License

Apache 2.0 (giống NeuTTS-Air gốc)

## 🙏 Credits

- **NeuTTS-Air**: [neuphonic/neutts-air](https://github.com/neuphonic/neutts-air)
- **NeuCodec**: [neuphonic/neucodec](https://github.com/neuphonic/neucodec)
- **Qwen2.5**: [Qwen/Qwen2.5](https://github.com/QwenLM/Qwen2.5)

## 📞 Support

Nếu gặp vấn đề, tạo issue trên GitHub!

---

**Happy training!** 🎉

