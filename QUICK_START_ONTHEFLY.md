# 🚀 QUICK START: ON-THE-FLY ENCODING

## ✨ Tính năng mới

**Bạn KHÔNG CẦN encode audio trước nữa!**

Script `finetune_vietnamese.py` đã được nâng cấp để hỗ trợ **ON-THE-FLY ENCODING**:
- ✅ Tự động encode audio khi training
- ✅ Không cần chạy `prepare_vietnamese_dataset.py`
- ✅ Tiết kiệm disk space (không tạo file pkl lớn)
- ✅ Linh hoạt - dễ thêm/bớt data

---

## 📋 Yêu cầu

Chỉ cần 2 thứ:

1. **File metadata.csv** (format: `audio|transcript`)
   ```
   vivoice_0.wav|Và mọi chuyện bắt đầu từ đây
   vivoice_1.wav|Tuy nhiên điều đó không quan trọng
   ```

2. **Thư mục wavs/** chứa các file audio
   ```
   wavs/
   ├── vivoice_0.wav
   ├── vivoice_1.wav
   └── ...
   ```

---

## 🎯 Cách dùng

### Bước 1: Cấu hình

Mở `finetune_vietnamese_config.yaml` và đảm bảo:

```yaml
# Dataset - ON-THE-FLY MODE
dataset_path: "metadata.csv"
audio_dir: "wavs"
```

### Bước 2: Chạy training

```bash
python finetune_vietnamese.py finetune_vietnamese_config.yaml
```

**Xong!** Script sẽ tự động:
1. Load metadata.csv
2. Load NeuCodec model
3. Encode audio on-the-fly khi training
4. Phonemize Vietnamese text
5. Train model

---

## 🧪 Test trước khi train

Chạy script test để đảm bảo mọi thứ hoạt động:

```bash
python test_onthefly.py
```

Output mong đợi:
```
============================================================
TESTING ON-THE-FLY ENCODING
============================================================

✓ Found test file: vivoice_0.wav
  Text: Và mọi chuyện bắt đầu từ đây...

[1/3] Loading NeuCodec...
✓ NeuCodec loaded on cuda

[2/3] Loading audio from wavs/vivoice_0.wav...
✓ Audio loaded: 80000 samples, 16000 Hz

[3/3] Encoding with NeuCodec...
✓ Encoded successfully!
  Codes shape: torch.Size([500])
  Codes range: [0, 65535]
  First 10 codes: [12345, 23456, ...]

============================================================
✅ ON-THE-FLY ENCODING WORKS!
============================================================
```

---

## 📊 So sánh với cách cũ

### Cách CŨ (Pre-encode):
```bash
# Bước 1: Encode trước (mất nhiều giờ với dataset lớn)
python prepare_vietnamese_dataset.py \
    --metadata metadata.csv \
    --audio_dir wavs \
    --output vietnamese_dataset.pkl \
    --device cuda

# Bước 2: Training
python finetune_vietnamese.py finetune_vietnamese_config.yaml
```

**Vấn đề:**
- ❌ Mất thời gian encode trước
- ❌ Tốn disk (file pkl 10-20GB với 2.6M samples)
- ❌ Khó thêm/bớt data

### Cách MỚI (On-the-fly):
```bash
# Chỉ 1 bước!
python finetune_vietnamese.py finetune_vietnamese_config.yaml
```

**Ưu điểm:**
- ✅ Nhanh hơn (không cần encode trước)
- ✅ Tiết kiệm disk
- ✅ Linh hoạt hơn

---

## ⚙️ Tùy chọn nâng cao

### Dùng subset nhỏ để test

```bash
# Tạo metadata nhỏ (100 samples)
head -n 101 metadata.csv > metadata_test.csv

# Sửa config
# dataset_path: "metadata_test.csv"

# Train
python finetune_vietnamese.py finetune_vietnamese_config.yaml
```

### Chỉ định audio_dir tuyệt đối

```yaml
# finetune_vietnamese_config.yaml
dataset_path: "/path/to/metadata.csv"
audio_dir: "/absolute/path/to/wavs"
```

### Vẫn muốn dùng pre-encoded

```yaml
# finetune_vietnamese_config.yaml
dataset_path: "vietnamese_dataset.pkl"
# Không cần audio_dir
```

---

## 🐛 Troubleshooting

### Lỗi: "librosa not available"
```bash
pip install librosa
```

### Lỗi: "neucodec not available"
```bash
pip install neucodec
```

### Lỗi: "Audio file not found"
Kiểm tra:
- `audio_dir` đúng chưa?
- Tên file trong metadata.csv khớp với file trong wavs/?

### Training chậm
- On-the-fly sẽ chậm hơn pre-encoded một chút
- Nếu train nhiều lần với cùng data, nên dùng pre-encoded

---

## 💡 Tips

1. **Dataset lớn (>100k samples)**: Dùng on-the-fly để tiết kiệm disk
2. **Dataset nhỏ (<10k samples)**: Cả 2 cách đều OK
3. **Train nhiều lần**: Nên pre-encode 1 lần, dùng lại nhiều lần
4. **Thử nghiệm**: Dùng on-the-fly cho linh hoạt

---

## 📚 Tài liệu đầy đủ

Xem `TRAINING_VIETNAMESE.md` để biết thêm chi tiết!

