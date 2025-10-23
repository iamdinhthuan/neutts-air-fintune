# Vietnamese TTS Inference Guide

Hướng dẫn sử dụng model NeuTTS-Air đã finetune cho tiếng Việt.

## 📋 Yêu cầu

Model đã được finetune và có checkpoint trong thư mục `./checkpoints/neutts-vietnamese/`.

## 🚀 Cách 1: Quick Inference (Đơn giản nhất)

### Bước 1: Sửa config trong `quick_infer.py`

```python
CHECKPOINT_DIR = "./checkpoints/neutts-vietnamese"  # Thư mục chứa checkpoints
REF_AUDIO = "wavs/vivoice_0.wav"                    # Audio tham chiếu
REF_TEXT = "Xin chào"                                # Text của audio tham chiếu
TEXT = "Hôm nay trời đẹp quá"                       # Text cần tổng hợp
OUTPUT = "output_vietnamese.wav"                     # File output
```

### Bước 2: Chạy

```bash
python quick_infer.py
```

**Output:**
- File audio: `output_vietnamese.wav`
- Sample rate: 24kHz

---

## 🎯 Cách 2: Command Line (Linh hoạt hơn)

### Tự động tìm checkpoint mới nhất

```bash
python infer_vietnamese.py \
    --text "Xin chào Việt Nam" \
    --ref_audio "wavs/vivoice_0.wav" \
    --ref_text "Xin chào" \
    --output "output.wav"
```

### Chỉ định checkpoint cụ thể

```bash
python infer_vietnamese.py \
    --text "Hôm nay trời đẹp quá" \
    --ref_audio "wavs/vivoice_0.wav" \
    --ref_text "Xin chào" \
    --checkpoint "./checkpoints/neutts-vietnamese/checkpoint-5000" \
    --output "output.wav"
```

### Tùy chỉnh sampling

```bash
python infer_vietnamese.py \
    --text "Tôi yêu Việt Nam" \
    --ref_audio "wavs/vivoice_0.wav" \
    --ref_text "Xin chào" \
    --temperature 0.8 \
    --top_k 30 \
    --output "output.wav"
```

---

## ⚙️ Tham số

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--text` | Text tiếng Việt cần tổng hợp | **Bắt buộc** |
| `--ref_audio` | Đường dẫn audio tham chiếu | **Bắt buộc** |
| `--ref_text` | Text của audio tham chiếu | **Bắt buộc** |
| `--checkpoint` | Đường dẫn checkpoint cụ thể | Auto (mới nhất) |
| `--checkpoints_dir` | Thư mục chứa checkpoints | `./checkpoints/neutts-vietnamese` |
| `--output` | File output | `output.wav` |
| `--device` | Device (cuda/cpu) | `cuda` |
| `--temperature` | Sampling temperature (0.1-2.0) | `1.0` |
| `--top_k` | Top-k sampling | `50` |

---

## 📝 Ví dụ

### Ví dụ 1: Tổng hợp câu đơn giản

```bash
python infer_vietnamese.py \
    --text "Chào buổi sáng" \
    --ref_audio "wavs/vivoice_0.wav" \
    --ref_text "Xin chào" \
    --output "morning.wav"
```

### Ví dụ 2: Tổng hợp câu dài

```bash
python infer_vietnamese.py \
    --text "Việt Nam là một đất nước xinh đẹp với lịch sử lâu đời và văn hóa phong phú" \
    --ref_audio "wavs/vivoice_0.wav" \
    --ref_text "Xin chào" \
    --output "long_sentence.wav"
```

### Ví dụ 3: Dùng CPU (không có GPU)

```bash
python infer_vietnamese.py \
    --text "Tôi yêu Việt Nam" \
    --ref_audio "wavs/vivoice_0.wav" \
    --ref_text "Xin chào" \
    --device "cpu" \
    --output "output.wav"
```

---

## 🎨 Điều chỉnh giọng nói

### Temperature (Nhiệt độ)

- **Thấp (0.5-0.8)**: Giọng ổn định, ít biến đổi
- **Trung bình (0.9-1.1)**: Cân bằng (khuyến nghị)
- **Cao (1.2-1.5)**: Giọng đa dạng, nhiều biến đổi

```bash
# Giọng ổn định
python infer_vietnamese.py --text "..." --temperature 0.7

# Giọng đa dạng
python infer_vietnamese.py --text "..." --temperature 1.3
```

### Top-k

- **Thấp (10-30)**: Chọn từ ít token nhất → ổn định
- **Trung bình (40-60)**: Cân bằng (khuyến nghị)
- **Cao (70-100)**: Chọn từ nhiều token → đa dạng

```bash
# Ổn định
python infer_vietnamese.py --text "..." --top_k 20

# Đa dạng
python infer_vietnamese.py --text "..." --top_k 80
```

---

## 🔍 Chọn Reference Audio

Reference audio ảnh hưởng đến:
- **Giọng điệu**: Cao/thấp, nhanh/chậm
- **Phong cách**: Trang trọng/thân mật
- **Chất lượng**: Rõ ràng/nhiễu

**Khuyến nghị:**
- Dùng audio **sạch**, **rõ ràng**
- Độ dài: **3-10 giây**
- Nội dung: Càng giống text cần tổng hợp càng tốt

---

## 📊 Workflow

```
1. Load checkpoint mới nhất
   ↓
2. Load model + tokenizer + codec
   ↓
3. Encode reference audio → codes
   ↓
4. Phonemize text (Vietnamese → IPA)
   ↓
5. Generate speech codes
   ↓
6. Decode codes → audio waveform
   ↓
7. Save to WAV file (24kHz)
```

---

## 🐛 Troubleshooting

### Lỗi: "No checkpoints found"

```bash
# Kiểm tra thư mục checkpoints
ls -la ./checkpoints/neutts-vietnamese/

# Chỉ định checkpoint thủ công
python infer_vietnamese.py --checkpoint "./checkpoints/neutts-vietnamese/checkpoint-5000" ...
```

### Lỗi: CUDA out of memory

```bash
# Dùng CPU
python infer_vietnamese.py --device "cpu" ...
```

### Lỗi: "Failed to phonemize"

- Kiểm tra espeak-ng đã cài đặt:
  ```bash
  espeak-ng --version
  ```
- Cài đặt nếu chưa có:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install espeak-ng
  
  # macOS
  brew install espeak-ng
  ```

### Audio output bị lỗi/nhiễu

- Thử giảm temperature: `--temperature 0.8`
- Thử giảm top_k: `--top_k 30`
- Dùng reference audio chất lượng cao hơn

---

## 💡 Tips

1. **Reference audio tốt = Output tốt**
   - Dùng audio sạch, rõ ràng
   - Tránh audio có nhiễu, echo

2. **Text reference nên khớp với audio**
   - Nếu audio nói "Xin chào", ref_text phải là "Xin chào"
   - Không khớp → giọng có thể bị lỗi

3. **Thử nghiệm với temperature/top_k**
   - Mỗi checkpoint có thể cần tham số khác nhau
   - Thử vài giá trị để tìm tốt nhất

4. **Checkpoint càng về sau càng tốt**
   - Checkpoint-5000 thường tốt hơn checkpoint-1000
   - Nhưng cũng có thể overfit → test nhiều checkpoint

---

## 📞 Support

Nếu gặp vấn đề, kiểm tra:
1. Checkpoint có tồn tại không
2. Reference audio có đúng format không (WAV/MP3)
3. espeak-ng đã cài đặt chưa
4. GPU/CUDA có hoạt động không

---

**Chúc bạn tổng hợp giọng nói thành công!** 🎉

