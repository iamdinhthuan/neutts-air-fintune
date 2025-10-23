# 🇻🇳 HƯỚNG DẪN FINETUNE NEUTTS-AIR CHO TIẾNG VIỆT

## 📋 MỤC LỤC
1. [Giới thiệu](#giới-thiệu)
2. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
3. [Cài đặt môi trường](#cài-đặt-môi-trường)
4. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
5. [Encode audio](#encode-audio)
6. [Training model](#training-model)
7. [Sử dụng model đã train](#sử-dụng-model-đã-train)
8. [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)

---

## 🎯 GIỚI THIỆU

NeuTTS-Air là một Text-to-Speech model dựa trên Qwen2.5 0.5B. Model này học cách chuyển đổi text (dạng phoneme) thành speech codes (audio đã được encode bởi NeuCodec).

**Quy trình hoạt động:**
```
Text → Phonemizer (espeak) → Phonemes
Audio WAV → NeuCodec Encoder → Speech Codes
→ Training: Phonemes → Speech Codes
```

---

## 💻 YÊU CẦU HỆ THỐNG

### Phần cứng
- **GPU**: NVIDIA GPU với CUDA (khuyến nghị >= 8GB VRAM)
- **RAM**: >= 16GB
- **Disk**: >= 20GB trống

### Phần mềm
- **Python**: 3.10 hoặc 3.11 (khuyến nghị 3.10)
- **CUDA**: 11.8 hoặc 12.1
- **Git**: Để clone repository

---

## 🔧 CÀI ĐẶT MÔI TRƯỜNG

### Bước 1: Tạo môi trường ảo

```bash
# Tạo môi trường mới
python -m venv venv

# Kích hoạt môi trường
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Bước 2: Cài đặt PyTorch

**Quan trọng**: Cài PyTorch trước tiên!

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (không khuyến nghị cho training)
pip install torch torchvision torchaudio
```

### Bước 3: Cài đặt dependencies cơ bản

```bash
pip install transformers==4.44.2 datasets accelerate
pip install librosa phonemizer tqdm omegaconf loguru fire
```

### Bước 4: Cài đặt NeuCodec

**⚠️ LƯU Ý VỀ DEPENDENCY CONFLICTS:**

Trên Windows, có thể gặp conflict giữa `torchao`, `triton`, và `transformers`. Có 2 cách xử lý:

#### Cách 1: Cài neucodec và bỏ qua dependency errors (Khuyến nghị)

```bash
pip install neucodec --no-deps
pip install einops  # dependency cần thiết của neucodec
```

#### Cách 2: Sử dụng pre-encoded data

Nếu cách 1 không work, bạn có thể:
1. Encode audio trên máy Linux/Colab
2. Copy file `.pkl` về máy Windows để train

### Bước 5: Cài đặt espeak-ng (cho phonemizer)

**Windows:**
1. Download từ: https://github.com/espeak-ng/espeak-ng/releases
2. Cài đặt và thêm vào PATH
3. Kiểm tra: `espeak-ng --version`

**Linux:**
```bash
sudo apt-get install espeak-ng
```

**Mac:**
```bash
brew install espeak-ng
```

### Bước 6: Kiểm tra cài đặt

```python
# Test PyTorch + CUDA
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Test phonemizer
from phonemizer.backend import EspeakBackend
backend = EspeakBackend(language='vi')
print(backend.phonemize(['Xin chào']))

# Test neucodec (nếu đã cài)
try:
    from neucodec import NeuCodec
    print("NeuCodec OK")
except Exception as e:
    print(f"NeuCodec error: {e}")
```

---

## 📁 CHUẨN BỊ DỮ LIỆU

### Format dữ liệu

Bạn cần 2 thứ:
1. **File metadata.csv** với format:
   ```
   audio|transcript
   audio1.wav|Câu tiếng Việt thứ nhất
   audio2.wav|Câu tiếng Việt thứ hai
   ```

2. **Thư mục chứa audio** (ví dụ: `wavs/`)
   - Format: WAV, 16kHz, mono
   - Độ dài: 1-10 giây mỗi file
   - Chất lượng: Càng rõ càng tốt

### Kiểm tra dữ liệu

```python
import pandas as pd
import librosa

# Đọc metadata
df = pd.read_csv('metadata.csv', sep='|', names=['audio', 'transcript'])
print(f"Tổng số samples: {len(df)}")
print(df.head())

# Kiểm tra một file audio
audio_path = f"wavs/{df.iloc[0]['audio']}"
y, sr = librosa.load(audio_path, sr=16000)
print(f"Sample rate: {sr}, Duration: {len(y)/sr:.2f}s")
```

---

## 🎵 ENCODE AUDIO (TÙY CHỌN)

**⚡ MỚI: Bạn KHÔNG CẦN encode trước nữa!**

Script training đã hỗ trợ **ON-THE-FLY ENCODING** - tự động encode audio khi training!

### Cách 1: ON-THE-FLY Encoding (Khuyến nghị) ⚡

**Ưu điểm:**
- ✅ Không cần chạy script encode trước
- ✅ Tiết kiệm disk (không tạo file pkl lớn)
- ✅ Linh hoạt - dễ thêm/bớt data
- ✅ Phù hợp với dataset lớn

**Cách dùng:**
```yaml
# finetune_vietnamese_config.yaml
dataset_path: "metadata.csv"
audio_dir: "wavs"
```

Chỉ cần có `metadata.csv` và thư mục `wavs/` là đủ!

### Cách 2: Pre-encode (Tùy chọn)

**Ưu điểm:**
- ✅ Training nhanh hơn (không cần encode mỗi epoch)
- ✅ Phù hợp nếu train nhiều lần với cùng data

**Bước 1: Encode audio**
```bash
python prepare_vietnamese_dataset.py \
    --metadata metadata.csv \
    --audio_dir wavs \
    --output vietnamese_dataset.pkl \
    --device cuda  # hoặc 'cpu' nếu không có GPU
```

**Bước 2: Cấu hình**
```yaml
# finetune_vietnamese_config.yaml
dataset_path: "vietnamese_dataset.pkl"
# Không cần audio_dir
```

### So sánh 2 cách:

| Tiêu chí | On-the-fly | Pre-encode |
|----------|------------|------------|
| Disk space | ✅ Ít | ❌ Nhiều (10-20GB) |
| Setup time | ✅ Nhanh | ❌ Chậm (cần encode trước) |
| Training speed | ⚠️ Hơi chậm | ✅ Nhanh hơn |
| Linh hoạt | ✅ Cao | ⚠️ Thấp |
| **Khuyến nghị** | **✅ Dùng cách này** | Chỉ khi train nhiều lần |

---

## 🚀 TRAINING MODEL

### Cấu hình training

File `finetune_vietnamese_config.yaml`:

```yaml
# Model checkpoint
restore_from: "neuphonic/neutts-air"

# Dataset - ON-THE-FLY MODE (Khuyến nghị)
dataset_path: "metadata.csv"
audio_dir: "wavs"

# Hoặc dùng pre-encoded:
# dataset_path: "vietnamese_dataset.pkl"

# Training hyperparameters
lr: 0.00004
max_steps: 1000  # Tăng lên 2000-5000 cho dataset lớn
per_device_train_batch_size: 1
warmup_ratio: 0.05

# Logging & Saving
save_root: "./checkpoints"
run_name: "neutts-vietnamese"
logging_steps: 10
save_steps: 100
```

### Chạy training

**Với on-the-fly encoding:**
```bash
# Đảm bảo có metadata.csv và thư mục wavs/
python finetune_vietnamese.py finetune_vietnamese_config.yaml
```

**Với pre-encoded data:**
```bash
# Đảm bảo đã chạy prepare_vietnamese_dataset.py trước
python finetune_vietnamese.py finetune_vietnamese_config.yaml
```

### Theo dõi training

Training sẽ in ra:
- Loss mỗi 10 steps
- Checkpoint được lưu mỗi 100 steps vào `./checkpoints/neutts-vietnamese/`

**Thời gian ước tính:**
- 11 samples (test): ~5-10 phút
- 1000 samples: ~2-4 giờ (GPU)
- 10000 samples: ~1-2 ngày (GPU)

---

## 🎤 SỬ DỤNG MODEL ĐÃ TRAIN

### Load model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from phonemizer.backend import EspeakBackend
import torch

# Load model đã finetune
model_path = "./checkpoints/neutts-vietnamese/checkpoint-1000"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Phonemizer
g2p = EspeakBackend(language='vi', preserve_punctuation=True, with_stress=True)

# Chuyển sang eval mode
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

### Sinh audio

```python
def text_to_speech(text):
    # Text → Phonemes
    phonemes = g2p.phonemize([text])[0]
    
    # Tạo prompt (theo format training)
    prompt = f"<|SPEECH_GENERATION_START|><|TEXT_PROMPT_START|>{phonemes}<|TEXT_PROMPT_END|><|SPEECH_START|>"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )
    
    # Decode codes
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract speech codes (cần parse từ generated_text)
    # TODO: Implement code extraction và decode với NeuCodec
    
    return generated_text

# Test
result = text_to_speech("Xin chào, đây là bài test tiếng Việt")
print(result)
```

---

## ⚠️ XỬ LÝ LỖI THƯỜNG GẶP

### 1. ImportError: cannot import name 'AttrsDescriptor' from 'triton'

**Nguyên nhân**: Conflict giữa torchao và triton trên Windows

**Giải pháp**:
```bash
# Cài neucodec không dependencies
pip uninstall torchao triton -y
pip install neucodec --no-deps
pip install einops
```

### 2. ModuleNotFoundError: No module named 'torchao'

**Nguyên nhân**: torchtune cần torchao

**Giải pháp**: Encode trên Linux/Colab, copy file `.pkl` về

### 3. espeak-ng not found

**Giải pháp Windows**:
1. Download: https://github.com/espeak-ng/espeak-ng/releases
2. Cài đặt vào `C:\Program Files\eSpeak NG`
3. Thêm vào PATH: `C:\Program Files\eSpeak NG`
4. Restart terminal

### 4. CUDA out of memory

**Giải pháp**:
```yaml
# Giảm batch size trong config
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # Tăng để bù batch size nhỏ
```

### 5. Loss không giảm

**Kiểm tra**:
- Dataset có đủ lớn? (>= 100 samples)
- Learning rate có phù hợp? (thử 0.00002 - 0.0001)
- Audio quality có tốt?
- Phonemizer có hoạt động đúng với tiếng Việt?

---

## 📊 TIPS ĐỂ CẢI THIỆN CHẤT LƯỢNG

### 1. Dữ liệu
- **Số lượng**: >= 1000 samples cho kết quả tốt
- **Chất lượng**: Audio rõ ràng, ít noise
- **Đa dạng**: Nhiều giọng, nhiều ngữ cảnh khác nhau
- **Độ dài**: 2-8 giây mỗi file là tối ưu

### 2. Training
- **Warmup**: Dùng warmup_steps để model ổn định
- **Learning rate**: Bắt đầu với 0.00004, điều chỉnh nếu cần
- **Steps**: Train đủ lâu (1000-5000 steps)
- **Validation**: Tách 10% data để validate

### 3. Inference
- **Temperature**: 0.7-0.9 cho tự nhiên hơn
- **Top-p**: 0.85-0.95
- **Max tokens**: Điều chỉnh theo độ dài câu

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề:
1. Kiểm tra lại từng bước trong hướng dẫn
2. Xem phần "Xử lý lỗi thường gặp"
3. Kiểm tra log chi tiết khi chạy script

---

## 📝 CHANGELOG

- **v1.0**: Hướng dẫn ban đầu cho tiếng Việt
- Hỗ trợ Windows/Linux/Mac
- Xử lý dependency conflicts
- Test với 11 samples

---

**Chúc bạn training thành công! 🎉**

