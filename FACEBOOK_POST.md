# Bài đăng Facebook - Vietnamese TTS Model

---

## 🎤 CHIA SẺ: MÔ HÌNH TEXT-TO-SPEECH TIẾNG VIỆT CHẤT LƯỢNG CAO

Xin chào mọi người! 👋

Mình vừa hoàn thành việc finetune mô hình **Text-to-Speech (TTS) cho tiếng Việt** và muốn chia sẻ với cộng đồng!

---

## 🔥 THÔNG TIN MÔ HÌNH:

✅ **Base Model:** NeuTTS-Air (Qwen2.5 0.5B - 552M parameters)  
✅ **Dataset:** 3000 giờ audio tiếng Việt (2.6M+ samples)  
✅ **Chất lượng:** 24kHz, giọng tự nhiên  
✅ **Tính năng:** Voice cloning, text normalization tự động  
✅ **Training time:** 3 ngày trên RTX 3090  

---

## 🎯 TÍNH NĂNG NỔI BẬT:

🎙️ **Voice Cloning** - Nhân bản giọng nói từ audio tham chiếu (3-10 giây)  
🔢 **Text Normalization** - Tự động chuẩn hóa số, ngày tháng, từ viết tắt  
⚡ **Inference nhanh** - ~0.5 giây/câu trên GPU  
🎨 **Gradio UI** - Giao diện web dễ sử dụng  
📦 **Open Source** - Code và hướng dẫn đầy đủ  

---

## 🚀 DEMO TRỰC TIẾP:

👉 **Thử ngay tại đây:** https://6795a47848e3d59592.gradio.live/

*(Chỉ cần nhập text tiếng Việt và upload audio tham chiếu là có thể tạo giọng nói!)*

---

## 💻 SOURCE CODE & HƯỚNG DẪN:

📂 **GitHub:** https://github.com/iamdinhthuan/neutts-air-fintune

**Repo bao gồm:**
- ✅ Code training đầy đủ với optimizations (nhanh gấp 10x)
- ✅ Script inference (CLI + Gradio UI)
- ✅ Hướng dẫn chi tiết (dataset, training, inference)
- ✅ Pre-encoding workflow để training nhanh
- ✅ Tích hợp ViNorm cho text normalization

---

## 📊 KẾT QUẢ:

**Training Performance:**
- Baseline: 30 ngày → **Optimized: 2.5-3 ngày** (10x nhanh hơn!)
- GPU: RTX 3090 24GB
- Dataset: 3000 giờ audio tiếng Việt

**Inference:**
- Speed: ~0.5s/câu (GPU) | ~3-5s/câu (CPU)
- Quality: 24kHz, natural prosody
- Voice cloning: Supported ✅

---

## 🎓 HƯỚNG DẪN SỬ DỤNG:

### Quick Start:

```bash
# Clone repo
git clone https://github.com/iamdinhthuan/neutts-air-fintune
cd neutts-air-fintune

# Install dependencies
pip install -r requirements.txt

# Run Gradio UI
python gradio_app.py
```

### Hoặc dùng CLI:

```bash
python infer_vietnamese.py \
    --text "Xin chào Việt Nam" \
    --ref_audio "reference.wav" \
    --ref_text "Text của audio tham chiếu" \
    --output "output.wav"
```

---

## 🛠️ TECH STACK:

- **Model:** NeuTTS-Air (Qwen2.5 0.5B)
- **Codec:** NeuCodec (discrete speech codes)
- **Phonemizer:** espeak-ng (Vietnamese)
- **Text Norm:** ViNorm (Vietnamese text normalization)
- **Framework:** HuggingFace Transformers, PyTorch
- **UI:** Gradio

---

## 📈 TRAINING OPTIMIZATIONS:

Mình đã áp dụng nhiều optimizations để training nhanh hơn:

1. ✅ **Pre-encoded dataset** - Encode audio 1 lần, dùng nhiều lần (6x faster)
2. ✅ **TF32 precision** - Tăng tốc 20% trên GPU Ampere+
3. ✅ **Fused AdamW** - Optimizer nhanh hơn 10%
4. ✅ **Dataloader optimizations** - Pin memory, prefetch
5. ✅ **Increased batch size** - GPU utilization tốt hơn

**Kết quả:** Training nhanh gấp **10-12x** so với baseline! 🚀

---

## 🎯 USE CASES:

- 📚 **Audiobook** - Tạo sách nói tự động
- 🎓 **E-learning** - Giọng đọc cho bài giảng
- ♿ **Accessibility** - Hỗ trợ người khiếm thị
- 🤖 **Virtual Assistant** - Trợ lý ảo tiếng Việt
- 🎮 **Game/App** - Tích hợp giọng nói vào ứng dụng
- 🎬 **Content Creation** - Tạo voice-over cho video

---

## ⚠️ LƯU Ý:

**Sử dụng có trách nhiệm:**
- ⚠️ Chỉ clone giọng với sự đồng ý của chủ sở hữu
- ⚠️ Không dùng cho mục đích lừa đảo, giả mạo
- ⚠️ Tôn trọng quyền riêng tư và sở hữu trí tuệ

---

## 🙏 CREDITS:

- **Neuphonic** - NeuTTS-Air base model
- **Qwen Team** - Qwen2.5 backbone
- **espeak-ng** - Vietnamese phonemizer
- **ViNorm** - Vietnamese text normalization
- **Cộng đồng AI Việt Nam** - Support và feedback

---

## 📞 LIÊN HỆ:

- **GitHub:** https://github.com/iamdinhthuan/neutts-air-fintune
- **Demo:** https://6795a47848e3d59592.gradio.live/
- **Issues:** https://github.com/iamdinhthuan/neutts-air-fintune/issues

---

## 🎉 KẾT LUẬN:

Mình rất vui được chia sẻ project này với cộng đồng! Hy vọng nó sẽ hữu ích cho các bạn đang làm về TTS, AI, hoặc các ứng dụng liên quan đến xử lý tiếng nói tiếng Việt.

**Nếu thấy hữu ích, đừng quên:**
- ⭐ Star repo trên GitHub
- 🔄 Share cho bạn bè
- 💬 Feedback và góp ý

Cảm ơn mọi người đã đọc! 🙏

---

**#AI #MachineLearning #TTS #TextToSpeech #Vietnamese #DeepLearning #NLP #VoiceCloning #OpenSource #PyTorch #HuggingFace**

---

## 📸 HÌNH ẢNH DEMO:

*(Đính kèm screenshots của Gradio UI hoặc kết quả inference)*

---

## 🎬 VIDEO DEMO:

*(Nếu có, đính kèm video demo sử dụng model)*

---

**P/S:** Model vẫn đang được cải thiện. Mọi đóng góp và feedback đều được hoan nghênh! 💪

**Thử ngay:** https://6795a47848e3d59592.gradio.live/  
**Source code:** https://github.com/iamdinhthuan/neutts-air-fintune

---

*Bài viết này được tạo để chia sẻ trong các nhóm AI/ML Việt Nam*

---
---

# PHIÊN BẢN NGẮN GỌN (Cho Facebook Post)

---

## 🎤 MÔ HÌNH TEXT-TO-SPEECH TIẾNG VIỆT - OPEN SOURCE

Xin chào mọi người! 👋

Mình vừa hoàn thành finetune mô hình **TTS tiếng Việt** trên **3000 giờ audio** và muốn chia sẻ với cộng đồng!

---

### 🔥 HIGHLIGHTS:

✅ **3000 giờ audio** tiếng Việt (2.6M+ samples)
✅ **Voice cloning** - Clone giọng từ 3-10s audio
✅ **Text normalization** - Tự động chuẩn hóa số, ngày tháng
✅ **24kHz** - Chất lượng cao, giọng tự nhiên
✅ **Open Source** - Code + hướng dẫn đầy đủ

---

### 🚀 THỬ NGAY:

👉 **Demo:** https://6795a47848e3d59592.gradio.live/
👉 **GitHub:** https://github.com/iamdinhthuan/neutts-air-fintune

---

### 💡 TÍNH NĂNG:

🎙️ Nhân bản giọng nói từ audio tham chiếu
🔢 Đọc số, ngày tháng tự động (8/2019 → "tám tháng hai năm...")
⚡ Inference nhanh (~0.5s/câu)
🎨 Giao diện Gradio dễ dùng

---

### 📊 TECH:

- **Model:** NeuTTS-Air (Qwen2.5 0.5B - 552M params)
- **Training:** 3 ngày trên RTX 3090
- **Optimizations:** 10x nhanh hơn baseline
- **Framework:** PyTorch + HuggingFace

---

### 🎯 USE CASES:

📚 Audiobook | 🎓 E-learning | ♿ Accessibility
🤖 Virtual Assistant | 🎮 Game/App | 🎬 Voice-over

---

### ⚠️ SỬ DỤNG CÓ TRÁCH NHIỆM:

- Chỉ clone giọng với sự đồng ý
- Không dùng cho lừa đảo, giả mạo
- Tôn trọng quyền riêng tư

---

**Thử ngay:** https://6795a47848e3d59592.gradio.live/
**Code:** https://github.com/iamdinhthuan/neutts-air-fintune

Nếu thấy hữu ích, đừng quên ⭐ star repo nhé! 🙏

**#AI #TTS #Vietnamese #VoiceCloning #OpenSource #MachineLearning**

---
---

# PHIÊN BẢN CỰC NGẮN (Cho comment hoặc share nhanh)

---

🎤 **Vietnamese TTS Model - Open Source**

✅ 3000h audio tiếng Việt
✅ Voice cloning
✅ 24kHz quality
✅ Text normalization

🚀 **Demo:** https://6795a47848e3d59592.gradio.live/
💻 **GitHub:** https://github.com/iamdinhthuan/neutts-air-fintune

Thử ngay! 🔥

#AI #TTS #Vietnamese #OpenSource

