---
language:
- vi
license: apache-2.0
tags:
- text-to-speech
- tts
- vietnamese
- audio
- speech-synthesis
- neutts-air
- qwen2.5
datasets:
- custom
metrics:
- wer
library_name: transformers
pipeline_tag: text-to-speech
---

# NeuTTS-Air Vietnamese TTS

Vietnamese Text-to-Speech model finetuned from [NeuTTS-Air](https://huggingface.co/neuphonic/neutts-air) on 2.6M+ Vietnamese audio samples.

## Model Description

**NeuTTS-Air Vietnamese** là mô hình Text-to-Speech (TTS) cho tiếng Việt, được finetune từ NeuTTS-Air base model trên dataset lớn 2.6M+ mẫu audio tiếng Việt.

- **Base Model:** [neuphonic/neutts-air](https://huggingface.co/neuphonic/neutts-air) (Qwen2.5 0.5B - 552M parameters)
- **Language:** Vietnamese (vi)
- **Task:** Text-to-Speech (TTS)
- **Training Data:** 2.6M+ Vietnamese audio samples
- **Audio Codec:** [NeuCodec](https://huggingface.co/neuphonic/neucodec)
- **Sample Rate:** 24kHz
- **License:** Apache 2.0

## Features

✅ **High Quality Vietnamese TTS** - Natural Vietnamese speech synthesis  
✅ **Large-scale Training** - Trained on 2.6M+ samples  
✅ **Voice Cloning** - Clone voice from reference audio  
✅ **Text Normalization** - Automatic Vietnamese text normalization with ViNorm  
✅ **Fast Inference** - Optimized for production use  
✅ **Easy to Use** - Simple API and Gradio UI  

## Quick Start

### Installation

```bash
pip install torch transformers neucodec phonemizer librosa soundfile vinorm
```

**Install espeak-ng:**

```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng

# macOS
brew install espeak-ng
```

### Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from neucodec import NeuCodec
from phonemizer.backend import EspeakBackend
from vinorm import TTSnorm
import soundfile as sf
import numpy as np

# Load model
model_id = "YOUR_USERNAME/neutts-air-vietnamese"  # Replace with your model ID
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")
model.eval()

# Load codec
codec = NeuCodec.from_pretrained("neuphonic/neucodec").to("cuda")
codec.eval()

# Initialize phonemizer
phonemizer = EspeakBackend(language='vi', preserve_punctuation=True, with_stress=True)

# Normalize and phonemize text
text = "Xin chào, đây là mô hình text to speech tiếng Việt"
text_normalized = TTSnorm(text, punc=False, unknown=True, lower=False, rule=False)
phones = phonemizer.phonemize([text_normalized])[0]

# Encode reference audio (for voice cloning)
from librosa import load as librosa_load
ref_audio_path = "reference.wav"
ref_text = "Đây là văn bản tham chiếu"
ref_text_normalized = TTSnorm(ref_text, punc=False, unknown=True, lower=False, rule=False)
ref_phones = phonemizer.phonemize([ref_text_normalized])[0]

wav, _ = librosa_load(ref_audio_path, sr=16000, mono=True)
wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    ref_codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0).cpu()

# Generate speech
codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes.tolist()])
combined_phones = ref_phones + " " + phones
chat = f"""user: Convert the text to speech:<|TEXT_PROMPT_START|>{combined_phones}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"""

input_ids = tokenizer.encode(chat, return_tensors="pt").to("cuda")
speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=2048,
        temperature=1.0,
        top_k=50,
        eos_token_id=speech_end_id,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode to audio
output_text = tokenizer.decode(output[0], skip_special_tokens=False)
# Extract speech codes and decode with codec...
# (See full implementation in repository)

# Save audio
sf.write("output.wav", audio, 24000)
```

### Using the Inference Script

For easier usage, use the provided inference script:

```bash
# Clone repository
git clone https://github.com/iamdinhthuan/neutts-air-fintune
cd neutts-air-fintune

# Install dependencies
pip install -r requirements.txt

# Run inference
python infer_vietnamese.py \
    --text "Xin chào Việt Nam" \
    --ref_audio "reference.wav" \
    --ref_text "Text của reference audio" \
    --output "output.wav" \
    --checkpoint "path/to/checkpoint"
```

### Gradio UI

```bash
python gradio_app.py
```

Then open http://localhost:7860 in your browser.

## Training Details

### Training Data

- **Dataset Size:** 2.6M+ Vietnamese audio samples
- **Audio Format:** WAV, 16kHz, mono
- **Text:** Vietnamese with diacritics
- **Train/Val Split:** 99.5% / 0.5%

### Training Configuration

- **Base Model:** neuphonic/neutts-air (Qwen2.5 0.5B)
- **Epochs:** 3
- **Batch Size:** 4 per device
- **Gradient Accumulation:** 2 steps (effective batch size: 8)
- **Learning Rate:** 4e-5
- **Optimizer:** AdamW (fused)
- **Precision:** BFloat16
- **Hardware:** NVIDIA RTX 3090 (24GB)
- **Training Time:** ~2.5-3 days

### Optimizations

- ✅ **Pre-encoded Dataset** - 6x faster training
- ✅ **TF32 Precision** - 20% speedup on Ampere GPUs
- ✅ **Fused AdamW** - 10% faster optimizer
- ✅ **Dataloader Optimizations** - Pin memory, prefetch
- ✅ **Increased Batch Size** - Better GPU utilization

**Total Speedup:** 10-12x faster than baseline (30 days → 2.5-3 days)

## Performance

### Audio Quality

- **Sample Rate:** 24kHz
- **Natural Prosody:** Yes
- **Voice Cloning:** Supported
- **Text Normalization:** Automatic (numbers, dates, abbreviations)

### Inference Speed

- **GPU (RTX 3090):** ~0.5s per sentence
- **CPU:** ~3-5s per sentence

## Limitations

- Requires reference audio for voice cloning
- Best results with clear, high-quality reference audio (3-10 seconds)
- May struggle with very long sentences (>100 words)
- Requires Vietnamese text with proper diacritics for best quality

## Ethical Considerations

⚠️ **Voice Cloning Ethics:**
- Only use reference audio with proper consent
- Do not use for impersonation or fraud
- Respect privacy and intellectual property rights

⚠️ **Potential Misuse:**
- Deepfake audio generation
- Unauthorized voice cloning
- Misinformation campaigns

**Recommended Use:**
- Accessibility tools (text-to-speech for visually impaired)
- Educational content
- Virtual assistants
- Audiobook narration (with consent)
- Language learning applications

## Citation

If you use this model, please cite:

```bibtex
@misc{neutts-air-vietnamese,
  author = {Your Name},
  title = {NeuTTS-Air Vietnamese TTS},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/YOUR_USERNAME/neutts-air-vietnamese}},
}

@misc{neutts-air,
  author = {Neuphonic},
  title = {NeuTTS-Air: Scalable TTS with Qwen2.5},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/neuphonic/neutts-air}},
}
```

## Acknowledgments

- **Base Model:** [Neuphonic](https://github.com/neuphonic) for NeuTTS-Air
- **Backbone:** [Qwen Team](https://github.com/QwenLM) for Qwen2.5
- **Codec:** [Neuphonic](https://github.com/neuphonic) for NeuCodec
- **Phonemizer:** [espeak-ng](https://github.com/espeak-ng/espeak-ng)
- **Text Normalization:** [ViNorm](https://github.com/v-nhandt21/ViNorm)

## Repository

Full training and inference code: [https://github.com/iamdinhthuan/neutts-air-fintune](https://github.com/iamdinhthuan/neutts-air-fintune)

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Contact

For questions or issues, please open an issue on [GitHub](https://github.com/iamdinhthuan/neutts-air-fintune/issues).

---

**Model Card Authors:** Your Name  
**Last Updated:** 2025-01-01

