#!/usr/bin/env python3
"""
Gradio UI cho Vietnamese TTS (NeuTTS-Air)

- Tải checkpoint gần nhất trong `./checkpoints/neutts-vietnamese` (mặc định)
- Nhập văn bản tiếng Việt, tách câu theo dấu chấm '.'
- Dùng một audio tham chiếu và ref text để giữ giọng
- Tổng hợp từng câu và ghép lại thành một file âm thanh
"""

import os
import re
import tempfile
from typing import List, Tuple

import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from infer_vietnamese import VietnameseTTS, find_latest_checkpoint

# Import ViNorm for Vietnamese text normalization
try:
    from vinorm import TTSnorm
    VINORM_AVAILABLE = True
except ImportError:
    VINORM_AVAILABLE = False
    print("Warning: vinorm not installed. Install with: pip install vinorm")
    print("Text normalization will be skipped in Gradio UI.")


def split_sentences_by_dot(text: str) -> List[str]:
    # Tách theo dấu chấm, loại bỏ khoảng trắng thừa và câu rỗng
    parts = [s.strip() for s in re.split(r"\.+", text) if s.strip()]
    return parts


_tts_cache = {}
_latest_ckpt_cache = {}
_ref_codes_cache = {}


def get_tts(checkpoint: str | None, checkpoints_dir: str, device: str) -> VietnameseTTS:
    # Resolve checkpoint path (cache latest for a checkpoints_dir)
    if checkpoint and len(checkpoint.strip()) > 0:
        checkpoint_path = checkpoint.strip()
    else:
        if checkpoints_dir in _latest_ckpt_cache:
            checkpoint_path = _latest_ckpt_cache[checkpoints_dir]
        else:
            checkpoint_path = find_latest_checkpoint(checkpoints_dir)
            _latest_ckpt_cache[checkpoints_dir] = checkpoint_path

    cache_key = (checkpoint_path, device)
    if cache_key in _tts_cache:
        return _tts_cache[cache_key]

    tts = VietnameseTTS(
        checkpoint_path=checkpoint_path,
        device=device,
        codec_device=device,
    )
    _tts_cache[cache_key] = tts
    return tts


def get_ref_codes_cached(tts: VietnameseTTS, ref_audio_file: str):
    try:
        mtime = os.path.getmtime(ref_audio_file)
    except Exception:
        mtime = 0.0
    key = (ref_audio_file, mtime)
    if key in _ref_codes_cache:
        return _ref_codes_cache[key]
    ref_codes = tts.encode_reference(ref_audio_file)
    _ref_codes_cache[key] = ref_codes
    return ref_codes


_whisper_pipe = None


def get_whisper_pipeline(device: str = "cuda"):
    global _whisper_pipe
    if _whisper_pipe is not None:
        return _whisper_pipe

    model_id = "openai/whisper-large-v3"
    torch_dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32

    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    asr_processor = AutoProcessor.from_pretrained(model_id)

    asr_model.to(device)

    _whisper_pipe = pipeline(
        task="automatic-speech-recognition",
        model=asr_model,
        tokenizer=asr_processor.tokenizer,
        feature_extractor=asr_processor.feature_extractor,
        device=0 if (device == "cuda" and torch.cuda.is_available()) else -1,
    )
    return _whisper_pipe


def transcribe_ref_audio(ref_audio_file: str, language: str = "vi", device: str = "cuda") -> str:
    if not ref_audio_file:
        raise gr.Error("Vui lòng chọn audio tham chiếu để nhận dạng")

    asr = get_whisper_pipeline(device)
    # Force language if provided
    result = asr(ref_audio_file, generate_kwargs={"language": language, "task": "transcribe"})
    text = result["text"].strip() if isinstance(result, dict) else str(result)
    if not text:
        raise gr.Error("Không nhận dạng được nội dung từ audio tham chiếu")
    return text


def synthesize_full_text(
    input_text: str,
    ref_audio_file: str,
    ref_text: str,
    temperature: float = 1.0,
    top_k: int = 50,
    checkpoint: str | None = None,
    checkpoints_dir: str = "./checkpoints/neutts-vietnamese",
    device: str = "cuda",
) -> Tuple[int, np.ndarray, str]:
    if not input_text or not input_text.strip():
        raise gr.Error("Vui lòng nhập văn bản tiếng Việt")

    if not ref_audio_file:
        raise gr.Error("Vui lòng cung cấp audio tham chiếu")

    if not ref_text or not ref_text.strip():
        # Tự động nhận dạng nếu ref_text trống
        try:
            ref_text = transcribe_ref_audio(ref_audio_file, language="vi", device=device)
        except Exception as e:
            raise gr.Error("Vui lòng nhập ref text hoặc bật nhận dạng Whisper: " + str(e))

    # Load/Cached model
    tts = get_tts(checkpoint, checkpoints_dir, device)

    # Encode ref audio 1 lần (cached theo đường dẫn + mtime)
    ref_codes = get_ref_codes_cached(tts, ref_audio_file)

    # Tách câu theo dấu chấm
    sentences = split_sentences_by_dot(input_text)
    if len(sentences) == 0:
        raise gr.Error("Không tách được câu hợp lệ từ văn bản đã nhập")

    # Tổng hợp từng câu và ghép âm thanh
    sr = tts.sample_rate
    audios: List[np.ndarray] = []

    # 0.2s khoảng nghỉ giữa các câu
    pause = np.zeros(int(0.2 * sr), dtype=np.float32)

    for sent in sentences:
        codes_str = tts.generate(
            text=sent,
            ref_codes=ref_codes,
            ref_text=ref_text,
            temperature=temperature,
            top_k=top_k,
        )
        wav = tts.decode_to_audio(codes_str).astype(np.float32)
        audios.append(wav)
        audios.append(pause)

    # Ghép lại, bỏ pause cuối
    if audios:
        audios = audios[:-1]
    full_wav = np.concatenate(audios) if audios else np.array([], dtype=np.float32)

    # Trả về audio và thông tin debug
    # Lấy checkpoint thực tế từ cache nếu có
    actual_ckpt = None
    if (checkpoint and len(checkpoint.strip()) > 0):
        actual_ckpt = checkpoint.strip()
    else:
        actual_ckpt = _latest_ckpt_cache.get(checkpoints_dir, None) or find_latest_checkpoint(checkpoints_dir)

    debug_info = f"Câu: {len(sentences)} | Checkpoint: {actual_ckpt} | Thiết bị: {device}"
    return (sr, full_wav), debug_info


with gr.Blocks(title="NeuTTS-Air Vietnamese TTS") as demo:
    gr.Markdown("""
    # NeuTTS-Air Vietnamese TTS
    Nhập văn bản tiếng Việt, ứng dụng sẽ tách câu theo dấu chấm '.' và tổng hợp giọng dựa trên audio tham chiếu.
    """)

    with gr.Row():
        input_text = gr.Textbox(
            label="Văn bản tiếng Việt",
            lines=6,
            placeholder="Nhập văn bản. Ví dụ: Xin chào. Hôm nay trời đẹp.",
        )

    with gr.Row():
        ref_audio = gr.Audio(label="Audio tham chiếu (wav)", type="filepath")
        ref_text = gr.Textbox(label="Ref text của audio tham chiếu", lines=2)

    with gr.Accordion("Tùy chọn nâng cao", open=False):
        with gr.Row():
            temperature = gr.Slider(0.1, 1.5, value=1.0, step=0.05, label="Temperature")
            top_k = gr.Slider(10, 200, value=50, step=1, label="Top-K")
        with gr.Row():
            checkpoint = gr.Textbox(label="Checkpoint (để trống để tự tìm mới nhất)", value="")
            checkpoints_dir = gr.Textbox(label="Thư mục checkpoints", value="./checkpoints/neutts-vietnamese")
            device = gr.Dropdown(choices=["cuda", "cpu"], value="cuda", label="Thiết bị")

    with gr.Row():
        asr_lang = gr.Dropdown(choices=["vi", "en", "auto"], value="vi", label="Ngôn ngữ ASR (Whisper)")
        asr_btn = gr.Button("Nhận dạng ref text (Whisper v3 Large)")

    with gr.Row():
        btn = gr.Button("Tổng hợp")

    with gr.Row():
        audio_out = gr.Audio(label="Kết quả", type="numpy")
    debug = gr.Textbox(label="Thông tin", interactive=False)

    btn.click(
        fn=synthesize_full_text,
        inputs=[input_text, ref_audio, ref_text, temperature, top_k, checkpoint, checkpoints_dir, device],
        outputs=[audio_out, debug],
    )

    asr_btn.click(
        fn=lambda audio_path, lang, dev: transcribe_ref_audio(audio_path, language=lang if lang != "auto" else "vi", device=dev),
        inputs=[ref_audio, asr_lang, device],
        outputs=ref_text,
    )


if __name__ == "__main__":
    # Chạy demo
    demo.queue().launch(share=True)


