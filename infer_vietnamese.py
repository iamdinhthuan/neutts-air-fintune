"""
Vietnamese TTS Inference Script for NeuTTS-Air
Generates speech from Vietnamese text using finetuned checkpoint.
"""

import os
import glob
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from neucodec import NeuCodec
from phonemizer.backend import EspeakBackend
from huggingface_hub import snapshot_download
import re
import argparse
from typing import Dict, Tuple

# Import Soe-ViNorm for Vietnamese text normalization
try:
    from soe_vinorm import SoeNormalizer
    VINORM_AVAILABLE = True
except ImportError:
    VINORM_AVAILABLE = False
    print("Warning: soe_vinorm not installed. Install with: pip install soe-vinorm")
    print("Text normalization will be skipped.")


DEFAULT_HF_REPO = "dinhthuan/neutts-air-vi"
DEFAULT_HF_CACHE_DIR = os.path.join("checkpoints", "hf-cache")


class VietnameseTTS:
    """Vietnamese TTS using finetuned NeuTTS-Air model."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        codec_device: str = "cuda",
    ):
        """
        Initialize Vietnamese TTS.
        
        Args:
            checkpoint_path: Path to finetuned checkpoint directory
            device: Device for backbone model ('cuda' or 'cpu')
            codec_device: Device for codec ('cuda' or 'cpu')
        """
        self.device = device
        self.codec_device = codec_device
        self.sample_rate = 24000
        self.max_context = 2048
        
        print("=" * 60)
        print("LOADING VIETNAMESE TTS MODEL")
        print("=" * 60)
        
        # Load tokenizer
        print(f"\n[1/3] Loading tokenizer from {checkpoint_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print("✓ Tokenizer loaded")
        
        # Load finetuned model
        print(f"\n[2/3] Loading finetuned model from {checkpoint_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        print(f"✓ Model loaded on {device}")
        
        # Load NeuCodec for decoding
        print(f"\n[3/3] Loading NeuCodec decoder...")
        self.codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(codec_device)
        self.codec.eval()
        print(f"✓ NeuCodec loaded on {codec_device}")
        
        # Load Vietnamese phonemizer
        print(f"\n[4/5] Loading Vietnamese phonemizer...")
        self.phonemizer = EspeakBackend(
            language='vi',
            preserve_punctuation=True,
            with_stress=True
        )
        print("✓ Phonemizer loaded")

        # Initialize ViNorm for text normalization
        print(f"\n[5/5] Initializing Vietnamese text normalizer...")
        if VINORM_AVAILABLE:
            self.normalizer = SoeNormalizer()
            print("✓ Soe ViNorm loaded (text will be normalized)")
        else:
            self.normalizer = None
            print("⚠ ViNorm not available (text normalization skipped)")

        print("\n" + "=" * 60)
        print("✅ MODEL READY FOR INFERENCE")
        print("=" * 60)
    
    def encode_reference(self, ref_audio_path: str):
        """
        Encode reference audio to speech codes.
        
        Args:
            ref_audio_path: Path to reference audio file
            
        Returns:
            torch.Tensor: Encoded speech codes
        """
        from librosa import load as librosa_load
        
        print(f"\nEncoding reference audio: {ref_audio_path}")
        wav, _ = librosa_load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor)
        
        ref_codes = ref_codes.squeeze(0).squeeze(0).cpu()
        print(f"✓ Encoded {len(ref_codes)} codes")
        return ref_codes
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Vietnamese text using ViNorm.

        Args:
            text: Raw Vietnamese text

        Returns:
            str: Normalized text
        """
        if self.normalizer is not None:
            normalized = self.normalizer.normalize(text)
            return normalized
        else:
            # No normalization available
            return text

    def phonemize(self, text: str) -> str:
        """
        Convert Vietnamese text to phonemes.

        Args:
            text: Vietnamese text

        Returns:
            str: Phonemized text
        """
        phones = self.phonemizer.phonemize([text])
        if not phones or not phones[0]:
            raise ValueError(f"Failed to phonemize text: {text}")

        phones = phones[0].split()
        phones = ' '.join(phones)
        return phones
    
    def generate(
        self,
        text: str,
        ref_codes: torch.Tensor,
        ref_text: str,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str:
        """
        Generate speech codes from text.
        
        Args:
            text: Vietnamese text to synthesize
            ref_codes: Reference audio codes
            ref_text: Reference text (Vietnamese)
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            str: Generated speech codes as string
        """
        # Normalize texts first
        print(f"\nNormalizing text...")
        ref_text_normalized = self.normalize_text(ref_text)
        text_normalized = self.normalize_text(text)
        if self.normalizer is not None:
            print(f"  Original: {text[:50]}...")
            print(f"  Normalized: {text_normalized[:50]}...")

        # Phonemize texts
        print(f"\nPhonemizing text...")
        ref_phones = self.phonemize(ref_text_normalized)
        input_phones = self.phonemize(text_normalized)
        print(f"  Ref: {ref_phones[:50]}...")
        print(f"  Input: {input_phones[:50]}...")
        
        # Create prompt (same format as training)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes.tolist()])
        combined_phones = ref_phones + " " + input_phones
        
        chat = f"""user: Convert the text to speech:<|TEXT_PROMPT_START|>{combined_phones}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"""
        
        # Tokenize
        input_ids = self.tokenizer.encode(chat, return_tensors="pt").to(self.device)
        print(f"\nGenerating speech codes...")
        print(f"  Input length: {input_ids.shape[1]} tokens")
        
        # Generate
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                input_ids,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                use_cache=True,
                min_new_tokens=50,
            )
        
        # Decode tokens to string
        input_length = input_ids.shape[-1]
        output_str = self.tokenizer.decode(
            output_tokens[0, input_length:].cpu().tolist(),
            skip_special_tokens=False
        )
        
        print(f"✓ Generated {len(output_str)} characters")
        return output_str
    
    def decode_to_audio(self, codes_str: str) -> np.ndarray:
        """
        Decode speech codes to audio waveform.
        
        Args:
            codes_str: String containing speech codes
            
        Returns:
            np.ndarray: Audio waveform
        """
        # Extract speech token IDs using regex
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes_str)]
        
        if len(speech_ids) == 0:
            raise ValueError("No valid speech tokens found in output!")
        
        print(f"\nDecoding {len(speech_ids)} speech codes to audio...")
        
        # Decode with NeuCodec
        with torch.no_grad():
            codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(self.codec_device)
            recon = self.codec.decode_code(codes).cpu().numpy()
        
        wav = recon[0, 0, :]
        print(f"✓ Generated {len(wav)} samples ({len(wav)/self.sample_rate:.2f}s)")
        return wav
    
    def synthesize(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        output_path: str = "output.wav",
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        """
        Full TTS pipeline: text → speech codes → audio.
        
        Args:
            text: Vietnamese text to synthesize
            ref_audio_path: Path to reference audio
            ref_text: Reference text (Vietnamese)
            output_path: Path to save output audio
            temperature: Sampling temperature
            top_k: Top-k sampling
        """
        print("\n" + "=" * 60)
        print("SYNTHESIZING SPEECH")
        print("=" * 60)
        print(f"Text: {text}")
        print(f"Reference: {ref_audio_path}")
        print(f"Output: {output_path}")
        
        # Encode reference
        ref_codes = self.encode_reference(ref_audio_path)
        
        # Generate speech codes
        codes_str = self.generate(text, ref_codes, ref_text, temperature, top_k)
        
        # Decode to audio
        wav = self.decode_to_audio(codes_str)
        
        # Save
        print(f"\nSaving to {output_path}...")
        sf.write(output_path, wav, self.sample_rate)
        print(f"✓ Saved!")
        
        print("\n" + "=" * 60)
        print("✅ SYNTHESIS COMPLETE")
        print("=" * 60)


def find_latest_checkpoint(checkpoints_dir: str) -> str:
    """
    Find the latest checkpoint in the checkpoints directory.
    
    Args:
        checkpoints_dir: Root checkpoints directory
        
    Returns:
        str: Path to latest checkpoint
    """
    # Find all checkpoint-* directories
    checkpoint_dirs = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*"))
    
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}")
    
    # Sort by step number
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    
    latest = checkpoint_dirs[-1]
    step = latest.split("-")[-1]
    
    print(f"Found {len(checkpoint_dirs)} checkpoints")
    print(f"Latest checkpoint: {latest} (step {step})")
    
    return latest


def download_hf_checkpoint(repo_id: str, cache_dir: str | None = DEFAULT_HF_CACHE_DIR) -> str:
    """Download (or reuse cached) checkpoint snapshot from Hugging Face Hub."""
    repo_id = (repo_id or "").strip()
    if not repo_id:
        raise ValueError("Hugging Face repo ID cannot be empty.")

    print(f"\nFetching checkpoint from Hugging Face: {repo_id}")
    snapshot_kwargs = {
        "repo_id": repo_id,
        "local_dir_use_symlinks": False,
        "resume_download": True,
    }

    cache_dir = (cache_dir or "").strip()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        local_dir = os.path.join(cache_dir, repo_id.replace("/", "_"))
        snapshot_kwargs["local_dir"] = local_dir

    local_path = snapshot_download(**snapshot_kwargs)
    print(f"✓ Hugging Face checkpoint ready at: {local_path}")
    return local_path


def resolve_checkpoint_reference(
    checkpoint: str | None = None,
    checkpoints_dir: str | None = None,
    hf_repo: str | None = DEFAULT_HF_REPO,
    hf_cache_dir: str | None = DEFAULT_HF_CACHE_DIR,
    latest_checkpoint_cache: Dict[str, str] | None = None,
) -> Tuple[str, str]:
    """Resolve which checkpoint path should be used for inference."""
    explicit_checkpoint = (checkpoint or "").strip()
    if explicit_checkpoint:
        desc = f"Explicit checkpoint provided: {explicit_checkpoint}"
        return explicit_checkpoint, desc

    hf_repo = (hf_repo or "").strip()
    if hf_repo:
        repo_path = download_hf_checkpoint(hf_repo, hf_cache_dir)
        desc = f"Hugging Face repo: {hf_repo}"
        return repo_path, desc

    if not checkpoints_dir:
        raise ValueError("No checkpoint path or Hugging Face repo provided.")

    checkpoints_dir = checkpoints_dir.strip()
    if latest_checkpoint_cache is not None and checkpoints_dir in latest_checkpoint_cache:
        cached_path = latest_checkpoint_cache[checkpoints_dir]
        desc = f"Cached latest checkpoint in {checkpoints_dir}: {cached_path}"
        return cached_path, desc

    latest_local = find_latest_checkpoint(checkpoints_dir)
    if latest_checkpoint_cache is not None:
        latest_checkpoint_cache[checkpoints_dir] = latest_local
    desc = f"Latest checkpoint in {checkpoints_dir}: {latest_local}"
    return latest_local, desc


def main():
    parser = argparse.ArgumentParser(description="Vietnamese TTS Inference")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Vietnamese text to synthesize"
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        required=True,
        help="Path to reference audio file"
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        required=True,
        help="Reference text (Vietnamese)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (default: auto-find latest)"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./checkpoints/neutts-vietnamese",
        help="Root checkpoints directory (used if --checkpoint not specified)"
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=DEFAULT_HF_REPO,
        help=f"Hugging Face repo ID to download checkpoints (set empty to skip, default: {DEFAULT_HF_REPO})"
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=DEFAULT_HF_CACHE_DIR,
        help="Local directory to cache Hugging Face checkpoints (empty string to use default HF cache)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model (cuda/cpu)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    
    args = parser.parse_args()
    
    checkpoint_path, checkpoint_desc = resolve_checkpoint_reference(
        checkpoint=args.checkpoint,
        checkpoints_dir=args.checkpoints_dir,
        hf_repo=args.hf_repo,
        hf_cache_dir=args.hf_cache_dir,
    )
    print(f"\nUsing checkpoint source: {checkpoint_desc}")
    
    # Initialize TTS
    tts = VietnameseTTS(
        checkpoint_path=checkpoint_path,
        device=args.device,
        codec_device=args.device,
    )
    
    # Synthesize
    tts.synthesize(
        text=args.text,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        output_path=args.output,
        temperature=args.temperature,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
