"""
Quick Vietnamese TTS Inference
Simple script to test the finetuned model.
"""

from infer_vietnamese import VietnameseTTS, find_latest_checkpoint

# Configuration
CHECKPOINT_DIR = "./checkpoints/neutts-vietnamese"
REF_AUDIO = "wavs/vivoice_0.wav"  # Change to your reference audio
REF_TEXT = "Và mọi chuyện thì chưa dừng lại ở đó."  # Change to match your reference audio
TEXT = "Hôm nay trời đẹp quá"  # Text to synthesize
OUTPUT = "output_vietnamese.wav"

def main():
    print("=" * 60)
    print("QUICK VIETNAMESE TTS INFERENCE")
    print("=" * 60)
    
    # Find latest checkpoint
    print("\n[1/3] Finding latest checkpoint...")
    checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
    
    # Load model
    print("\n[2/3] Loading model...")
    tts = VietnameseTTS(
        checkpoint_path=checkpoint,
        device="cuda",  # Change to "cpu" if no GPU
        codec_device="cuda",
    )
    
    # Synthesize
    print("\n[3/3] Synthesizing...")
    tts.synthesize(
        text=TEXT,
        ref_audio_path=REF_AUDIO,
        ref_text=REF_TEXT,
        output_path=OUTPUT,
    )
    
    print(f"\n✅ Done! Audio saved to: {OUTPUT}")


if __name__ == "__main__":
    main()

