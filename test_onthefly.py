"""
Quick test script to verify on-the-fly encoding works.
Tests the OnTheFlyDataCollator with a small batch.
"""

import torch
from neucodec import NeuCodec
from librosa import load as librosa_load
from transformers import AutoTokenizer
import phonemizer
import os
from functools import partial

# Import from finetune script
import sys
sys.path.insert(0, '.')
from finetune_vietnamese import OnTheFlyDataCollator, preprocess_sample

def test_collator():
    print("=" * 60)
    print("TESTING ON-THE-FLY DATA COLLATOR")
    print("=" * 60)

    # Check if test files exist
    if not os.path.exists("metadata.csv"):
        print("❌ metadata.csv not found!")
        return

    # Read first 3 lines from metadata
    samples = []
    with open("metadata.csv", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            if '|' in line:
                audio_file, text = line.strip().split('|', 1)
                samples.append({"audio_file": audio_file, "text": text})

    print(f"\n✓ Loaded {len(samples)} test samples")

    # Load components
    print(f"\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("neuphonic/neutts-air")
    print(f"✓ Tokenizer loaded")

    print(f"\n[2/5] Loading phonemizer...")
    g2p = phonemizer.backend.EspeakBackend(language='vi')
    print(f"✓ Phonemizer loaded")

    print(f"\n[3/5] Loading NeuCodec...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device)
    codec.eval()
    print(f"✓ NeuCodec loaded on {device}")

    # Create preprocessing function
    print(f"\n[4/5] Creating data collator...")
    partial_preprocess = partial(
        preprocess_sample,
        tokenizer=tokenizer,
        max_len=2048,
        g2p=g2p,
        codec=codec,
        audio_dir="wavs",
        device=device,
    )

    collator = OnTheFlyDataCollator(partial_preprocess)
    print(f"✓ Data collator created")

    # Test collator
    print(f"\n[5/5] Testing collator with batch of {len(samples)} samples...")
    print(f"  This will encode audio ON-THE-FLY...")

    batch = collator(samples)

    print(f"\n✓ Batch processed successfully!")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")

    print("\n" + "=" * 60)
    print("✅ ON-THE-FLY COLLATOR WORKS!")
    print("=" * 60)
    print("\nNow you can run training:")
    print("  python finetune_vietnamese.py finetune_vietnamese_config.yaml")
    print("\nAudio will be encoded DURING training, not before!")

if __name__ == "__main__":
    test_collator()

