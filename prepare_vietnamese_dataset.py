"""
Script to prepare Vietnamese dataset for NeuTTS-Air finetuning.
This script:
1. Reads metadata.csv
2. Encodes all audio files using NeuCodec
3. Saves the encoded dataset as a pickle file
"""

import os
import csv
import torch
import pickle
from tqdm import tqdm
from librosa import load
from neucodec import NeuCodec

def encode_audio_file(audio_path, codec):
    """Encode a single audio file using NeuCodec."""
    try:
        # Load audio at 16kHz (required by NeuCodec)
        wav, _ = load(audio_path, sr=16000, mono=True)
        
        # Convert to tensor format [1, 1, T]
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
        
        # Encode to codes
        with torch.no_grad():
            codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        
        # Convert to list of integers for storage
        codes_list = codes.cpu().numpy().tolist()
        
        return codes_list
    except Exception as e:
        print(f"Error encoding {audio_path}: {e}")
        return None

def prepare_dataset(metadata_path, audio_dir, output_path, device="cuda"):
    """
    Prepare the Vietnamese dataset.
    
    Args:
        metadata_path: Path to metadata.csv
        audio_dir: Directory containing audio files
        output_path: Path to save the encoded dataset
        device: Device to use for encoding (cuda/cpu)
    """
    print("=" * 60)
    print("PREPARING VIETNAMESE DATASET FOR NEUTTS-AIR")
    print("=" * 60)
    
    # Load NeuCodec
    print(f"\n[1/4] Loading NeuCodec model on {device}...")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to(device)
    print("✓ NeuCodec loaded successfully!")
    
    # Read metadata
    print(f"\n[2/4] Reading metadata from {metadata_path}...")
    dataset = []
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            audio_file = row['audio']
            transcript = row['transcript']
            
            audio_path = os.path.join(audio_dir, audio_file)
            
            if not os.path.exists(audio_path):
                print(f"⚠️  Warning: Audio file not found: {audio_path}")
                continue
            
            dataset.append({
                'audio_file': audio_file,
                'audio_path': audio_path,
                'text': transcript
            })
    
    print(f"✓ Found {len(dataset)} samples")
    
    # Encode all audio files
    print(f"\n[3/4] Encoding audio files with NeuCodec...")
    encoded_dataset = []
    
    for sample in tqdm(dataset, desc="Encoding"):
        codes = encode_audio_file(sample['audio_path'], codec)
        
        if codes is not None:
            encoded_dataset.append({
                'audio_file': sample['audio_file'],
                'text': sample['text'],
                'codes': codes
            })
        else:
            print(f"⚠️  Skipping {sample['audio_file']} due to encoding error")
    
    print(f"✓ Successfully encoded {len(encoded_dataset)} samples")
    
    # Save dataset
    print(f"\n[4/4] Saving encoded dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(encoded_dataset, f)
    
    print(f"✓ Dataset saved successfully!")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(encoded_dataset)}")
    
    if encoded_dataset:
        avg_codes_len = sum(len(s['codes']) for s in encoded_dataset) / len(encoded_dataset)
        avg_text_len = sum(len(s['text']) for s in encoded_dataset) / len(encoded_dataset)
        print(f"Average codes length: {avg_codes_len:.1f}")
        print(f"Average text length: {avg_text_len:.1f} characters")
        
        print("\nSample data:")
        sample = encoded_dataset[0]
        print(f"  Audio: {sample['audio_file']}")
        print(f"  Text: {sample['text']}")
        print(f"  Codes length: {len(sample['codes'])}")
        print(f"  First 10 codes: {sample['codes'][:10]}")
    
    print("\n✅ Dataset preparation complete!")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Vietnamese dataset for NeuTTS-Air")
    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata.csv",
        help="Path to metadata.csv file"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="wavs",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vietnamese_dataset.pkl",
        help="Output path for encoded dataset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for encoding"
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        metadata_path=args.metadata,
        audio_dir=args.audio_dir,
        output_path=args.output,
        device=args.device
    )

