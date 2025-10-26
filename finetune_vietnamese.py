"""
Finetune NeuTTS-Air for Vietnamese language.
Based on the original finetune.py but adapted for Vietnamese dataset.

This version supports ON-THE-FLY audio encoding:
- No need to pre-encode audio files
- Encodes audio during training using NeuCodec
- Supports both CSV metadata and pre-encoded pickle files
"""

import warnings
import re
import os
import torch
import pickle
import phonemizer
import pandas as pd
from pathlib import Path
from fire import Fire
from omegaconf import OmegaConf
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator
from loguru import logger as LOGGER
from datasets import Dataset

# Try to import audio processing libraries
try:
    from librosa import load as librosa_load
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    LOGGER.warning("librosa not available - on-the-fly encoding disabled")

try:
    from neucodec import NeuCodec
    NEUCODEC_AVAILABLE = True
except ImportError:
    NEUCODEC_AVAILABLE = False
    LOGGER.warning("neucodec not available - on-the-fly encoding disabled")

warnings.filterwarnings("ignore")


class OnTheFlyDataCollator:
    """
    Custom data collator that encodes audio on-the-fly during training.
    """
    def __init__(self, preprocess_fn):
        self.preprocess_fn = preprocess_fn

    def __call__(self, batch):
        """
        Process a batch of samples.

        Args:
            batch: List of raw samples (with 'audio_file' and 'text')

        Returns:
            Dict with batched tensors
        """
        processed_samples = []

        for sample in batch:
            # Preprocess (including encoding audio)
            processed = self.preprocess_fn(sample)

            if processed is not None:
                processed_samples.append(processed)

        if len(processed_samples) == 0:
            # Return empty batch if all samples failed
            return {
                "input_ids": torch.tensor([]),
                "labels": torch.tensor([]),
                "attention_mask": torch.tensor([]),
            }

        # Stack tensors and ensure correct dtypes
        input_ids = torch.stack([s["input_ids"] for s in processed_samples]).long()
        labels = torch.stack([s["labels"] for s in processed_samples]).long()
        attention_mask = torch.stack([s["attention_mask"] for s in processed_samples]).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# Vietnamese-specific filters (less strict than English)
def data_filter(sample):
    """Filter out invalid samples."""
    text = sample["text"]

    if len(text) == 0:
        return False

    # Allow Vietnamese text with numbers and special characters
    # Just check if text is not empty and has reasonable length
    if len(text) > 500:  # Skip very long texts
        return False

    return True


def encode_audio_on_the_fly(audio_path, codec, device="cuda"):
    """
    Encode audio file on-the-fly using NeuCodec.

    Args:
        audio_path: Path to audio file
        codec: NeuCodec model instance
        device: Device to use for encoding

    Returns:
        list: Encoded codes as list of integers
    """
    try:
        # Load audio at 16kHz (NeuCodec requirement)
        import numpy as np
        wav, _ = librosa_load(audio_path, sr=16000, mono=True)

        # Ensure wav is numpy array (not tensor)
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()

        # NeuCodec's feature_extractor needs CPU tensor!
        # Convert to tensor format: [batch, channels, samples] on CPU
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)

        # Encode with NeuCodec (it will move to device internally)
        with torch.no_grad():
            codes = codec.encode_code(audio_or_path=wav_tensor)

        # Return as list of integers (CPU)
        codes = codes.squeeze(0).squeeze(0).cpu()

        # Convert to list of integers
        if codes.dtype == torch.float32 or codes.dtype == torch.float16:
            codes = codes.long()

        codes_list = codes.tolist()
        return codes_list

    except Exception as e:
        LOGGER.error(f"Failed to encode {audio_path}: {e}")
        import traceback
        LOGGER.error(traceback.format_exc())
        return None


def preprocess_sample(sample, tokenizer, max_len, g2p, codec=None, audio_dir=None, device="cuda"):
    """
    Preprocess a single sample for training.

    Args:
        sample: Dict with 'text' and optionally 'codes' or 'audio_file'
        tokenizer: HuggingFace tokenizer
        max_len: Maximum sequence length
        g2p: Phonemizer backend
        codec: NeuCodec model (optional, for on-the-fly encoding)
        audio_dir: Directory containing audio files (optional)
        device: Device for encoding

    Returns:
        Dict with 'input_ids', 'labels', 'attention_mask'
    """
    # Get special tokens
    speech_gen_start = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    ignore_index = -100  # Standard ignore index for PyTorch

    # Get text
    text = sample["text"]

    # Get codes: either pre-encoded or encode on-the-fly
    if "codes" in sample and sample["codes"] is not None:
        # Use pre-encoded codes
        vq_codes = sample["codes"]
        if isinstance(vq_codes, list):
            vq_codes = torch.tensor(vq_codes)
    elif "audio_file" in sample and codec is not None and audio_dir is not None:
        # Encode on-the-fly
        audio_path = os.path.join(audio_dir, sample["audio_file"])
        vq_codes = encode_audio_on_the_fly(audio_path, codec, device)
        if vq_codes is None:
            LOGGER.warning(f"Failed to encode {audio_path}")
            return None
    else:
        LOGGER.error("Sample has neither 'codes' nor 'audio_file' field!")
        return None
    
    # Phonemize Vietnamese text
    try:
        phones = g2p.phonemize([text])
        
        # SAFE CHECK
        if not phones or not phones[0]:
            LOGGER.warning(f"⚠️ Empty phonemization output for text: {text}")
            return None
        
        phones = phones[0].split()
        phones = ' '.join(phones)
    except Exception as e:
        LOGGER.warning(f"⚠️ Phonemization error for text '{text}': {e}")
        return None
    
    # Convert codes to string format
    codes_str = "".join([f"<|speech_{i}|>" for i in vq_codes])
    
    # Create chat format (same as original)
    chat = f"""user: Convert the text to speech:<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"""
    
    # Tokenize
    ids = tokenizer.encode(chat)
    
    # Pad or truncate to max_len
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    
    # Convert to tensor
    input_ids = torch.tensor(ids, dtype=torch.long)
    
    # Create labels (only train on speech generation part)
    labels = torch.full_like(input_ids, ignore_index)
    speech_gen_start_idx = (input_ids == speech_gen_start).nonzero(as_tuple=True)[0]
    if len(speech_gen_start_idx) > 0:
        speech_gen_start_idx = speech_gen_start_idx[0]
        labels[speech_gen_start_idx:] = input_ids[speech_gen_start_idx:]
    
    # Create attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    # Return in HuggingFace format
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def main(config_fpath: str):
    """
    Main training function.
    
    Args:
        config_fpath: Path to config YAML file
    """
    print("=" * 60)
    print("FINETUNING NEUTTS-AIR FOR VIETNAMESE")
    print("=" * 60)
    
    # Load config
    print(f"\n[1/6] Loading config from {config_fpath}")
    config = OmegaConf.load(config_fpath)
    checkpoints_dir = os.path.join(config.save_root, config.run_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    LOGGER.info(f"Checkpoints will be saved to: {checkpoints_dir}")
    
    # Load model and tokenizer
    restore_from = config.restore_from
    print(f"\n[2/6] Loading model and tokenizer from {restore_from}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(restore_from)
        print(f"✓ Tokenizer loaded")

        # Try loading with different options
        print(f"  Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            restore_from,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print(f"✓ Model loaded: {model.num_parameters():,} parameters")

    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Clear HuggingFace cache:")
        print("   rm -rf ~/.cache/huggingface/hub/models--neuphonic--neutts-air")
        print("\n2. Try downloading manually:")
        print("   huggingface-cli download neuphonic/neutts-air")
        print("\n3. Check transformers version:")
        print("   pip install --upgrade transformers")
        raise
    
    # Initialize Vietnamese phonemizer
    print(f"\n[3/6] Initializing Vietnamese phonemizer...")
    try:
        g2p = phonemizer.backend.EspeakBackend(
            language='vi',  # Vietnamese language code
            preserve_punctuation=True,
            with_stress=True,
            words_mismatch="ignore",
            language_switch="remove-flags"
        )
        # Test phonemizer
        test_result = g2p.phonemize(["Xin chào"])
        print(f"✓ Phonemizer initialized successfully!")
        print(f"  Test: 'Xin chào' → {test_result}")
    except Exception as e:
        print(f"❌ Error initializing phonemizer: {e}")
        print("Make sure espeak-ng is installed with Vietnamese support:")
        print("  Ubuntu/Debian: sudo apt-get install espeak-ng")
        print("  Windows: Download from https://github.com/espeak-ng/espeak-ng/releases")
        raise
    
    # Load Vietnamese dataset
    print(f"\n[4/6] Loading Vietnamese dataset...")

    # Determine dataset type
    dataset_path = config.dataset_path

    if dataset_path.endswith('.pkl'):
        # Load pre-encoded pickle file
        print(f"  Loading pre-encoded dataset from {dataset_path}")
        with open(dataset_path, 'rb') as f:
            dataset_list = pickle.load(f)
        print(f"✓ Loaded {len(dataset_list)} pre-encoded samples")
        dataset = Dataset.from_list(dataset_list)
        codec = None
        audio_dir = None

    elif dataset_path.endswith('.csv'):
        # Load CSV metadata for on-the-fly encoding
        print(f"  Loading CSV metadata from {dataset_path}")

        if not LIBROSA_AVAILABLE or not NEUCODEC_AVAILABLE:
            raise RuntimeError(
                "On-the-fly encoding requires librosa and neucodec!\n"
                "Install: pip install librosa neucodec"
            )

        # Read CSV
        df = pd.read_csv(dataset_path, sep='|', names=['audio_file', 'text'])
        print(f"✓ Loaded {len(df)} samples from CSV")

        # Limit dataset size if specified
        max_samples = getattr(config, 'max_samples', None)
        if max_samples is not None and max_samples > 0:
            print(f"  ⚠️  Limiting to {max_samples} samples (set in config)")
            df = df.head(max_samples)
            print(f"  Using {len(df)} samples")

        # Get audio directory
        audio_dir = getattr(config, 'audio_dir', 'wavs')
        if not os.path.isabs(audio_dir):
            # Make relative to dataset path
            audio_dir = os.path.join(os.path.dirname(dataset_path), audio_dir)

        print(f"  Audio directory: {audio_dir}")

        # Load NeuCodec
        print(f"  Loading NeuCodec model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device)
        codec.eval()
        print(f"✓ NeuCodec loaded on {device}")

        # Convert to dataset
        dataset_list = df.to_dict('records')
        dataset = Dataset.from_list(dataset_list)

    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}. Use .pkl or .csv")

    print(f"  Dataset columns: {dataset.column_names}")
    
    # Preprocess dataset
    print(f"\n[5/6] Preprocessing dataset...")

    # Prepare preprocessing function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    partial_preprocess = partial(
        preprocess_sample,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        g2p=g2p,
        codec=codec,
        audio_dir=audio_dir,
        device=device,
    )

    # Filter dataset
    dataset = dataset.filter(data_filter)
    print(f"  After filtering: {len(dataset)} samples")

    if codec is not None:
        # ON-THE-FLY MODE: Don't preprocess now, will encode during training!
        print(f"  ⚡ ON-THE-FLY ENCODING MODE")
        print(f"  Audio will be encoded DURING training (not now)")
        print(f"  This saves time and disk space!")

        # Set format to enable lazy loading
        dataset.set_format(type=None)  # Keep original format

    else:
        # PRE-ENCODED MODE: Don't preprocess! Will do on-the-fly with collator
        print(f"  📦 USING PRE-ENCODED DATA")
        print(f"  ⚡ Will preprocess on-the-fly during training (saves RAM!)")

        # Keep original format - don't load everything into RAM
        dataset.set_format(type=None)

    print(f"✓ Dataset ready: {len(dataset)} samples")

    # Split dataset into train/val (99.5% / 0.5%)
    print(f"\n[6/7] Splitting dataset into train/val...")
    dataset_dict = dataset.train_test_split(test_size=0.005, seed=config.seed)
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["test"]
    print(f"✓ Train: {len(train_dataset)} samples (99.5%)")
    print(f"✓ Val: {len(val_dataset)} samples (0.5%)")

    # Setup training
    print(f"\n[7/7] Setting up training...")

    # Always use custom collator for on-the-fly preprocessing
    # This saves RAM by not loading all samples at once
    data_collator = OnTheFlyDataCollator(partial_preprocess)
    print(f"  Using OnTheFlyDataCollator (on-the-fly preprocessing)")

    # Set num_workers based on whether we need CUDA for encoding
    if codec is not None:
        dataloader_num_workers = 0  # Must be 0 for CUDA encoding
        print(f"  Dataloader workers: 0 (CUDA encoding)")
    else:
        dataloader_num_workers = 8  # Can use multiple workers for pre-encoded
        print(f"  Dataloader workers: 8 (pre-encoded data)")

    # Use epochs or max_steps (epochs takes priority)
    num_train_epochs = getattr(config, 'num_train_epochs', None)
    max_steps = getattr(config, 'max_steps', -1)
    eval_steps = getattr(config, 'eval_steps', config.save_steps)
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)

    # Speed optimizations
    use_torch_compile = getattr(config, 'torch_compile', False)
    gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)
    tf32 = getattr(config, 'tf32', True)  # Enable TF32 for Ampere+ GPUs
    dataloader_pin_memory = getattr(config, 'dataloader_pin_memory', True)
    dataloader_prefetch_factor = getattr(config, 'dataloader_prefetch_factor', 2)

    # Enable TF32 for faster training on Ampere GPUs (RTX 30xx, A100, etc.)
    if tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"  ✓ TF32 enabled for faster training")

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        do_train=True,
        do_eval=False,
        learning_rate=config.lr,
        num_train_epochs=num_train_epochs if num_train_epochs else None,
        max_steps=max_steps if not num_train_epochs else -1,
        bf16=True,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,  # Trade memory for speed
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        eval_steps=eval_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        ignore_data_skip=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        torch_compile=use_torch_compile,  # PyTorch 2.0 compilation
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,  # Faster data transfer to GPU
        dataloader_prefetch_factor=dataloader_prefetch_factor if dataloader_num_workers > 0 else None,  # Prefetch batches
        optim="adamw_torch_fused",  # Faster fused AdamW optimizer
        ddp_find_unused_parameters=False,  # Faster DDP
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size per device: {config.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {config.per_device_train_batch_size * gradient_accumulation_steps}")

    if num_train_epochs:
        total_steps = (len(train_dataset) // (config.per_device_train_batch_size * gradient_accumulation_steps)) * num_train_epochs
        print(f"Training epochs: {num_train_epochs}")
        print(f"Estimated total steps: ~{total_steps:,}")
    else:
        print(f"Max steps: {max_steps}")

    print(f"Learning rate: {config.lr}")
    print(f"Save every: {config.save_steps} steps")
    print(f"Eval every: {eval_steps} steps")
    print("=" * 60 + "\n")
    
    # Train!
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 60)
    print("SAVING FINAL MODEL")
    print("=" * 60)
    trainer.save_model(checkpoints_dir)
    print(f"✓ Model saved to: {checkpoints_dir}")
    print("\n✅ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    Fire(main)

