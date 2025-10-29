#!/usr/bin/env python3
"""
Script để resume training từ checkpoint mới nhất.
Sử dụng: python resume_training.py
"""

import os
import glob
import argparse
from pathlib import Path


def find_latest_checkpoint(checkpoints_dir: str) -> str:
    """
    Tìm checkpoint mới nhất trong thư mục checkpoints.
    
    Args:
        checkpoints_dir: Thư mục chứa checkpoints
        
    Returns:
        str: Đường dẫn đến checkpoint mới nhất
    """
    # Tìm tất cả thư mục checkpoint-*
    checkpoint_dirs = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*"))
    
    if not checkpoint_dirs:
        print(f"❌ Không tìm thấy checkpoint nào trong {checkpoints_dir}")
        return None
    
    # Sắp xếp theo số step
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    
    latest = checkpoint_dirs[-1]
    step = latest.split("-")[-1]
    
    print(f"📁 Tìm thấy {len(checkpoint_dirs)} checkpoints:")
    for cp in checkpoint_dirs[-5:]:  # Hiển thị 5 checkpoint gần nhất
        step_num = cp.split("-")[-1]
        print(f"  - {cp} (step {step_num})")
    
    print(f"\n✅ Checkpoint mới nhất: {latest} (step {step})")
    
    return latest


def update_config_with_checkpoint(config_path: str, checkpoint_path: str):
    """
    Cập nhật file config để sử dụng checkpoint cụ thể.
    
    Args:
        config_path: Đường dẫn đến file config
        checkpoint_path: Đường dẫn đến checkpoint
    """
    print(f"\n📝 Cập nhật config file: {config_path}")
    
    # Đọc file config
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Tìm và thay thế dòng restore_from
    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith('restore_from:'):
            lines[i] = f"restore_from: \"{checkpoint_path}\"  # Resume from checkpoint\n"
            updated = True
            break
    
    if not updated:
        print("❌ Không tìm thấy dòng 'restore_from' trong config!")
        return False
    
    # Ghi lại file
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✅ Đã cập nhật config để resume từ: {checkpoint_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Resume Vietnamese TTS Training")
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./checkpoints/neutts-vietnamese",
        help="Thư mục chứa checkpoints (default: ./checkpoints/neutts-vietnamese)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="finetune_vietnamese_config.yaml",
        help="File config để cập nhật (default: finetune_vietnamese_config.yaml)"
    )
    parser.add_argument(
        "--auto_update",
        action="store_true",
        help="Tự động cập nhật config file"
    )
    
    args = parser.parse_args()
    
    print("🔄 RESUME TRAINING SCRIPT")
    print("=" * 50)
    
    # Kiểm tra thư mục checkpoints
    if not os.path.exists(args.checkpoints_dir):
        print(f"❌ Thư mục checkpoints không tồn tại: {args.checkpoints_dir}")
        print("\n💡 Các khả năng:")
        print("1. Bạn chưa train lần nào")
        print("2. Đường dẫn checkpoints_dir sai")
        print("3. Checkpoint được lưu ở nơi khác")
        return
    
    # Tìm checkpoint mới nhất
    latest_checkpoint = find_latest_checkpoint(args.checkpoints_dir)
    
    if latest_checkpoint is None:
        return
    
    # Kiểm tra file config
    if not os.path.exists(args.config):
        print(f"❌ File config không tồn tại: {args.config}")
        return
    
    print(f"\n📋 THÔNG TIN:")
    print(f"  Checkpoint mới nhất: {latest_checkpoint}")
    print(f"  Config file: {args.config}")
    
    if args.auto_update:
        # Tự động cập nhật config
        success = update_config_with_checkpoint(args.config, latest_checkpoint)
        if success:
            print(f"\n🚀 SẴN SÀNG RESUME!")
            print(f"Chạy lệnh sau để tiếp tục training:")
            print(f"python finetune_vietnamese.py {args.config}")
        else:
            print(f"\n❌ Không thể cập nhật config file")
    else:
        # Chỉ hiển thị thông tin
        print(f"\n💡 Để resume training:")
        print(f"1. Sửa file {args.config}")
        print(f"2. Thay đổi dòng 'restore_from:' thành:")
        print(f"   restore_from: \"{latest_checkpoint}\"")
        print(f"3. Chạy: python finetune_vietnamese.py {args.config}")
        
        print(f"\n🔄 Hoặc chạy với --auto_update để tự động:")
        print(f"python resume_training.py --auto_update")


if __name__ == "__main__":
    main()
