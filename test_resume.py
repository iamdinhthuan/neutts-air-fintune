#!/usr/bin/env python3
"""
Script test để kiểm tra resume training.
"""

import os
import glob

def find_latest_checkpoint():
    """Tìm checkpoint mới nhất"""
    checkpoints_dir = "./checkpoints/neutts-vietnamese"
    
    if not os.path.exists(checkpoints_dir):
        print(f"❌ Thư mục checkpoints không tồn tại: {checkpoints_dir}")
        return None
    
    checkpoint_dirs = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*"))
    
    if not checkpoint_dirs:
        print(f"❌ Không tìm thấy checkpoint nào trong {checkpoints_dir}")
        return None
    
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    latest = checkpoint_dirs[-1]
    
    print(f"📁 Tìm thấy {len(checkpoint_dirs)} checkpoints")
    print(f"✅ Checkpoint mới nhất: {latest}")
    
    # Kiểm tra trainer_state.json
    trainer_state_path = os.path.join(latest, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        print(f"✅ Có trainer_state.json - có thể resume")
        
        # Đọc thông tin từ trainer_state.json
        import json
        with open(trainer_state_path, 'r') as f:
            state = json.load(f)
        
        print(f"📊 Thông tin checkpoint:")
        print(f"  - Global step: {state.get('global_step', 'N/A')}")
        print(f"  - Epoch: {state.get('epoch', 'N/A')}")
        print(f"  - Best metric: {state.get('best_metric', 'N/A')}")
        
        return latest
    else:
        print(f"❌ Không có trainer_state.json - không thể resume")
        return None

if __name__ == "__main__":
    print("🔍 KIỂM TRA CHECKPOINT")
    print("=" * 40)
    
    checkpoint = find_latest_checkpoint()
    
    if checkpoint:
        print(f"\n💡 Để resume training:")
        print(f"1. Sửa finetune_vietnamese_config.yaml:")
        print(f"   restore_from: \"{checkpoint}\"")
        print(f"2. Chạy: python finetune_vietnamese.py finetune_vietnamese_config.yaml")
    else:
        print(f"\n❌ Không thể resume training")
