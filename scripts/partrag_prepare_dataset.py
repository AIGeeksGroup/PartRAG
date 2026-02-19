"""
retrieval database
"""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict


def check_retrieval_database(db_path):
    """retrieval database"""
    print(f"\n{'='*80}")
    print(f" Retrieval Database: {db_path}")
    print(f"{'='*80}")
    
    # 
    required_files = ["metadata.json", "image_embeddings.npy", "mesh_embeddings.npy"]
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(db_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
        else:
            print(f" {file}")
    
    if missing_files:
        print(f"\n : {missing_files}")
        return False
    
    # metadata
    with open(os.path.join(db_path, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    num_samples = len(metadata['uids'])
    print(f"\n : {num_samples}")
    
    # embeddings shape
    image_emb = np.load(os.path.join(db_path, "image_embeddings.npy"), mmap_mode='r')
    mesh_emb = np.load(os.path.join(db_path, "mesh_embeddings.npy"), mmap_mode='r')
    
    print(f" Image embeddings shape: {image_emb.shape}")
    print(f" Mesh embeddings shape: {mesh_emb.shape}")
    
    # fused embeddings
    fused_path = os.path.join(db_path, "fused_embeddings.npy")
    if os.path.exists(fused_path):
        fused_emb = np.load(fused_path, mmap_mode='r')
        print(f" Fused embeddings shape: {fused_emb.shape}")
    else:
        print(f"  No fused embeddings (will use image+mesh separately)")
    
    # 
    valid_images = 0
    valid_meshes = 0
    for img_path, mesh_path in zip(metadata['image_paths'], metadata['mesh_paths']):
        if os.path.exists(img_path):
            valid_images += 1
        if os.path.exists(mesh_path):
            valid_meshes += 1
    
    print(f"\n : {valid_images}/{num_samples} ({valid_images/num_samples*100:.1f}%)")
    print(f" mesh: {valid_meshes}/{num_samples} ({valid_meshes/num_samples*100:.1f}%)")
    
    return True


def check_training_dataset(config_path):
    """"""
    print(f"\n{'='*80}")
    print(f" Training Dataset: {config_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(config_path):
        print(f" : {config_path}")
        return False, []
    
    with open(config_path, 'r') as f:
        train_data = json.load(f)
    
    print(f" : {len(train_data)}")
    
    # 
    part_counts = defaultdict(int)
    uids = []
    
    for item in train_data:
        uid = item.get('uid')
        uids.append(uid)
        
        # 
        num_parts = 0
        if 'part_configs' in item:
            num_parts = len(item['part_configs'])
        elif 'parts' in item:
            num_parts = len(item['parts'])
        
        part_counts[num_parts] += 1
    
    print(f"\n:")
    for num, count in sorted(part_counts.items()):
        percentage = count / len(train_data) * 100
        print(f"  {num}: {count} ({percentage:.1f}%)")
    
    if part_counts[0] == len(train_data):
        print(f"\n  : part!")
        print(f"   part-level")
        print(f"   ")
    
    return True, uids


def check_database_training_overlap(db_path, train_config_path):
    """retrieval database"""
    print(f"\n{'='*80}")
    print(f" Database  Training Data ")
    print(f"{'='*80}")
    
    # database UIDs
    with open(os.path.join(db_path, "metadata.json"), 'r') as f:
        db_metadata = json.load(f)
    db_uids = set(db_metadata['uids'])
    
    # training UIDs
    with open(train_config_path, 'r') as f:
        train_data = json.load(f)
    train_uids = set([item['uid'] for item in train_data])
    
    # 
    overlap = db_uids & train_uids
    only_db = db_uids - train_uids
    only_train = train_uids - db_uids
    
    print(f"\n :")
    print(f"  Database: {len(db_uids)}")
    print(f"  Training: {len(train_uids)}")
    print(f"  : {len(overlap)}")
    print(f"  : {len(overlap)/len(train_uids)*100:.1f}%")
    
    if only_train:
        print(f"\n   {len(only_train)} database")
        print(f"   : {list(only_train)[:5]}")
    
    if len(overlap) / len(train_uids) < 0.9:
        print(f"\n  90%,retrieval database")
        return False
    else:
        print(f"\n !")
        return True


def main():
    print("="*80)
    print(" PartRAG ")
    print("="*80)
    
    # 
    db_high_quality = "/root/autodl-tmp/retrieval_database_high_quality"
    db_filtered = "/root/autodl-tmp/retrieval_database_high_quality_filtered"
    train_config = "/root/autodl-tmp/dataset/Objaverse/processed/high_quality_object_part_configs.json"
    
    # 1. retrieval databases
    print("\n" + "="*80)
    print(": Retrieval Databases")
    print("="*80)
    
    db1_ok = check_retrieval_database(db_high_quality)
    db2_ok = check_retrieval_database(db_filtered)
    
    # database
    if db1_ok and db2_ok:
        print(f"\n{'='*80}")
        print(f" : {db_high_quality}")
        print(f"   :  (1139 vs 568),")
        print(f"{'='*80}")
        recommended_db = db_high_quality
    elif db1_ok:
        recommended_db = db_high_quality
    elif db2_ok:
        recommended_db = db_filtered
    else:
        print("\n database,")
        return
    
    # 2. 
    print("\n" + "="*80)
    print(": Training Dataset")
    print("="*80)
    
    train_ok, train_uids = check_training_dataset(train_config)
    
    # 3. 
    if train_ok and db1_ok:
        print("\n" + "="*80)
        print(":")
        print("="*80)
        
        match_ok = check_database_training_overlap(recommended_db, train_config)
    
    # 4. 
    print("\n" + "="*80)
    print(" ")
    print("="*80)
    
    print("\n :")
    print(f"   database_path: {recommended_db}")
    print(f"   training_config: {train_config}")
    
    print("\n  :")
    print("   1. retrieval database")
    print("   2. part-level,part")
    print("   3. min_num_parts>=2")
    
    print("\n :")
    print("   1. ,:")
    print("      ./start_training_IMPROVED.sh")
    print("   2. ,")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

