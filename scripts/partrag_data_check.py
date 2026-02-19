"""
:,retrieval database,mesh,embeddings
"""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def check_training_data(config_path):
    """"""
    print("\n" + "=" * 80)
    print("  ")
    print("=" * 80)
    
    if not os.path.exists(config_path):
        print(f" : {config_path}")
        return None, set()
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    print(f" ")
    print(f"   : {len(data)}")
    
    # 
    part_dist = defaultdict(int)
    uids = set()
    issues = []
    
    for idx, item in enumerate(data):
        uid = item.get('uid')
        if not uid:
            issues.append(f" {idx}  uid")
            continue
        
        uids.add(uid)
        num_parts = len(item.get('part_configs', []))
        part_dist[num_parts] += 1
        
        # 
        if 'part_configs' not in item:
            issues.append(f" {uid}  part_configs")
    
    print(f"\n :")
    for num in sorted(part_dist.keys())[:10]:
        count = part_dist[num]
        pct = count / len(data) * 100
        print(f"   {num:2d}: {count:4d} ({pct:5.1f}%)")
    
    if part_dist[0] > 0:
        print(f"\n  :  {part_dist[0]} part")
    
    # 2+(part-level)
    multi_part_samples = sum(count for num, count in part_dist.items() if num >= 2)
    print(f"\n part-level: {multi_part_samples} ({multi_part_samples/len(data)*100:.1f}%)")
    
    if issues:
        print(f"\n   {len(issues)} :")
        for issue in issues[:5]:
            print(f"   - {issue}")
        if len(issues) > 5:
            print(f"   - ...  {len(issues)-5} ")
    
    return data, uids


def check_retrieval_database(db_path):
    """retrieval database"""
    print("\n" + "=" * 80)
    print("   Retrieval Database")
    print("=" * 80)
    print(f": {db_path}")
    
    issues = []
    
    # 
    required_files = {
        'metadata.json': 'metadata',
        'image_embeddings.npy': 'image embeddings',
        'mesh_embeddings.npy': 'mesh embeddings',
    }
    
    for filename, desc in required_files.items():
        filepath = os.path.join(db_path, filename)
        if not os.path.exists(filepath):
            issues.append(f": {filename}")
            print(f" {desc}: ")
        else:
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f" {desc}: {size_mb:.1f} MB")
    
    if issues:
        print(f"\n Database,")
        return None, set()
    
    # metadata
    with open(os.path.join(db_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    uids = set(metadata['uids'])
    num_samples = len(uids)
    
    print(f"\n Database:")
    print(f"   : {num_samples}")
    
    # embeddings shape
    img_emb = np.load(os.path.join(db_path, 'image_embeddings.npy'), mmap_mode='r')
    mesh_emb = np.load(os.path.join(db_path, 'mesh_embeddings.npy'), mmap_mode='r')
    
    print(f"   Image embeddings shape: {img_emb.shape}")
    print(f"   Mesh embeddings shape: {mesh_emb.shape}")
    
    # shape
    if img_emb.shape[0] != num_samples:
        issues.append(f"Image embeddings({img_emb.shape[0]})metadata({num_samples})")
    if mesh_emb.shape[0] != num_samples:
        issues.append(f"Mesh embeddings({mesh_emb.shape[0]})metadata({num_samples})")
    if img_emb.shape != mesh_emb.shape:
        issues.append(f"ImageMesh embeddings shape")
    
    # fused embeddings
    fused_path = os.path.join(db_path, 'fused_embeddings.npy')
    if os.path.exists(fused_path):
        fused_emb = np.load(fused_path, mmap_mode='r')
        print(f"   Fused embeddings shape: {fused_emb.shape}")
        if fused_emb.shape[0] != num_samples:
            issues.append(f"Fused embeddings({fused_emb.shape[0]})metadata({num_samples})")
    else:
        print(f"   Fused embeddings:  ()")
    
    # ()
    print(f"\n  (100)...")
    sample_size = min(100, num_samples)
    sample_indices = np.random.choice(num_samples, sample_size, replace=False)
    
    valid_images = 0
    valid_meshes = 0
    
    for idx in sample_indices:
        img_path = metadata['image_paths'][idx]
        mesh_path = metadata['mesh_paths'][idx]
        
        if os.path.exists(img_path):
            valid_images += 1
        if os.path.exists(mesh_path):
            valid_meshes += 1
    
    img_pct = valid_images / sample_size * 100
    mesh_pct = valid_meshes / sample_size * 100
    
    print(f"   : {valid_images}/{sample_size} ({img_pct:.1f}%)")
    print(f"   mesh: {valid_meshes}/{sample_size} ({mesh_pct:.1f}%)")
    
    if img_pct < 100:
        issues.append(f" {100-img_pct:.1f}% ")
    if mesh_pct < 100:
        issues.append(f" {100-mesh_pct:.1f}% mesh")
    
    if issues:
        print(f"\n   {len(issues)} :")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"\n Database,")
    
    return metadata, uids


def check_mesh_files(train_data, base_dir):
    """mesh"""
    print("\n" + "=" * 80)
    print("   Mesh ")
    print("=" * 80)
    
    print(f": {base_dir}")
    print(f" {len(train_data)} mesh...")
    
    missing_meshes = []
    valid_meshes = 0
    
    for item in tqdm(train_data, desc="mesh"):
        uid = item['uid']
        
        # mesh
        possible_paths = [
            os.path.join(base_dir, 'hf-objaverse-v1/glbs/000-023', f"{uid}.glb"),
            os.path.join(base_dir, 'hf-objaverse-v1/glbs/024-047', f"{uid}.glb"),
            os.path.join(base_dir, 'hf-objaverse-v1/glbs/048-071', f"{uid}.glb"),
            # 
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                valid_meshes += 1
                found = True
                break
        
        if not found:
            missing_meshes.append(uid)
    
    pct = valid_meshes / len(train_data) * 100
    
    print(f"\n Mesh:")
    print(f"   mesh: {valid_meshes}/{len(train_data)} ({pct:.1f}%)")
    
    if missing_meshes:
        print(f"\n   {len(missing_meshes)} mesh:")
        for uid in missing_meshes[:10]:
            print(f"   - {uid}")
        if len(missing_meshes) > 10:
            print(f"   - ...  {len(missing_meshes)-10} ")
    else:
        print(f"\n mesh")
    
    return valid_meshes, missing_meshes


def check_processed_data(train_data, processed_dir):
    """processed(,points)"""
    print("\n" + "=" * 80)
    print("   Processed ")
    print("=" * 80)
    
    print(f"Processed: {processed_dir}")
    print(f" {len(train_data)} processed...")
    
    stats = {
        'has_image': 0,
        'has_points': 0,
        'has_num_parts': 0,
        'complete': 0,
    }
    
    missing_data = []
    
    for item in tqdm(train_data[:100], desc="processed(100)"):  # 
        uid = item['uid']
        sample_dir = os.path.join(processed_dir, uid)
        
        if not os.path.exists(sample_dir):
            missing_data.append(uid)
            continue
        
        has_all = True
        
        # 
        img_path = os.path.join(sample_dir, 'rendering_rmbg.png')
        if os.path.exists(img_path):
            stats['has_image'] += 1
        else:
            has_all = False
        
        # points
        points_path = os.path.join(sample_dir, 'points.npy')
        if os.path.exists(points_path):
            stats['has_points'] += 1
        else:
            has_all = False
        
        # num_parts
        num_parts_path = os.path.join(sample_dir, 'num_parts.json')
        if os.path.exists(num_parts_path):
            stats['has_num_parts'] += 1
        else:
            has_all = False
        
        if has_all:
            stats['complete'] += 1
    
    total_checked = min(100, len(train_data))
    
    print(f"\n Processed ({total_checked}):")
    print(f"   : {stats['has_image']}/{total_checked} ({stats['has_image']/total_checked*100:.1f}%)")
    print(f"   points: {stats['has_points']}/{total_checked} ({stats['has_points']/total_checked*100:.1f}%)")
    print(f"   num_parts: {stats['has_num_parts']}/{total_checked} ({stats['has_num_parts']/total_checked*100:.1f}%)")
    print(f"   : {stats['complete']}/{total_checked} ({stats['complete']/total_checked*100:.1f}%)")
    
    if missing_data:
        print(f"\n  {len(missing_data)} processed")
    
    return stats, missing_data


def check_data_consistency(train_uids, db_uids):
    """database"""
    print("\n" + "=" * 80)
    print("  ")
    print("=" * 80)
    
    overlap = train_uids & db_uids
    only_train = train_uids - db_uids
    only_db = db_uids - train_uids
    
    print(f" :")
    print(f"   : {len(train_uids)} ")
    print(f"   Database: {len(db_uids)} ")
    print(f"   : {len(overlap)} ")
    print(f"   : {len(overlap)/len(train_uids)*100:.1f}%")
    
    if only_train:
        print(f"\n  {len(only_train)} database:")
        for uid in list(only_train)[:5]:
            print(f"   - {uid}")
        if len(only_train) > 5:
            print(f"   - ...  {len(only_train)-5} ")
    
    if only_db:
        print(f"\n {len(only_db)} database ()")
    
    if len(overlap) / len(train_uids) >= 0.99:
        print(f"\n  (>=99%)")
        return True
    else:
        print(f"\n   (<99%),")
        return False


def main():
    print("=" * 80)
    print(" ")
    print("=" * 80)
    
    # 
    train_config = "/root/autodl-tmp/dataset/Objaverse/processed/high_quality_object_part_configs_FIXED.json"
    db_path = "/root/autodl-tmp/retrieval_database_high_quality"
    objaverse_base = "/root/autodl-tmp/dataset/Objaverse"
    processed_dir = "/root/autodl-tmp/dataset/Objaverse/processed"
    
    # 1. 
    train_data, train_uids = check_training_data(train_config)
    if train_data is None:
        print("\n ,")
        return
    
    # 2. retrieval database
    db_metadata, db_uids = check_retrieval_database(db_path)
    if db_metadata is None:
        print("\n Database,")
        return
    
    # 3. mesh
    valid_meshes, missing_meshes = check_mesh_files(train_data, objaverse_base)
    
    # 4. processed
    processed_stats, missing_processed = check_processed_data(train_data, processed_dir)
    
    # 5. 
    consistency_ok = check_data_consistency(train_uids, db_uids)
    
    # 
    print("\n" + "=" * 80)
    print(" ")
    print("=" * 80)
    
    issues_found = []
    
    # 
    if train_data:
        multi_part = sum(1 for item in train_data if len(item.get('part_configs', [])) >= 2)
        multi_part_pct = multi_part / len(train_data) * 100
        print(f"\n : {len(train_data)} ")
        print(f"   - {multi_part} part-level ({multi_part_pct:.1f}%)")
        if multi_part_pct < 50:
            issues_found.append("part-level")
    
    # Retrieval database
    if db_metadata:
        print(f"\n Retrieval Database: {len(db_uids)} ")
        if 'fused_embeddings.npy' not in os.listdir(db_path):
            issues_found.append("Databasefused_embeddings,")
    
    # Mesh
    if valid_meshes:
        mesh_pct = valid_meshes / len(train_data) * 100
        print(f"\n Mesh: {valid_meshes}/{len(train_data)} ({mesh_pct:.1f}%)")
        if mesh_pct < 95:
            issues_found.append(f" {100-mesh_pct:.1f}% mesh")
    
    # 
    if consistency_ok:
        print(f"\n :  (>=99%)")
    else:
        issues_found.append("database <99%")
    
    # 
    print("\n" + "=" * 80)
    if not issues_found:
        print(" !,!")
        print("=" * 80)
        print("\n:")
        print("  cd /root/autodl-tmp/PartRAG")
        print("  ./start_training_IMPROVED.sh")
    else:
        print("  :")
        print("=" * 80)
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        print("\n")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

