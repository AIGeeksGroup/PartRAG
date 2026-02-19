"""
:part num_parts.json ,
"""

import json
import os
from pathlib import Path
from tqdm import tqdm


def fix_part_configs(
    input_config_path: str,
    output_config_path: str,
    processed_data_dir: str,
):
    """
    ,part
    
    Args:
        input_config_path: 
        output_config_path: 
        processed_data_dir: 
    """
    print("=" * 80)
    print(" :part")
    print("=" * 80)
    
    # 
    print(f"\n : {input_config_path}")
    with open(input_config_path, 'r') as f:
        original_config = json.load(f)
    
    print(f" : {len(original_config)}")
    
    # 
    updated_config = []
    skipped = 0
    
    print(f"\n ...")
    for item in tqdm(original_config):
        uid = item['uid']
        sample_dir = os.path.join(processed_data_dir, uid)
        num_parts_file = os.path.join(sample_dir, 'num_parts.json')
        
        # num_parts
        if os.path.exists(num_parts_file):
            try:
                with open(num_parts_file, 'r') as f:
                    num_parts_data = json.load(f)
                num_parts = num_parts_data.get('num_parts', 1)
                
                # part_configs
                # :,part
                part_configs = []
                for part_idx in range(num_parts):
                    part_config = {
                        'part_id': part_idx,
                        'part_label': f'part_{part_idx}',
                        # part-specific
                    }
                    part_configs.append(part_config)
                
                # item
                item['num_parts'] = num_parts
                item['part_configs'] = part_configs
                updated_config.append(item)
                
            except Exception as e:
                print(f"\n   {uid}: {e}")
                skipped += 1
        else:
            # num_parts.json,1
            item['num_parts'] = 1
            item['part_configs'] = [{'part_id': 0, 'part_label': 'whole_object'}]
            updated_config.append(item)
    
    print(f"\n :")
    print(f"   : {len(updated_config)} ")
    print(f"   : {skipped} ")
    
    # 
    from collections import defaultdict
    part_counts = defaultdict(int)
    for item in updated_config:
        num_parts = len(item.get('part_configs', []))
        part_counts[num_parts] += 1
    
    print(f"\n :")
    for num, count in sorted(part_counts.items())[:10]:
        percentage = count / len(updated_config) * 100
        print(f"   {num}: {count} ({percentage:.1f}%)")
    
    # 
    print(f"\n : {output_config_path}")
    with open(output_config_path, 'w') as f:
        json.dump(updated_config, f, indent=2, ensure_ascii=False)
    
    print(f" !")
    
    # 
    print(f"\n ...")
    with open(output_config_path, 'r') as f:
        verified_config = json.load(f)
    
    sample = verified_config[0]
    print(f"\n:")
    print(f"   UID: {sample['uid']}")
    print(f"   num_parts: {sample.get('num_parts', 'N/A')}")
    print(f"   part_configs: {len(sample.get('part_configs', []))}")
    if sample.get('part_configs'):
        print(f"   part: {sample['part_configs'][0]}")
    
    return updated_config


def main():
    # 
    input_config = "/root/autodl-tmp/dataset/Objaverse/processed/high_quality_object_part_configs.json"
    output_config = "/root/autodl-tmp/dataset/Objaverse/processed/high_quality_object_part_configs_FIXED.json"
    processed_dir = "/root/autodl-tmp/dataset/Objaverse/processed"
    
    # 
    updated_config = fix_part_configs(
        input_config_path=input_config,
        output_config_path=output_config,
        processed_data_dir=processed_dir,
    )
    
    print("\n" + "=" * 80)
    print(" !")
    print("=" * 80)
    print(f"\n: {output_config}")
    print(f"\n:")
    print(f"  1. dataset.config:")
    print(f"     dataset:")
    print(f"       config:")
    print(f"       - {output_config}")
    print(f"  2. :")
    print(f"     python scripts/partrag_prepare_dataset.py")
    print(f"  3. :")
    print(f"     ./start_training_IMPROVED.sh")


if __name__ == "__main__":
    main()
