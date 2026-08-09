#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LVS-TAA Token Generator: Local Variance Similarity-based Token Area Allocation

Direct PET Feature Extraction Token Generator - Optimized Version
Based on variance similarity for PET feature extraction


import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from tqdm import tqdm
import json
import time


def compute_patch_variance_similarity_batch(pet_tensor, patch_size):
    """
    Batch compute variance similarity for all patches using unfold
    
    Args:
        pet_tensor: [B, C, D, H, W]
        patch_size: patch size
    
    Returns:
        variance_similarities: [num_patches]
    """
    B, C, D, H, W = pet_tensor.shape
    
    # Check divisibility
    if D % patch_size != 0 or H % patch_size != 0 or W % patch_size != 0:
        # When not divisible, use sliding window
        # First compute number of complete patches
        num_patches_d = D // patch_size
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        
        # Only compute complete patches
        patches_list = []
        for pd in range(num_patches_d):
            for ph in range(num_patches_h):
                for pw in range(num_patches_w):
                    start_d = pd * patch_size
                    end_d = start_d + patch_size
                    start_h = ph * patch_size
                    end_h = start_h + patch_size
                    start_w = pw * patch_size
                    end_w = start_w + patch_size
                    
                    patch = pet_tensor[:, :, start_d:end_d, start_h:end_h, start_w:end_w]
                    var = torch.var(patch)
                    sim = 1.0 / (1.0 + var)
                    patches_list.append(sim)
        
        return torch.stack(patches_list) if patches_list else torch.tensor([], device=pet_tensor.device)
    
    # Use unfold to extract all patches at once
    patches_d = pet_tensor.unfold(2, patch_size, patch_size)
    patches_dh = patches_d.unfold(3, patch_size, patch_size)
    patches_all = patches_dh.unfold(4, patch_size, patch_size)
    
    # Get patch count
    num_patches_d = patches_all.shape[2]
    num_patches_h = patches_all.shape[3]
    num_patches_w = patches_all.shape[4]
    num_patches = num_patches_d * num_patches_h * num_patches_w
    
    # Reshape to [num_patches, C * patch_size^3]
    patches = patches_all.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
    patches = patches.view(B, num_patches, -1)
    
    # Batch compute variance: var = mean(x^2) - mean(x)^2
    mean = patches.mean(dim=2)
    mean_sq = (patches ** 2).mean(dim=2)
    variance = mean_sq - mean ** 2
    
    # Avoid division by zero
    variance = torch.clamp(variance, min=1e-8)
    similarity = 1.0 / (1.0 + variance)
    
    return similarity.squeeze(0)


def compute_patch_variance_similarity_single(pet_tensor, start_d, end_d, start_h, end_h, start_w, end_w):
    """
    Single patch variance computation (for merging step with few patches)
    """
    patch_pet = pet_tensor[:, :, start_d:end_d, start_h:end_h, start_w:end_w]
    
    if patch_pet.numel() == 0:
        return 0.0
    
    patch_var = torch.var(patch_pet)
    similarity = 1.0 / (1.0 + patch_var)
    return similarity.item()


def direct_pet_tokenization(pet_tensor, initial_threshold=0.3, max_patch_size=64, min_patch_size=2, min_core_tokens=5):
    """
    LVS-TAA: Direct PET Feature Extraction Tokenization Algorithm - Optimized Version
    
    Key features:
    - Batch compute all min_patch_size variance at once
    - Threshold iteration only updates judgment, no recalculation needed
    
    Args:
        pet_tensor: PET image tensor [B, C, D, H, W]
        initial_threshold: Initial threshold for core feature identification
        max_patch_size: Maximum patch size
        min_patch_size: Minimum patch size
        min_core_tokens: Minimum number of core tokens
    
    Returns:
        tokens: Token information list
        actual_threshold: Actual threshold used
    """
    B, C, D, H, W = pet_tensor.shape
    device = pet_tensor.device
    t0 = time.time()

    print(f"=== LVS-TAA Tokenization (Optimized) ===")
    print(f"PET shape: {pet_tensor.shape}")
    print(f"Initial threshold: {initial_threshold}, Max patch: {max_patch_size}, Min patch: {min_patch_size}")
    print(f"Min core tokens: {min_core_tokens}")

    # Compute patch count
    num_patches_d = D // min_patch_size
    num_patches_h = H // min_patch_size
    num_patches_w = W // min_patch_size
    total_patches = num_patches_d * num_patches_h * num_patches_w
    
    print(f"Patch count: {num_patches_d} x {num_patches_h} x {num_patches_w} = {total_patches}")

    # Batch compute all min_patch_size variance similarity
    print(f"\n[Optimized] Batch computing {min_patch_size}x{min_patch_size}x{min_patch_size} patch variance similarity...")
    t0 = time.time()
    all_variance_sims = compute_patch_variance_similarity_batch(pet_tensor, min_patch_size)
    t1 = time.time()
    print(f"    Batch variance computation time: {t1-t0:.4f}s")

    num_patches_d = D // min_patch_size
    num_patches_h = H // min_patch_size
    num_patches_w = W // min_patch_size
    all_variance_sims = all_variance_sims.view(num_patches_d, num_patches_h, num_patches_w)
    print(f"    Variance similarity reshaped to 3D: {all_variance_sims.shape}")

    # Threshold iteration
    current_threshold = initial_threshold
    max_threshold = 0.95
    threshold_step = 0.05
    
    core_mask = None
    while current_threshold <= max_threshold:
        print(f"\n=== Trying threshold: {current_threshold:.2f} ===")
        
        # Vectorized filtering
        t4 = time.time()
        core_mask = all_variance_sims < current_threshold
        core_count = core_mask.sum().item()
        t5 = time.time()
        print(f"    Core feature filtering time: {t5-t4:.4f}s")
        
        print(f"Core feature count: {core_count}")
        print(f"Core feature ratio: {core_count / all_variance_sims.numel() * 100:.1f}%")
        
        if core_count >= min_core_tokens:
            print(f"Found {core_count} core features (>= {min_core_tokens}), using threshold: {current_threshold:.2f}")
            break
        else:
            print(f"Threshold {current_threshold:.2f} insufficient core features ({core_count} < {min_core_tokens}), trying higher threshold...")
            current_threshold += threshold_step
    
    if core_mask is None or core_mask.sum().item() < min_core_tokens:
        print(f"Warning: Even at max threshold {max_threshold}, insufficient core features found ({core_mask.sum().item() if core_mask is not None else 0} < {min_core_tokens}), continuing with current threshold {current_threshold:.2f}")
    
    actual_threshold = current_threshold
    
    # Step 2: Record core features
    print(f"\n=== Step 2: Record Core Features ===")
    t6 = time.time()
    tokens = []
    processed_mask = torch.zeros(D, H, W, dtype=torch.bool, device=device)
    
    # Vectorized: directly use True positions in core_mask to compute coordinates
    core_indices = torch.where(core_mask)
    
    for i in range(len(core_indices[0])):
        pd = core_indices[0][i].item()
        ph = core_indices[1][i].item()
        pw = core_indices[2][i].item()
        
        start_d = pd * min_patch_size
        end_d = (pd + 1) * min_patch_size
        start_h = ph * min_patch_size
        end_h = (ph + 1) * min_patch_size
        start_w = pw * min_patch_size
        end_w = (pw + 1) * min_patch_size
        
        variance_sim = all_variance_sims[pd, ph, pw].item()
        
        token_info = {
            'start_d': start_d,
            'end_d': end_d,
            'start_h': start_h,
            'end_h': end_h,
            'start_w': start_w,
            'end_w': end_w,
            'patch_size': min_patch_size,
            'variance_similarity': variance_sim,
            'layer_id': 0,
            'core_feature': True
        }
        tokens.append(token_info)
        
        processed_mask[start_d:end_d, start_h:end_h, start_w:end_w] = True
    
    core_patches_count = len(tokens)
    t7 = time.time()
    print(f"    Core feature recording time: {t7-t6:.4f}s")

    # Step 3: Adaptive merging of non-core regions
    print(f"\n=== Step 3: Adaptive Merging of Non-core Regions ===")
    
    all_merge_sizes = [64, 32, 16, 8, 4, 2]
    merge_sizes = [s for s in all_merge_sizes if s >= min_patch_size]
    print(f"Merge size list (>= {min_patch_size}): {merge_sizes}")
    
    merged_count = 0
    processed_np = processed_mask.cpu().numpy()
    
    for merge_size in merge_sizes:
        print(f"  Trying merge size: {merge_size}x{merge_size}x{merge_size}")
        
        num_merge_d = D // merge_size
        num_merge_h = H // merge_size
        num_merge_w = W // merge_size
        total_regions = num_merge_d * num_merge_h * num_merge_w
        
        print(f"    Total regions: {total_regions}")
        
        t8 = time.time()
        all_variance_sims = compute_patch_variance_similarity_batch(pet_tensor, merge_size)
        t9 = time.time()
        print(f"    Batch variance computation time: {t9-t8:.4f}s")
        
        t10 = time.time()
        region_covered = np.zeros(total_regions, dtype=bool)
        
        idx = 0
        for md in range(num_merge_d):
            for mh in range(num_merge_h):
                for mw in range(num_merge_w):
                    sd = md * merge_size
                    ed = sd + merge_size
                    sh = mh * merge_size
                    eh = sh + merge_size
                    sw = mw * merge_size
                    ew = sw + merge_size
                    region_covered[idx] = processed_np[sd:ed, sh:eh, sw:ew].any()
                    idx += 1
        
        t11 = time.time()
        print(f"    Coverage precomputation time: {t11-t10:.4f}s")
        
        t12 = time.time()
        size_merged_count = 0
        
        uncovered_indices = np.where(~region_covered)[0]
        
        w_vals = uncovered_indices % num_merge_w
        uncovered_indices = uncovered_indices // num_merge_w
        h_vals = uncovered_indices % num_merge_h
        d_vals = uncovered_indices // num_merge_h
        
        for i in range(len(uncovered_indices)):
            md = d_vals[i]
            mh = h_vals[i]
            mw = w_vals[i]
            
            merge_start_d = md * merge_size
            merge_end_d = (md + 1) * merge_size
            merge_start_h = mh * merge_size
            merge_end_h = (mh + 1) * merge_size
            merge_start_w = mw * merge_size
            merge_end_w = (mw + 1) * merge_size
            
            flat_idx = md * (num_merge_h * num_merge_w) + mh * num_merge_w + mw
            merge_variance_sim = all_variance_sims[flat_idx].item()
            
            token_info = {
                'start_d': merge_start_d,
                'end_d': merge_end_d,
                'start_h': merge_start_h,
                'end_h': merge_end_h,
                'start_w': merge_start_w,
                'end_w': merge_end_w,
                'patch_size': merge_size,
                'variance_similarity': merge_variance_sim,
                'layer_id': 0,
                'core_feature': False
            }
            tokens.append(token_info)
            
            processed_np[merge_start_d:merge_end_d, merge_start_h:merge_end_h, merge_start_w:merge_end_w] = True
            
            size_merged_count += 1
            merged_count += 1
        
        processed_mask = torch.from_numpy(processed_np).to(device)
        
        t13 = time.time()
        print(f"    Uncovered region processing time: {t13-t12:.4f}s")
        print(f"    {merge_size}x{merge_size}x{merge_size}: merged {size_merged_count} regions")
    
    print(f"Successfully processed {merged_count} regions")
    
    final_unprocessed = torch.where(~processed_mask)
    if len(final_unprocessed[0]) == 0:
        print("Full coverage verification passed")
    else:
        print(f"Warning: {len(final_unprocessed[0])} voxels still unprocessed")
    
    print(f"\nFinal Results:")
    print(f"  Total token count: {len(tokens)}")
    print(f"  Core feature count: {core_patches_count}")
    print(f"  Actual threshold used: {current_threshold:.2f}")
    print(f"  Total time: {time.time() - t0:.4f}s")
    
    return tokens, current_threshold


def load_pet_data(data_dir, case_id, dataset, split):
    """Load PET data"""
    print(f"Loading PET data: {case_id} (dataset: {dataset}, split: {split})")
    
    pet_path = os.path.join(data_dir, split, "pet", f"{case_id}.nii.gz")
    
    if not os.path.exists(pet_path):
        print(f"PET data file not found: {pet_path}")
        return None
    
    pet_img = tio.ScalarImage(pet_path)
    pet_tensor = pet_img.data.float()
    
    if pet_tensor.dim() == 4:
        pet_tensor = pet_tensor.unsqueeze(1)
    
    print(f"PET shape: {pet_tensor.shape}")
    
    return pet_tensor


def process_dataset(data_dir, dataset, output_dir, threshold=0.3, split='all', max_cases=None, min_core_tokens=5, min_patch_size=2):
    """Process entire dataset to generate LVS-TAA token information"""
    print(f"Processing dataset: {dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
    
    all_case_ids = []
    
    if split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [split]
    
    for split_name in splits:
        case_dir = os.path.join(data_dir, split_name, "pet")
        
        if os.path.exists(case_dir):
            case_files = [f for f in os.listdir(case_dir) if f.endswith('.nii.gz')]
            split_case_ids = [f.replace('.nii.gz', '') for f in case_files]
            all_case_ids.extend([(case_id, split_name) for case_id in split_case_ids])
            print(f"Found {split_name} split: {len(split_case_ids)} cases")
        else:
            print(f"Warning: {case_dir} does not exist")
    
    print(f"Total cases found: {len(all_case_ids)}")
    
    if max_cases is not None:
        all_case_ids = all_case_ids[:max_cases]
        print(f"Limited to: {len(all_case_ids)} cases")
    
    stats = {
        'total_cases': len(all_case_ids),
        'processed_cases': 0,
        'failed_cases': 0,
        'total_tokens': 0,
        'core_tokens': 0,
        'normal_tokens': 0,
        'patch_size_stats': {},
        'threshold_adjustments': {},
        'dataset': dataset,
        'initial_threshold': threshold,
        'min_core_tokens': min_core_tokens,
        'min_patch_size': min_patch_size
    }
    
    for case_id, split_name in tqdm(all_case_ids, desc=f"Processing {dataset}"):
        print(f"\nProcessing case: {case_id} (split: {split_name})")
        
        split_output_dir = os.path.join(output_dir, split_name)
        output_file = os.path.join(split_output_dir, f"{case_id}_token_info.npz")
        
        if os.path.exists(output_file):
            print(f"Skipping {case_id}, token info already exists")
            stats['processed_cases'] += 1
            continue
        
        pet_tensor = load_pet_data(data_dir, case_id, dataset, split_name)
        if pet_tensor is None:
            print(f"Skipping case {case_id}, data loading failed")
            stats['failed_cases'] += 1
            continue
        
        try:
            tokens, actual_threshold = direct_pet_tokenization(
                pet_tensor, 
                initial_threshold=threshold,
                max_patch_size=64,
                min_patch_size=min_patch_size,
                min_core_tokens=min_core_tokens
            )
            
            stats['threshold_adjustments'][case_id] = {
                'initial_threshold': threshold,
                'actual_threshold': actual_threshold,
                'threshold_increased': actual_threshold > threshold
            }
            
            core_tokens = [t for t in tokens if t.get('core_feature', False)]
            normal_tokens = [t for t in tokens if not t.get('core_feature', False)]
            
            patch_size_stats = {}
            for token in tokens:
                size = token['patch_size']
                if size not in patch_size_stats:
                    patch_size_stats[size] = 0
                patch_size_stats[size] += 1
            
            print(f"Token Statistics:")
            print(f"  Total tokens: {len(tokens)}")
            print(f"  Core features: {len(core_tokens)}")
            print(f"  Normal features: {len(normal_tokens)}")
            print(f"  Patch size distribution: {patch_size_stats}")
            print(f"  Threshold info: initial={threshold:.2f}, actual={actual_threshold:.2f}")
            
            stats['processed_cases'] += 1
            stats['total_tokens'] += len(tokens)
            stats['core_tokens'] += len(core_tokens)
            stats['normal_tokens'] += len(normal_tokens)
            
            for size, count in patch_size_stats.items():
                if size not in stats['patch_size_stats']:
                    stats['patch_size_stats'][size] = 0
                stats['patch_size_stats'][size] += count
            
            save_data = {
                'image_regions': np.array([[t['start_d'], t['end_d'], t['start_h'], t['end_h'], t['start_w'], t['end_w']] for t in tokens]),
                'patch_centers': np.array([[(t['start_d'] + t['end_d']) // 2, (t['start_h'] + t['end_h']) // 2, (t['start_w'] + t['end_w']) // 2] for t in tokens]),
                'patch_sizes': np.array([t['patch_size'] for t in tokens]),
                'variance_similarities': np.array([t['variance_similarity'] for t in tokens]),
                'core_features': np.array([t.get('core_feature', False) for t in tokens]),
                'original_shape': np.array(pet_tensor.shape[2:]),
                'threshold_info': {
                    'initial_threshold': threshold,
                    'actual_threshold': actual_threshold,
                    'threshold_increased': actual_threshold > threshold
                }
            }
            np.savez_compressed(output_file, **save_data)
            
            print(f"Saved token info to: {output_file}")
            
        except Exception as e:
            print(f"Error processing case {case_id}: {e}")
            stats['failed_cases'] += 1
            continue
    
    stats_file = os.path.join(output_dir, f"{dataset}_core{min_core_tokens}_patch{min_patch_size}_direct_pet_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset processing completed!")
    print(f"Total cases: {stats['total_cases']}")
    print(f"Successfully processed: {stats['processed_cases']}")
    print(f"Failed cases: {stats['failed_cases']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Core feature tokens: {stats['core_tokens']}")
    print(f"Normal feature tokens: {stats['normal_tokens']}")
    print(f"Patch size distribution: {stats['patch_size_stats']}")
    
    print(f"Results saved to: {output_dir}")
    print(f"Statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='LVS-TAA Token Generator: Local Variance Similarity-based Token Area Allocation')
    
    parser.add_argument('--dataset', type=str, default='both', choices=['Hecktor', 'AutoPet', 'PSMA', 'both'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, 
                       default='',
                       help='Dataset directory')
    parser.add_argument('--output_dir', type=str, 
                       default='',
                       help='Output directory for token information')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val', 'test', 'all'],
                       help='Which split to process')
    parser.add_argument('--max_cases', type=int, default=None,
                       help='Maximum number of cases to process (for testing)')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Initial threshold for core feature identification')
    parser.add_argument('--min_core_tokens', type=int, default=5,
                       help='Minimum number of core tokens required')
    parser.add_argument('--min_patch_size', type=int, default=2, choices=[2, 4, 8],
                       help='Minimum patch size for core token extraction')
    
    args = parser.parse_args()

    # Dataset configuration
    datasets_config = {
        'Hecktor': '',
        'AutoPet': '',
        'PSMA': ''
    }
    
    if args.dataset == 'both':
        target_datasets = list(datasets_config.keys())
    else:
        target_datasets = [args.dataset]
    
    for dataset_name in target_datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        data_dir = datasets_config[dataset_name]
        dataset_output_dir = os.path.join(args.output_dir, f"{dataset_name}_core{args.min_core_tokens}_patch{args.min_patch_size}")
        
        process_dataset(
            data_dir=data_dir,
            dataset=dataset_name,
            output_dir=dataset_output_dir,
            threshold=args.threshold,
            split=args.split,
            max_cases=args.max_cases,
            min_core_tokens=args.min_core_tokens,
            min_patch_size=args.min_patch_size
        )
    
    print(f"\n{'='*60}")
    print("All datasets processing completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

