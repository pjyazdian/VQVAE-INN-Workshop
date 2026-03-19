"""
Data loader for 20-frame window sampling from Salsa dance pairs dataset.
Follows the original DataPreprocessor pipeline to extract windows directly from raw LMDB.
"""

import os
import sys
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import lmdb
import pyarrow
import pickle
from typing import Tuple, Optional
from tqdm import tqdm
from .relationship_features import salsa_to_interhuman, extract_interhuman_relationship_features

# Debug flag for detailed verification output (set to True for debugging)
DEBUG = False

# Local-only module usage (no external repo imports)


class MotionWindowDataset(Dataset):
    """
    Dataset for 20-frame motion windows from Salsa dance pairs.
    Loads both leader and follower motions and treats them equally (joint training).
    Can also extract relationship features between leader and follower when train_relationship=True.
    """
    
    def __init__(
        self,
        args,
        lmdb_dir: str,
        window_size: int = 20,
        stride: int = 10,
        use_both_roles: bool = True,
        cache_dir: Optional[str] = None,
        normalize: bool = True,
        representation_type: str = 'humanml3d',
    ):
        """
        Initialize the dataset.
        
        Args:
            args: Arguments object (from existing codebase)
            lmdb_dir: Path to LMDB database directory
            window_size: Size of motion window (default: 20 frames)
            stride: Stride for window sampling (default: 10 frames)
            use_both_roles: If True, use both leader and follower motions (joint training)
            cache_dir: Optional cache directory for preprocessed windows
            normalize: If True, normalize data using precomputed mean/std (default: True)
            representation_type: Type of representation to use:
                - 'humanml3d': HumanML3D representation (263 dims) - default
                - 'interhuman': InterHuman representation (262 dims per person)
                - 'relationship': Relationship features (4 dims: [w, z, x, z] - quaternion components + position)
        """
        self.args = args
        self.lmdb_dir = lmdb_dir
        self.window_size = window_size
        self.stride = stride
        self.use_both_roles = use_both_roles
        self.normalize = normalize
        self.representation_type = representation_type
        
        # Validate representation type
        valid_types = ['humanml3d', 'interhuman', 'relationship']
        if representation_type not in valid_types:
            raise ValueError(f"representation_type must be one of {valid_types}, got {representation_type}")
        
        # Determine cache directory based on representation type
        # For 'interhuman' and 'relationship', use the same cache (both store in same dict)
        if cache_dir is None:
            if representation_type in ['interhuman', 'relationship']:
                # Use 'interhuman' as the shared cache name for both types
                cache_dir = lmdb_dir + "_interhuman_20frames_cache"
            else:
                cache_dir = lmdb_dir + f"_{representation_type}_20frames_cache"
        self.cache_dir = cache_dir
        
        # Feature dimensions for each representation type
        if representation_type == 'humanml3d':
            self.feature_dim = 263
        elif representation_type == 'interhuman':
            self.feature_dim = 262  # After process_motion_interhuman (reduces length by 1)
        elif representation_type == 'relationship':
            self.feature_dim = 4  # [w, z, x, z] - quaternion components [w, z] + position [x, z] (matches in2IN's losses.py line 79)
        
        # Check if cache exists, if not create it
        if not os.path.exists(self.cache_dir):
            print(f"Cache not found. Creating cache at {self.cache_dir}...")
            self._create_cache()
        else:
            print(f"Loading cached windows from {self.cache_dir}...")
        
        # Load cached windows
        self.lmdb_env = lmdb.open(self.cache_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_cache_entries = txn.stat()["entries"]
        
        # For interhuman with use_both_roles=True, double the dataset length (leader + follower)
        if self.representation_type == 'interhuman' and self.use_both_roles:
            self.n_samples = self.n_cache_entries * 2
        else:
            self.n_samples = self.n_cache_entries
        
        print(f"Loaded {self.n_cache_entries} motion windows from cache.")
        if self.representation_type == 'interhuman' and self.use_both_roles:
            print(f"Using both leader and follower: {self.n_samples} total training samples.")
        
        # Load normalization statistics
        # For 'interhuman' and 'relationship', we store separate stats for each type
        self.mean = None
        self.std = None
        self.epsilon = 1e-8  # Small value to avoid division by zero
        
        if self.normalize:
            # For interhuman/relationship shared cache, load stats specific to representation_type
            if self.representation_type in ['interhuman', 'relationship']:
                stats_path = os.path.join(self.cache_dir, f'normalization_stats_{self.representation_type}.pkl')
            else:
                stats_path = os.path.join(self.cache_dir, 'normalization_stats.pkl')
            
            if os.path.exists(stats_path):
                print(f"Loading normalization statistics from {stats_path}...")
                with open(stats_path, 'rb') as f:
                    stats = pickle.load(f)
                    self.mean = torch.from_numpy(stats['mean']).float()
                    self.std = torch.from_numpy(stats['std']).float()
                    if DEBUG:
                        print(f"  Mean shape: {self.mean.shape}, Std shape: {self.std.shape}")
                        print(f"  Mean range: [{self.mean.min():.6f}, {self.mean.max():.6f}]")
                        print(f"  Std range: [{self.std.min():.6f}, {self.std.max():.6f}]")
                    # Check for constant dimensions
                    constant_dims = (self.std < self.epsilon).sum().item()
                    if constant_dims > 0:
                        print(f"  Warning: {constant_dims} dimensions have std < {self.epsilon} (will not be normalized)")
            else:
                print(f"Warning: Normalization requested but stats file not found at {stats_path}")
                print(f"  Set normalize=False or recreate cache to compute statistics.")
    
    def _create_cache(self):
        """
        Create cache of 20-frame windows directly from raw LMDB.
        Also computes normalization statistics from all original frames (not windows).
        Follows the same pipeline as DataPreprocessor._sample_from_clip_pair.
        If train_relationship=True, extracts relationship features instead of individual motions.
        """
        # Open raw LMDB (same as DataPreprocessor does)
        src_lmdb_env = lmdb.open(self.lmdb_dir, readonly=True, lock=False)
        
        # Create cache LMDB
        map_size = 1024 * 30  # 30 GB
        map_size <<= 20  # Convert to bytes
        cache_env = lmdb.open(self.cache_dir, map_size=map_size)
        
        window_idx = 0
        skeleton_resampling_fps = 20  # Same as original pipeline
        
        print(f"Creating 20-frame cache from raw LMDB...")
        if self.representation_type in ['interhuman', 'relationship']:
            print("Using shared cache for InterHuman data (both motions and relationship features stored in dictionaries)")
        
        # Collect all frames for normalization statistics (not windows, to avoid double-counting)
        # For interhuman/relationship shared cache, we collect both motion and relationship frames separately
        all_motion_frames = []  # For InterHuman motions (262 dims)
        all_relationship_frames = []  # For relationship features (4 dims)
        all_frames = []  # For other representation types (humanml3d)
        
        # Follow the same structure as DataPreprocessor.run()
        src_txn = src_lmdb_env.begin(write=False)
        total_count = src_txn.stat()['entries']
        
        cursor = src_txn.cursor()
        counter = 0
        
        for key, value in tqdm(cursor, total=total_count, desc="Processing videos"):
            try:
                video = pyarrow.deserialize(value)
                vid = video['vid']
                clips = video['clips']
                counter += 1
                # if counter > 3:
                #     break    
                for clip_idx, clip in enumerate(clips):
                    # Extract windows from this clip (following _sample_from_clip_pair logic)
                    windows_from_clip = self._extract_windows_from_clip(clip)
                    
                    # Save all windows from this clip
                    for window in windows_from_clip:
                        with cache_env.begin(write=True) as txn:
                            key = "{:010}".format(window_idx).encode("ascii")
                            value = pyarrow.serialize(window).to_buffer()
                            txn.put(key, value)
                            window_idx += 1
                    
                    # Collect all frames from this clip for normalization stats
                    # Extract raw motion data (not windows) to avoid double-counting overlapping frames
                    # For interhuman/relationship, we always extract both motions and relationship features
                    if self.representation_type in ['interhuman', 'relationship']:
                        # For InterHuman/relationship shared cache, extract both motions and relationship features
                        # local vendored helpers
                        
                        clip_keypoints3d_L = clip['keypoints3d_L']
                        clip_rotmat_L = clip['rotmat_L']
                        clip_keypoints3d_F = clip['keypoints3d_F']
                        clip_rotmat_F = clip['rotmat_F']
                        
                        # Convert to numpy if needed
                        if isinstance(clip_keypoints3d_L, torch.Tensor):
                            clip_keypoints3d_L = clip_keypoints3d_L.cpu().numpy()
                        if isinstance(clip_rotmat_L, torch.Tensor):
                            clip_rotmat_L = clip_rotmat_L.cpu().numpy()
                        if isinstance(clip_keypoints3d_F, torch.Tensor):
                            clip_keypoints3d_F = clip_keypoints3d_F.cpu().numpy()
                        if isinstance(clip_rotmat_F, torch.Tensor):
                            clip_rotmat_F = clip_rotmat_F.cpu().numpy()
                        
                        # Both must have same sequence length
                        seq_len_L = clip_keypoints3d_L.shape[0]
                        seq_len_F = clip_keypoints3d_F.shape[0]
                        seq_len = min(seq_len_L, seq_len_F)
                        
                        # Truncate to same length if needed
                        if seq_len_L != seq_len_F:
                            clip_keypoints3d_L = clip_keypoints3d_L[:seq_len]
                            clip_rotmat_L = clip_rotmat_L[:seq_len]
                            clip_keypoints3d_F = clip_keypoints3d_F[:seq_len]
                            clip_rotmat_F = clip_rotmat_F[:seq_len]
                        
                        # Convert to InterHuman representation
                        # Apply +90° rotation to reverse the -90° preprocessing rotation (salsa_utils.py lines 797-803)
                        motion_L_ih, root_quat_init_L, root_pos_init_L = salsa_to_interhuman(
                            clip_keypoints3d_L, clip_rotmat_L, rotation_deg=90
                        )  # (seq_len-1, 262), (seq_len-1, 4), (seq_len-1, 3)
                        
                        motion_F_ih, root_quat_init_F, root_pos_init_F = salsa_to_interhuman(
                            clip_keypoints3d_F, clip_rotmat_F, rotation_deg=90
                        )  # (seq_len-1, 262), (seq_len-1, 4), (seq_len-1, 3)
                        
                        # Collect motion frames for normalization stats
                        all_motion_frames.append(motion_L_ih)  # (seq_len-1, 262)
                        if self.use_both_roles:
                            all_motion_frames.append(motion_F_ih)  # (seq_len-1, 262)
                        
                        # Extract relationship features for normalization stats
                        root_quat_init_L_frame0 = root_quat_init_L[0] if root_quat_init_L.ndim > 1 else root_quat_init_L
                        root_pos_init_L_frame0 = root_pos_init_L[0] if root_pos_init_L.ndim > 1 else root_pos_init_L
                        root_quat_init_F_frame0 = root_quat_init_F[0] if root_quat_init_F.ndim > 1 else root_quat_init_F
                        root_pos_init_F_frame0 = root_pos_init_F[0] if root_pos_init_F.ndim > 1 else root_pos_init_F
                        
                        # CRITICAL: Pass copies of motions because rigid_transform modifies input arrays in place!
                        relationship_features = extract_interhuman_relationship_features(
                            motion_L_ih.copy(), motion_F_ih.copy(),  # Make copies to avoid in-place modification
                            root_quat_init_L_frame0, root_pos_init_L_frame0,
                            root_quat_init_F_frame0, root_pos_init_F_frame0,
                            return_aligned_follower=False
                        )  # (seq_len-1, 4) - [w, z, x, z]
                        
                        all_relationship_frames.append(relationship_features)
                    else:  # humanml3d
                        # For regular motions, use HumanML3D representation
                        clip_HM3D_joint_vec_L = clip['HML3D_joints_vec_L']
                        clip_HM3D_joint_vec_F = clip['HML3D_joints_vec_F']
                        
                        # Convert to numpy if needed
                        if isinstance(clip_HM3D_joint_vec_L, torch.Tensor):
                            clip_HM3D_joint_vec_L = clip_HM3D_joint_vec_L.cpu().numpy()
                        if isinstance(clip_HM3D_joint_vec_F, torch.Tensor):
                            clip_HM3D_joint_vec_F = clip_HM3D_joint_vec_F.cpu().numpy()
                        
                        # Collect individual motions for normalization statistics
                        if self.use_both_roles:
                            all_frames.append(clip_HM3D_joint_vec_L)  # (seq_len, 263)
                            all_frames.append(clip_HM3D_joint_vec_F)  # (seq_len, 263)
                        else:
                            all_frames.append(clip_HM3D_joint_vec_L)  # (seq_len, 263)
                    
                    # counter += 1
                    # if counter > 3:
                    #     break
                    if counter % 10 == 0:
                        print(f"Processed {counter} clips, created {window_idx} windows...")
            
            except Exception as e:
                print(f"Error processing video {counter}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        cache_env.sync()
        cache_env.close()
        src_lmdb_env.close()
        print(f"Cache creation complete. Total windows: {window_idx}")
        
        # Compute normalization statistics from all original frames
        print(f"\nComputing normalization statistics from all original frames...")
        
        # For interhuman/relationship shared cache, compute separate stats for each type
        if self.representation_type in ['interhuman', 'relationship']:
            # Compute stats for InterHuman motions (262 dims)
            if len(all_motion_frames) > 0:
                print(f"  Computing stats for InterHuman motions...")
                if DEBUG:
                    print(f"    Collected {len(all_motion_frames)} motion sequences")
                all_motion_array = np.concatenate(all_motion_frames, axis=0)
                total_motion_frames = all_motion_array.shape[0]
                if DEBUG:
                    print(f"    Total frames: {total_motion_frames}")
                    print(f"    Frame shape: {all_motion_array.shape[1]} (should be 262)")
                
                motion_mean = np.mean(all_motion_array, axis=0)  # (262,)
                motion_std = np.std(all_motion_array, axis=0)   # (262,)
                
                epsilon = 1e-8
                constant_dims = (motion_std < epsilon).sum()
                if constant_dims > 0:
                    print(f"    Warning: {constant_dims} dimensions have std < {epsilon}")
                    motion_std[motion_std < epsilon] = 1.0
                
                stats_path = os.path.join(self.cache_dir, 'normalization_stats_interhuman.pkl')
                stats = {
                    'mean': motion_mean,
                    'std': motion_std,
                    'total_frames': total_motion_frames,
                    'total_sequences': len(all_motion_frames),
                    'window_size': self.window_size,
                    'stride': self.stride,
                    'use_both_roles': self.use_both_roles,
                    'representation_type': 'interhuman',
                }
                with open(stats_path, 'wb') as f:
                    pickle.dump(stats, f)
                print(f"    Saved to: {stats_path}")
            
            # Compute stats for relationship features (4 dims)
            if len(all_relationship_frames) > 0:
                print(f"  Computing stats for relationship features...")
                if DEBUG:
                    print(f"    Collected {len(all_relationship_frames)} relationship sequences")
                all_relationship_array = np.concatenate(all_relationship_frames, axis=0)
                total_rel_frames = all_relationship_array.shape[0]
                if DEBUG:
                    print(f"    Total frames: {total_rel_frames}")
                    print(f"    Frame shape: {all_relationship_array.shape[1]} (should be 4)")
                
                rel_mean = np.mean(all_relationship_array, axis=0)  # (4,)
                rel_std = np.std(all_relationship_array, axis=0)   # (4,)
                
                epsilon = 1e-8
                constant_dims = (rel_std < epsilon).sum()
                if constant_dims > 0:
                    print(f"    Warning: {constant_dims} dimensions have std < {epsilon}")
                    rel_std[rel_std < epsilon] = 1.0
                
                stats_path = os.path.join(self.cache_dir, 'normalization_stats_relationship.pkl')
                stats = {
                    'mean': rel_mean,
                    'std': rel_std,
                    'total_frames': total_rel_frames,
                    'total_sequences': len(all_relationship_frames),
                    'window_size': self.window_size,
                    'stride': self.stride,
                    'use_both_roles': self.use_both_roles,
                    'representation_type': 'relationship',
                }
                with open(stats_path, 'wb') as f:
                    pickle.dump(stats, f)
                print(f"    Saved to: {stats_path}")
        
        # For humanml3d, compute stats from all_frames
        if len(all_frames) > 0:
            # For humanml3d, use single stats file
            if DEBUG:
                print(f"  Collected {len(all_frames)} motion sequences")
            expected_dim = self.feature_dim
            
            all_frames_array = np.concatenate(all_frames, axis=0)
            total_frames = all_frames_array.shape[0]
            if DEBUG:
                print(f"  Total frames: {total_frames}")
                print(f"  Frame shape: {all_frames_array.shape[1]} (should be {expected_dim})")
            
            mean = np.mean(all_frames_array, axis=0)  # (feature_dim,)
            std = np.std(all_frames_array, axis=0)   # (feature_dim,)
            
            epsilon = 1e-8
            constant_dims = (std < epsilon).sum()
            if constant_dims > 0:
                print(f"  Warning: {constant_dims} dimensions have std < {epsilon}")
                std[std < epsilon] = 1.0
            
            stats_path = os.path.join(self.cache_dir, 'normalization_stats.pkl')
            stats = {
                'mean': mean,
                'std': std,
                'total_frames': total_frames,
                'total_sequences': len(all_frames),
                'window_size': self.window_size,
                'stride': self.stride,
                'use_both_roles': self.use_both_roles,
                'representation_type': self.representation_type,
            }
            with open(stats_path, 'wb') as f:
                pickle.dump(stats, f)
            print(f"  Saved normalization statistics to: {stats_path}")
            print(f"  Statistics computed from {total_frames} frames across {len(all_frames)} sequences")
    
    def _extract_windows_from_clip(self, clip: dict):
        """
        Extract 20-frame windows from a clip, following DataPreprocessor._sample_from_clip_pair.
        
        Args:
            clip: Clip dictionary from raw LMDB (same format as in DataPreprocessor)
        
        Returns:
            List of windows:
            - If representation_type='humanml3d': each is numpy array of shape (20, 263)
            - If representation_type='interhuman': each is dict with 'leader_motion', 'follower_motion', etc.
            - If representation_type='relationship': each is dict with 'leader_motion', 'follower_motion', 'relationship_features', etc.
        """
        windows = []
        
        if self.representation_type in ['relationship', 'interhuman']:
            # Extract InterHuman representation for both leader and follower
            # Store all data in dictionary format for later use
            # local vendored helpers
            
            # Load keypoints3d and rotmat from clip
            clip_keypoints3d_L = clip['keypoints3d_L']
            clip_rotmat_L = clip['rotmat_L']
            clip_keypoints3d_F = clip['keypoints3d_F']
            clip_rotmat_F = clip['rotmat_F']
            
            # Convert to numpy if needed
            if isinstance(clip_keypoints3d_L, torch.Tensor):
                clip_keypoints3d_L = clip_keypoints3d_L.cpu().numpy()
            if isinstance(clip_rotmat_L, torch.Tensor):
                clip_rotmat_L = clip_rotmat_L.cpu().numpy()
            if isinstance(clip_keypoints3d_F, torch.Tensor):
                clip_keypoints3d_F = clip_keypoints3d_F.cpu().numpy()
            if isinstance(clip_rotmat_F, torch.Tensor):
                clip_rotmat_F = clip_rotmat_F.cpu().numpy()
            
            # Both must have same sequence length
            seq_len_L = clip_keypoints3d_L.shape[0]
            seq_len_F = clip_keypoints3d_F.shape[0]
            seq_len = min(seq_len_L, seq_len_F)
            
            # Skip if sequence is too short
            if seq_len < self.window_size:
                return windows
            
            # Truncate to same length if needed
            if seq_len_L != seq_len_F:
                clip_keypoints3d_L = clip_keypoints3d_L[:seq_len]
                clip_rotmat_L = clip_rotmat_L[:seq_len]
                clip_keypoints3d_F = clip_keypoints3d_F[:seq_len]
                clip_rotmat_F = clip_rotmat_F[:seq_len]
            
            # Create overlapping windows (same logic as original subdivision)
            num_windows = math.floor((seq_len - self.window_size) / self.stride) + 1
            
            for i in range(num_windows):
                start_idx = i * self.stride
                fin_idx = start_idx + self.window_size
                
                # Check bounds
                if fin_idx > seq_len:
                    continue
                
                # Extract paired windows from keypoints3d/rotmat
                window_keypoints3d_L = clip_keypoints3d_L[start_idx:fin_idx]  # (window_size, 22, 3)
                window_rotmat_L = clip_rotmat_L[start_idx:fin_idx]  # (window_size, 498)
                window_keypoints3d_F = clip_keypoints3d_F[start_idx:fin_idx]  # (window_size, 22, 3)
                window_rotmat_F = clip_rotmat_F[start_idx:fin_idx]  # (window_size, 498)
                
                # Ensure exactly window_size frames
                if (window_keypoints3d_L.shape[0] != self.window_size or 
                    window_keypoints3d_F.shape[0] != self.window_size):
                    continue
                
                # Convert each window to InterHuman representation (canonical frames)
                # Apply +90° rotation to reverse the -90° preprocessing rotation (salsa_utils.py lines 797-803)
                motion_L_ih, root_quat_init_L, root_pos_init_L = salsa_to_interhuman(
                    window_keypoints3d_L, window_rotmat_L, rotation_deg=90
                )  # (window_size-1, 262), (window_size-1, 4), (window_size-1, 3)
                
                motion_F_ih, root_quat_init_F, root_pos_init_F = salsa_to_interhuman(
                    window_keypoints3d_F, window_rotmat_F, rotation_deg=90
                )  # (window_size-1, 262), (window_size-1, 4), (window_size-1, 3)
                
                # Extract root_quat_init and root_pos_init from frame 0 (for rigid_transform)
                # These are from the canonicalization process (frame 0 only)
                root_quat_init_L_frame0 = root_quat_init_L[0] if root_quat_init_L.ndim > 1 else root_quat_init_L  # (4,)
                root_pos_init_L_frame0 = root_pos_init_L[0] if root_pos_init_L.ndim > 1 else root_pos_init_L  # (3,)
                root_quat_init_F_frame0 = root_quat_init_F[0] if root_quat_init_F.ndim > 1 else root_quat_init_F  # (4,)
                root_pos_init_F_frame0 = root_pos_init_F[0] if root_pos_init_F.ndim > 1 else root_pos_init_F  # (3,)
                
                # Extract temporal relationship features from InterHuman representation
                # Store both leader and follower as separate canonicalized motions (not aligned)
                # This allows training separate networks: one for single canonicalized motions, one for relationship features
                # At visualization/inference, we'll apply rigid_transform using frame 0's relationship features
                # Pass per-frame root quaternions for computing relationship features using same method as in2IN preprocessing
                # CRITICAL: Pass copies of motions because rigid_transform modifies input arrays in place!
                # ToDo: Make copies to avoid in-place modification for all the inputs.
                relationship_window = extract_interhuman_relationship_features(
                    motion_L_ih.copy(), motion_F_ih.copy(),  # Make copies to avoid in-place modification
                    root_quat_init_L_frame0, root_pos_init_L_frame0,
                    root_quat_init_F_frame0, root_pos_init_F_frame0,
                    root_quat_init_L_all=root_quat_init_L,  # (window_size-1, 4) - per-frame root quaternions
                    root_pos_init_L_all=root_pos_init_L,      # (window_size-1, 3) - per-frame root positions
                    root_quat_init_F_all=root_quat_init_F,    # (window_size-1, 4) - per-frame root quaternions
                    root_pos_init_F_all=root_pos_init_F,      # (window_size-1, 3) - per-frame root positions
                    return_aligned_follower=False
                )  # (window_size-1, 4) - [w, z, x, z] for each frame
                
                # Store all InterHuman-related data in dictionary format
                # Both motions stored as separate canonicalized motions (for single-motion network training)
                # Relationship features stored separately (for relationship network training)
                window_dict = {
                    'leader_motion': motion_L_ih.astype(np.float32),  # (window_size-1, 262) - canonicalized
                    'follower_motion': motion_F_ih.astype(np.float32),  # (window_size-1, 262) - canonicalized (NOT aligned)
                    'relationship_features': relationship_window.astype(np.float32),  # (window_size-1, 4)
                    'root_quat_init_L': root_quat_init_L_frame0.astype(np.float32),  # (4,)
                    'root_pos_init_L': root_pos_init_L_frame0.astype(np.float32),  # (3,)
                    'root_quat_init_F': root_quat_init_F_frame0.astype(np.float32),  # (4,)
                    'root_pos_init_F': root_pos_init_F_frame0.astype(np.float32),  # (3,)
                    'cache_version': 2  # Version marker for new format
                }
                
                windows.append(window_dict)
            # Note: Both 'interhuman' and 'relationship' now use the same extraction logic above
            # Both store dictionaries with all InterHuman data
        else:  # humanml3d
            # Extract HML3D data from clip (same as _sample_from_clip_pair)
            clip_HM3D_joint_vec_L = clip['HML3D_joints_vec_L']
            clip_HM3D_joint_vec_F = clip['HML3D_joints_vec_F']
            
            # Convert to numpy if needed
            if isinstance(clip_HM3D_joint_vec_L, torch.Tensor):
                clip_HM3D_joint_vec_L = clip_HM3D_joint_vec_L.cpu().numpy()
            if isinstance(clip_HM3D_joint_vec_F, torch.Tensor):
                clip_HM3D_joint_vec_F = clip_HM3D_joint_vec_F.cpu().numpy()
            
            # Process motions (same logic as original, but extract 20-frame windows)
            motions_to_process = []
            if self.use_both_roles:
                motions_to_process = [
                    ('L', clip_HM3D_joint_vec_L),
                    ('F', clip_HM3D_joint_vec_F)
                ]
            else:
                motions_to_process = [('L', clip_HM3D_joint_vec_L)]
            
            for role, motion in motions_to_process:
                seq_len = motion.shape[0]
                
                # Skip if sequence is too short
                if seq_len < self.window_size:
                    continue
                
                # Create overlapping windows (same logic as original subdivision)
                # num_subdivision = floor((seq_len - window_size) / stride) + 1
                num_windows = math.floor((seq_len - self.window_size) / self.stride) + 1
                
                for i in range(num_windows):
                    start_idx = i * self.stride
                    fin_idx = start_idx + self.window_size
                    
                    # Check bounds (same as original)
                    if fin_idx > seq_len:
                        continue
                    
                    window = motion[start_idx:fin_idx]  # (window_size, 263)
                    
                    # Ensure exactly window_size frames
                    if window.shape[0] != self.window_size:
                        continue
                    
                    windows.append(window)
        
        return windows
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a motion window at the specified index.
        
        Args:
            idx: Index of the sample
        
        Returns:
            motion: Motion tensor of shape:
            - (window_size, 263) if representation_type='humanml3d', normalized if normalize=True
            - (window_size-1, 262) if representation_type='interhuman', normalized if normalize=True
            - (window_size-1, 4) if representation_type='relationship', normalized if normalize=True
        """
        # For interhuman with use_both_roles=True, map dataset idx to cache idx and role
        if self.representation_type == 'interhuman' and self.use_both_roles:
            cache_idx = idx // 2  # Which cache entry (pair)
            role_idx = idx % 2     # 0 = leader, 1 = follower
        else:
            cache_idx = idx
            role_idx = 0
        
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:010}".format(cache_idx).encode("ascii")
            sample = txn.get(key)
            
            if sample is None:
                raise IndexError(f"Index {idx} (cache_idx={cache_idx}) out of range")
            
            cached_item = pyarrow.deserialize(sample)
            
            # Handle new dictionary format (cache_version 2) for InterHuman-related representations
            if isinstance(cached_item, dict) and 'cache_version' in cached_item:
                # New format: extract data based on representation_type
                if self.representation_type == 'interhuman':
                    # Return leader or follower motion based on role_idx
                    motion = cached_item['follower_motion'].copy() if role_idx == 1 else cached_item['leader_motion'].copy()
                elif self.representation_type == 'relationship':
                    motion = cached_item['relationship_features'].copy()
                else:
                    raise ValueError(f"Unexpected representation_type {self.representation_type} with dictionary cache format")
            else:
                # Old format: direct array (backward compatibility)
                motion = cached_item.copy()
            
            # Convert to torch tensor
            motion = torch.from_numpy(motion).float()
            
            # Ensure correct shape based on representation type
            if self.representation_type == 'humanml3d':
                expected_shape = (self.window_size, 263)
            elif self.representation_type == 'interhuman':
                expected_shape = (self.window_size - 1, 262)  # process_motion_interhuman reduces by 1
            elif self.representation_type == 'relationship':
                expected_shape = (self.window_size - 1, 4)  # Extracted from InterHuman (reduced by 1): [w, z, x, z]
            
            assert motion.shape == expected_shape, \
                f"Expected shape {expected_shape}, got {motion.shape}"
            
            # Apply normalization if enabled and stats are available
            if self.normalize and self.mean is not None and self.std is not None:
                # Normalize: (motion - mean) / (std + epsilon)
                # mean and std are shape (feature_dim,), will broadcast to (window_size, feature_dim)
                motion = (motion - self.mean) / (self.std + self.epsilon)
            
            return motion
    
    def get_pair_data(self, idx: int) -> dict:
        """
        Get a complete pair data dictionary (for visualization).
        Only works with InterHuman-related representations (interhuman, relationship).
        
        Args:
            idx: Index of the sample
        
        Returns:
            Dictionary with keys:
            - 'leader_motion': (window_size-1, 262) - canonicalized leader motion
            - 'follower_motion': (window_size-1, 262) - canonicalized follower motion
            - 'relationship_features': (window_size-1, 4) - relationship features [w, z, x, z]
            - 'root_quat_init_L': (4,) - leader root quaternion from frame 0
            - 'root_pos_init_L': (3,) - leader root position from frame 0
            - 'root_quat_init_F': (4,) - follower root quaternion from frame 0
            - 'root_pos_init_F': (3,) - follower root position from frame 0
        """
        if self.representation_type not in ['interhuman', 'relationship']:
            raise ValueError(f"get_pair_data only works with 'interhuman' or 'relationship' representation_type, got {self.representation_type}")
        
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:010}".format(idx).encode("ascii")
            sample = txn.get(key)
            
            if sample is None:
                raise IndexError(f"Index {idx} out of range")
            
            cached_item = pyarrow.deserialize(sample)
            
            if isinstance(cached_item, dict) and 'cache_version' in cached_item:
                # Return the full dictionary
                return cached_item
            else:
                raise ValueError(f"Cache format is not dictionary format. Please regenerate cache with new format.")


def create_dataloader(
    args,
    lmdb_dir: str,
    window_size: int = 20,
    stride: int = 10,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    use_both_roles: bool = True,
    normalize: bool = True,
    representation_type: str = 'humanml3d',
    cache_dir: Optional[str] = None,
):
    """
    Create a DataLoader for motion windows.
    
    Args:
        args: Arguments object
        lmdb_dir: Path to LMDB database
        window_size: Size of motion window (default: 20)
        stride: Stride for window sampling (default: 10)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes
        use_both_roles: Whether to use both leader and follower
        normalize: Whether to normalize data using precomputed mean/std (default: True)
        representation_type: Type of representation ('humanml3d', 'interhuman', 'relationship')
        cache_dir: Optional path to an existing cache LMDB directory
    
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    dataset = MotionWindowDataset(
        args=args,
        lmdb_dir=lmdb_dir,
        window_size=window_size,
        stride=stride,
        use_both_roles=use_both_roles,
        normalize=normalize,
        representation_type=representation_type,
        cache_dir=cache_dir,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    return dataloader

