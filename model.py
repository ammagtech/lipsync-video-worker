# Copyright Alibaba Inc. All Rights Reserved.

import os
import torch
import torch.nn as nn
from typing import Optional, List, Tuple


class FantasyTalkingAudioConditionModel(nn.Module):
    """
    FantasyTalking Audio Condition Model for lip-sync video generation.
    This model processes audio features and integrates them with the video generation pipeline.
    """
    
    def __init__(self, dit_model, audio_dim: int = 768, hidden_dim: int = 2048):
        super().__init__()
        self.dit_model = dit_model
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        
        # Audio projection layers
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, audio_dim)
        )
        
        # Audio context processor
        self.audio_context_processor = nn.MultiheadAttention(
            embed_dim=audio_dim,
            num_heads=8,
            batch_first=True
        )
        
    def load_audio_processor(self, checkpoint_path: str, dit_model):
        """Load the audio processor weights from checkpoint."""
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"⚠️ Checkpoint file not found: {checkpoint_path}")
            print("Using randomly initialized weights for FantasyTalking model.")
            return False

        try:
            print(f"Loading FantasyTalking weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Extract audio-related weights from checkpoint
            audio_state_dict = {}
            for key, value in checkpoint.items():
                if 'audio' in key.lower() or 'proj' in key.lower():
                    audio_state_dict[key] = value

            # Load the state dict with strict=False to allow partial loading
            missing_keys, unexpected_keys = self.load_state_dict(audio_state_dict, strict=False)

            if missing_keys:
                print(f"⚠️ Missing keys in checkpoint: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys in checkpoint: {len(unexpected_keys)} keys")

            print(f"✅ Successfully loaded audio processor from {checkpoint_path}")
            return True

        except Exception as e:
            print(f"❌ Error loading audio processor weights: {e}")
            print("Using randomly initialized weights.")
            return False
    
    def get_proj_fea(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Project audio features to the required dimension.
        
        Args:
            audio_features: Input audio features [batch, seq_len, audio_dim]
            
        Returns:
            Projected audio features [batch, seq_len, audio_dim]
        """
        return self.audio_proj(audio_features)
    
    def split_audio_sequence(self, seq_length: int, num_frames: int) -> List[Tuple[int, int]]:
        """
        Split audio sequence into chunks that align with video frames.
        
        Args:
            seq_length: Length of the audio sequence
            num_frames: Number of video frames to generate
            
        Returns:
            List of (start, end) indices for audio chunks
        """
        chunk_size = seq_length // num_frames
        ranges = []
        
        for i in range(num_frames):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, seq_length)
            ranges.append((start, end))
            
        return ranges
    
    def split_tensor_with_padding(
        self, 
        tensor: torch.Tensor, 
        ranges: List[Tuple[int, int]], 
        expand_length: int = 4
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Split tensor into chunks with padding and context.
        
        Args:
            tensor: Input tensor [batch, seq_len, dim]
            ranges: List of (start, end) indices
            expand_length: Additional context length
            
        Returns:
            Tuple of (split_tensor, context_lengths)
        """
        batch_size, seq_len, dim = tensor.shape
        chunks = []
        context_lens = []
        
        for start, end in ranges:
            # Add context padding
            padded_start = max(0, start - expand_length)
            padded_end = min(seq_len, end + expand_length)
            
            chunk = tensor[:, padded_start:padded_end, :]
            chunks.append(chunk)
            context_lens.append(chunk.shape[1])
        
        # Pad all chunks to the same length
        max_len = max(context_lens)
        padded_chunks = []
        
        for chunk in chunks:
            if chunk.shape[1] < max_len:
                padding = torch.zeros(
                    batch_size, max_len - chunk.shape[1], dim,
                    device=chunk.device, dtype=chunk.dtype
                )
                chunk = torch.cat([chunk, padding], dim=1)
            padded_chunks.append(chunk)
        
        return torch.stack(padded_chunks, dim=1), context_lens
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for audio conditioning.
        
        Args:
            audio_features: Input audio features
            
        Returns:
            Processed audio features for conditioning
        """
        # Project audio features
        projected = self.get_proj_fea(audio_features)
        
        # Apply attention for context modeling
        attended, _ = self.audio_context_processor(projected, projected, projected)
        
        return attended + projected  # Residual connection
