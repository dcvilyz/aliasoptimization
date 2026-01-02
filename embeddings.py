"""
Embedding extraction for visual concepts and vocabulary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from config import Config, EmbeddingConfig


class VocabularyEmbeddings:
    """
    Pre-computed embeddings for all tokens in SAM3's vocabulary.
    
    This allows fast nearest-neighbor search to find tokens
    closest to a visual concept embedding.
    """
    
    def __init__(
        self,
        embeddings: torch.Tensor,  # [vocab_size, embed_dim]
        token_ids: List[int],  # Token IDs corresponding to embeddings
        embed_dim: int,
    ):
        self.embeddings = embeddings
        self.token_ids = token_ids
        self.embed_dim = embed_dim
        self.vocab_size = len(token_ids)
        
        # Normalize for cosine similarity
        self.embeddings_normalized = F.normalize(embeddings, dim=1)
        
    def find_nearest(
        self,
        query: torch.Tensor,  # [embed_dim] or [N, embed_dim]
        k: int = 100,
        exclude_ids: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find k nearest tokens to query embedding(s).
        
        Args:
            query: Query embedding(s)
            k: Number of neighbors to return
            exclude_ids: Token IDs to exclude from results
            
        Returns:
            token_ids: [k] or [N, k] nearest token IDs
            similarities: [k] or [N, k] cosine similarities
        """
        query = query.to(self.embeddings.device)
        
        if query.dim() == 1:
            query = query.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
            
        # Normalize query
        query_norm = F.normalize(query, dim=1)
        
        # Compute cosine similarities
        similarities = torch.mm(query_norm, self.embeddings_normalized.t())  # [N, vocab_size]
        
        # Mask excluded IDs
        if exclude_ids:
            for excl_id in exclude_ids:
                if excl_id in self.token_ids:
                    idx = self.token_ids.index(excl_id)
                    similarities[:, idx] = -float('inf')
        
        # Get top-k
        top_sims, top_indices = similarities.topk(k, dim=1)
        
        # Map back to token IDs
        top_token_ids = torch.tensor(
            [[self.token_ids[idx] for idx in row] for row in top_indices.tolist()],
            device=similarities.device
        )
        
        if squeeze:
            return top_token_ids.squeeze(0), top_sims.squeeze(0)
        return top_token_ids, top_sims
    
    def get_embedding(self, token_id: int) -> torch.Tensor:
        """Get embedding for a specific token."""
        if token_id in self.token_ids:
            idx = self.token_ids.index(token_id)
            return self.embeddings[idx]
        raise ValueError(f"Token ID {token_id} not in vocabulary")
    
    def get_embeddings(self, token_ids: List[int]) -> torch.Tensor:
        """Get embeddings for multiple tokens."""
        indices = [self.token_ids.index(tid) for tid in token_ids]
        return self.embeddings[indices]
    
    def save(self, path: str):
        """Save embeddings to disk."""
        torch.save({
            'embeddings': self.embeddings,
            'token_ids': self.token_ids,
            'embed_dim': self.embed_dim,
        }, path)
        
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'VocabularyEmbeddings':
        """Load embeddings from disk."""
        data = torch.load(path, map_location=device)
        return cls(
            embeddings=data['embeddings'],
            token_ids=data['token_ids'],
            embed_dim=data['embed_dim'],
        )


def build_vocabulary_embeddings(
    model,  # SAM3 model
    config: EmbeddingConfig,
    device: str = 'mps',
    use_cache: bool = True,
) -> VocabularyEmbeddings:
    """
    Build embeddings for all tokens in SAM3's vocabulary.
    
    This extracts the token embedding from the text encoder for each
    token in the vocabulary.
    """
    cache_path = Path(config.cache_path)
    
    # Try to load from cache
    if use_cache and cache_path.exists():
        print(f"Loading vocabulary embeddings from cache: {cache_path}")
        return VocabularyEmbeddings.load(str(cache_path), device=device)
    
    print("Building vocabulary embeddings...")
    
    # Get tokenizer and text encoder from model
    text_encoder = model.backbone.language_backbone
    tokenizer = text_encoder.tokenizer
    
    vocab_size = tokenizer.vocab_size
    embed_dim = text_encoder.encoder.width
    
    # We'll extract the token embedding directly
    token_embedding = text_encoder.encoder.token_embedding
    
    all_embeddings = []
    all_token_ids = []
    
    # Process in batches
    batch_size = config.batch_size
    
    for start_idx in tqdm(range(0, vocab_size, batch_size), desc="Building vocab embeddings"):
        end_idx = min(start_idx + batch_size, vocab_size)
        batch_ids = list(range(start_idx, end_idx))
        
        # Get raw token embeddings
        token_ids_tensor = torch.tensor(batch_ids, device=device)
        with torch.no_grad():
            embeddings = token_embedding(token_ids_tensor)  # [batch, embed_dim]
        
        all_embeddings.append(embeddings.cpu())
        all_token_ids.extend(batch_ids)
    
    # Concatenate
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    vocab_embeddings = VocabularyEmbeddings(
        embeddings=all_embeddings,
        token_ids=all_token_ids,
        embed_dim=embed_dim,
    )
    
    # Cache to disk
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_embeddings.save(str(cache_path))
        print(f"Saved vocabulary embeddings to: {cache_path}")
    
    return vocab_embeddings


def extract_visual_concept_embedding(
    model,  # SAM3 model
    images: torch.Tensor,  # [B, C, H, W]
    masks: List[torch.Tensor],  # List of [N_i, H, W] masks per image
    pool_type: str = 'masked_mean',
    device: str = 'mps',
) -> torch.Tensor:
    """
    Extract visual concept embedding from masked regions.
    
    This uses SAM3's vision encoder to extract features,
    then pools over the masked regions.
    
    Args:
        model: SAM3 model
        images: Batch of images
        masks: List of instance masks per image
        pool_type: How to pool features over mask
        
    Returns:
        Concept embedding [embed_dim]
    """
    images = images.to(device)
    
    with torch.no_grad():
        # Get vision features from backbone
        backbone_out = model.backbone.forward_image(images)
        
        # Use the main feature map
        # backbone_fpn contains multi-scale features
        # We'll use the highest resolution useful features
        features = backbone_out['backbone_fpn'][-1]  # [B, C, H, W]
        
    B, C, feat_H, feat_W = features.shape
    
    # Collect all masked features
    all_masked_features = []
    
    for b in range(B):
        if masks[b].shape[0] == 0:
            continue
            
        # Resize masks to feature map size
        batch_masks = masks[b].float().unsqueeze(1)  # [N, 1, H, W]
        batch_masks_resized = F.interpolate(
            batch_masks,
            size=(feat_H, feat_W),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, feat_H, feat_W]
        
        # Get features for this image
        img_features = features[b]  # [C, feat_H, feat_W]
        
        for mask in batch_masks_resized:
            mask_binary = (mask > 0.5).float()
            mask_sum = mask_binary.sum()
            
            if mask_sum > 0:
                if pool_type == 'masked_mean':
                    # Weighted average over mask
                    masked_feat = (img_features * mask_binary).sum(dim=[1, 2]) / mask_sum
                elif pool_type == 'masked_max':
                    # Max over masked region
                    masked_feat = (img_features * mask_binary).view(C, -1).max(dim=1)[0]
                else:
                    raise ValueError(f"Unknown pool_type: {pool_type}")
                    
                all_masked_features.append(masked_feat)
    
    if not all_masked_features:
        # No valid masks, return zero embedding
        return torch.zeros(C, device=device)
    
    # Stack and aggregate
    stacked = torch.stack(all_masked_features, dim=0)  # [N_total, C]
    
    # Aggregate across all instances (mean)
    concept_embedding = stacked.mean(dim=0)  # [C]
    
    return concept_embedding


def extract_concept_from_category(
    model,
    dataloader,
    pool_type: str = 'masked_mean',
    max_images: Optional[int] = None,
    device: str = 'mps',
) -> torch.Tensor:
    """
    Extract concept embedding from all images of a category.
    
    Args:
        model: SAM3 model
        dataloader: DataLoader yielding category images and masks
        pool_type: How to pool features over masks
        max_images: Maximum number of images to use (None for all)
        
    Returns:
        Aggregated concept embedding [embed_dim]
    """
    all_embeddings = []
    num_processed = 0
    
    model.eval()
    
    for batch in tqdm(dataloader, desc="Extracting concept embedding"):
        images = batch['images'].to(device)
        masks = batch['masks']  # List of tensors
        
        # Move masks to device
        masks = [m.to(device) for m in masks]
        
        # Extract embedding for this batch
        batch_embedding = extract_visual_concept_embedding(
            model, images, masks, pool_type, device
        )
        
        if batch_embedding.sum() != 0:  # Skip if no valid masks
            all_embeddings.append(batch_embedding)
        
        num_processed += images.shape[0]
        if max_images and num_processed >= max_images:
            break
    
    if not all_embeddings:
        raise ValueError("No valid embeddings extracted")
    
    # Aggregate across all batches
    stacked = torch.stack(all_embeddings, dim=0)
    concept_embedding = stacked.mean(dim=0)
    
    return concept_embedding


class VisualToTextMapper:
    """
    Maps visual concept embeddings to text token space.
    
    This handles the potential dimension mismatch between
    vision features and text embeddings.
    """
    
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        vocab_embeddings: VocabularyEmbeddings,
    ):
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.vocab_embeddings = vocab_embeddings
        
        # If dimensions don't match, we need a projection
        if vision_dim != text_dim:
            self.needs_projection = True
            # Learn a simple linear projection
            # For now, just use PCA-style dimensionality matching
            self.projection = None
        else:
            self.needs_projection = False
            
    def project_to_text_space(
        self,
        visual_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Project visual embedding to text embedding space."""
        if not self.needs_projection:
            return visual_embedding
            
        # Simple approach: truncate or pad
        if self.vision_dim > self.text_dim:
            return visual_embedding[:self.text_dim]
        else:
            padded = torch.zeros(self.text_dim, device=visual_embedding.device)
            padded[:self.vision_dim] = visual_embedding
            return padded
    
    def find_seed_tokens(
        self,
        visual_embedding: torch.Tensor,
        k: int = 100,
        exclude_special: bool = True,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Find tokens whose embeddings are closest to the visual concept.
        
        Args:
            visual_embedding: Visual concept embedding
            k: Number of candidate tokens to return
            exclude_special: Whether to exclude special tokens (SOT, EOT)
            
        Returns:
            token_ids: List of k nearest token IDs
            similarities: Similarity scores
        """
        # Project to text space if needed
        text_space_embedding = self.project_to_text_space(visual_embedding)
        
        # Exclude special tokens
        exclude_ids = []
        if exclude_special:
            # Typical special token IDs for CLIP-style tokenizers
            exclude_ids = [49406, 49407]  # SOT, EOT
        
        token_ids, similarities = self.vocab_embeddings.find_nearest(
            text_space_embedding, k=k, exclude_ids=exclude_ids
        )
        
        return token_ids.tolist(), similarities


def initialize_soft_prompt_from_visual(
    visual_embedding: torch.Tensor,
    vocab_embeddings: VocabularyEmbeddings,
    seq_length: int = 5,
    noise_scale: float = 0.1,
) -> torch.Tensor:
    """
    Initialize soft prompt embeddings from visual concept embedding.
    
    Strategy:
    1. Find nearest tokens to visual embedding
    2. Use their embeddings as initialization
    3. Add small noise for diversity
    
    Args:
        visual_embedding: Visual concept embedding
        vocab_embeddings: Pre-computed vocabulary embeddings
        seq_length: Number of tokens in soft prompt
        noise_scale: Scale of noise to add
        
    Returns:
        Soft prompt embeddings [seq_length, embed_dim]
    """
    mapper = VisualToTextMapper(
        vision_dim=visual_embedding.shape[0],
        text_dim=vocab_embeddings.embed_dim,
        vocab_embeddings=vocab_embeddings,
    )
    
    # Find nearest tokens
    token_ids, _ = mapper.find_seed_tokens(visual_embedding, k=seq_length)
    
    # Get their embeddings
    soft_prompt = vocab_embeddings.get_embeddings(token_ids)  # [seq_length, embed_dim]
    
    # Add noise
    noise = torch.randn_like(soft_prompt) * noise_scale
    soft_prompt = soft_prompt + noise
    
    return soft_prompt

def initialize_soft_prompt_from_visual_v2(
    visual_embedding: torch.Tensor,  # [256] - from vision backbone
    vocab_embeddings_256: VocabularyEmbeddings,  # [49408, 256] - post-resizer
    vocab_embeddings_1024: VocabularyEmbeddings,  # [49408, 1024] - raw token embeddings
    seq_length: int = 5,
    noise_scale: float = 0.1,
    exclude_special: bool = True,
) -> torch.Tensor:
    """
    Initialize 1024-dim soft prompt embeddings using 256-dim space for matching.
    
    1. Find nearest tokens in 256-dim space (aligned with vision)
    2. Get their 1024-dim embeddings for soft prompt optimization
    
    Args:
        visual_embedding: 256-dim visual concept embedding
        vocab_embeddings_256: Post-resizer token activations
        vocab_embeddings_1024: Raw token embeddings
        seq_length: Number of tokens in soft prompt
        noise_scale: Scale of noise to add
        
    Returns:
        Soft prompt embeddings [seq_length, 1024]
    """
    exclude_ids = [49406, 49407] if exclude_special else []
    
    # Find nearest tokens in 256-dim space
    token_ids, similarities = vocab_embeddings_256.find_nearest(
        visual_embedding, k=seq_length, exclude_ids=exclude_ids
    )
    token_ids = token_ids.tolist()
    
    print(f"  Nearest tokens (256-dim): {token_ids}")
    print(f"  Similarities: {[f'{s:.4f}' for s in similarities.tolist()]}")
    
    # Get their 1024-dim embeddings
    soft_prompt = vocab_embeddings_1024.get_embeddings(token_ids)  # [seq_length, 1024]
    
    # Add noise
    if noise_scale > 0:
        noise = torch.randn_like(soft_prompt) * noise_scale
        soft_prompt = soft_prompt + noise
    
    return soft_prompt


def build_vocabulary_embeddings_1024(
    model,  # SAM3 model
    cache_path: str = "cache/vocab_embeddings_1024d.pt",
    device: str = 'mps',
    use_cache: bool = True,
) -> VocabularyEmbeddings:
    """
    Build 1024-dim embeddings for all tokens (pre-transformer).
    
    These are the raw token embeddings from the token_embedding layer,
    before any transformer processing.
    
    Args:
        model: SAM3 model
        cache_path: Where to cache the embeddings
        device: Device to use for computation
        use_cache: Whether to use cached embeddings if available
        
    Returns:
        VocabularyEmbeddings with 1024-dim embeddings
    """
    cache_path = Path(cache_path)
    
    # Try to load from cache
    if use_cache and cache_path.exists():
        print(f"Loading 1024-dim vocabulary embeddings from cache: {cache_path}")
        return VocabularyEmbeddings.load(str(cache_path), device='cpu')
    
    print("Building 1024-dim vocabulary embeddings (pre-transformer)...")
    
    # Get tokenizer and text encoder from model
    text_encoder = model.backbone.language_backbone
    tokenizer = text_encoder.tokenizer
    
    vocab_size = tokenizer.vocab_size
    embed_dim = text_encoder.encoder.width  # 1024
    
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embed dim: {embed_dim}")
    
    # Get the raw token embedding layer
    token_embedding = text_encoder.encoder.token_embedding
    
    all_embeddings = []
    all_token_ids = []
    
    batch_size = 256
    
    for start_idx in tqdm(range(0, vocab_size, batch_size), desc="Building 1024d vocab"):
        end_idx = min(start_idx + batch_size, vocab_size)
        batch_ids = list(range(start_idx, end_idx))
        
        # Get raw token embeddings (pre-transformer)
        token_ids_tensor = torch.tensor(batch_ids, device=device)
        with torch.no_grad():
            embeddings = token_embedding(token_ids_tensor)  # [batch, 1024]
        
        all_embeddings.append(embeddings.cpu())
        all_token_ids.extend(batch_ids)
    
    # Concatenate
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    print(f"  Final shape: {all_embeddings.shape}")
    
    vocab_embeddings = VocabularyEmbeddings(
        embeddings=all_embeddings,
        token_ids=all_token_ids,
        embed_dim=embed_dim,
    )
    
    # Cache to disk
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_embeddings.save(str(cache_path))
        print(f"Saved 1024-dim vocabulary embeddings to: {cache_path}")
    
    return vocab_embeddings


def load_both_vocab_embeddings(
    model=None,  # Make optional
    cache_256_path: str = "cache/token_activation_map_256d.pt",
    cache_1024_path: str = "cache/vocab_embeddings_1024d.pt",
    device: str = 'cuda',
) -> tuple:
    """
    Load both 256-dim and 1024-dim vocabulary embeddings.
    """
    # Skip 256-dim (we don't use it anymore)
    vocab_embeddings_256 = None
    
    # Load or build 1024-dim
    if Path(cache_1024_path).exists():
        print(f"Loading 1024-dim embeddings from: {cache_1024_path}")
        vocab_embeddings_1024 = VocabularyEmbeddings.load(cache_1024_path, device='cpu')
    elif model is not None:
        print("Building 1024-dim embeddings (this takes a couple min)...")
        vocab_embeddings_1024 = build_vocabulary_embeddings_1024(
            model=model,
            cache_path=cache_1024_path,
            device=device,
            use_cache=True,
        )
    else:
        raise ValueError("No 1024-dim cache found and no model provided to build it")
    
    print(f"  1024-dim: {vocab_embeddings_1024.vocab_size} tokens, {vocab_embeddings_1024.embed_dim}-dim")
    
    return vocab_embeddings_256, vocab_embeddings_1024
