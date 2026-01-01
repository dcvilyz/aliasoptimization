"""
Augmentation utilities for robust evaluation.

This module provides augmentation functions that can be applied to 
image-mask pairs from either data_loader.py or saco_loader.py.

Usage:
    from augmentation import AugmentedDataLoader, AugmentationConfig
    
    # Wrap any existing dataloader
    aug_loader = AugmentedDataLoader(
        base_loader=get_concept_dataloader(...),
        config=AugmentationConfig(enabled=True)
    )
"""

import random
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Iterator
from torch.utils.data import DataLoader


@dataclass
class AugmentationConfig:
    """Configuration for augmentation."""
    enabled: bool = True
    
    # Probability of applying each augmentation
    horizontal_flip_prob: float = 0.5
    gaussian_blur_prob: float = 0.3
    gaussian_noise_prob: float = 0.3
    brightness_prob: float = 0.2
    contrast_prob: float = 0.2
    
    # Augmentation parameters
    blur_kernel_size: int = 5
    blur_sigma_range: tuple = (0.5, 2.0)
    noise_std_range: tuple = (0.02, 0.1)
    brightness_range: tuple = (0.7, 1.3)
    contrast_range: tuple = (0.7, 1.3)
    
    # How many augmented versions per original image
    # 1 = only augmented version, 2 = original + augmented, etc.
    augmentation_multiplier: int = 1
    
    # Whether to always include original (non-augmented) in evaluation
    include_original: bool = True


def horizontal_flip(image: torch.Tensor, masks: torch.Tensor) -> tuple:
    """
    Flip image and masks horizontally.
    
    Args:
        image: [C, H, W] tensor
        masks: [N, H, W] tensor
        
    Returns:
        flipped image, flipped masks
    """
    image_flipped = torch.flip(image, dims=[-1])
    if masks.shape[0] > 0:
        masks_flipped = torch.flip(masks, dims=[-1])
    else:
        masks_flipped = masks
    return image_flipped, masks_flipped


def gaussian_blur(image: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Apply Gaussian blur to image.
    
    Args:
        image: [C, H, W] tensor
        kernel_size: Size of blur kernel (must be odd)
        sigma: Standard deviation of Gaussian
        
    Returns:
        blurred image
    """
    # Create Gaussian kernel
    channels = image.shape[0]
    
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create 2D kernel
    kernel_2d = kernel_1d.outer(kernel_1d)
    kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    
    # Apply convolution
    padding = kernel_size // 2
    image_padded = F.pad(image.unsqueeze(0), [padding]*4, mode='reflect')
    blurred = F.conv2d(image_padded, kernel_2d.to(image.device), groups=channels)
    
    return blurred.squeeze(0)


def gaussian_noise(image: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """
    Add Gaussian noise to image.
    
    Args:
        image: [C, H, W] tensor
        std: Standard deviation of noise
        
    Returns:
        noisy image
    """
    noise = torch.randn_like(image) * std
    noisy = image + noise
    return torch.clamp(noisy, -1, 1)  # Assuming normalized to [-1, 1]


def adjust_brightness(image: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    """
    Adjust image brightness.
    
    Args:
        image: [C, H, W] tensor (normalized to [-1, 1])
        factor: Brightness factor (1.0 = no change)
        
    Returns:
        brightness-adjusted image
    """
    # Convert from [-1, 1] to [0, 1]
    image_01 = (image + 1) / 2
    adjusted = image_01 * factor
    adjusted = torch.clamp(adjusted, 0, 1)
    # Convert back to [-1, 1]
    return adjusted * 2 - 1


def adjust_contrast(image: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    """
    Adjust image contrast.
    
    Args:
        image: [C, H, W] tensor (normalized to [-1, 1])
        factor: Contrast factor (1.0 = no change)
        
    Returns:
        contrast-adjusted image
    """
    # Convert from [-1, 1] to [0, 1]
    image_01 = (image + 1) / 2
    mean = image_01.mean()
    adjusted = (image_01 - mean) * factor + mean
    adjusted = torch.clamp(adjusted, 0, 1)
    # Convert back to [-1, 1]
    return adjusted * 2 - 1


def apply_augmentation(
    image: torch.Tensor,
    masks: torch.Tensor,
    config: AugmentationConfig,
) -> tuple:
    """
    Apply random augmentations to image and masks.
    
    Args:
        image: [C, H, W] tensor
        masks: [N, H, W] tensor
        config: AugmentationConfig
        
    Returns:
        augmented image, augmented masks
    """
    if not config.enabled:
        return image, masks
    
    # Horizontal flip (affects both image and masks)
    if random.random() < config.horizontal_flip_prob:
        image, masks = horizontal_flip(image, masks)
    
    # Gaussian blur (image only)
    if random.random() < config.gaussian_blur_prob:
        sigma = random.uniform(*config.blur_sigma_range)
        image = gaussian_blur(image, config.blur_kernel_size, sigma)
    
    # Gaussian noise (image only)
    if random.random() < config.gaussian_noise_prob:
        std = random.uniform(*config.noise_std_range)
        image = gaussian_noise(image, std)
    
    # Brightness (image only)
    if random.random() < config.brightness_prob:
        factor = random.uniform(*config.brightness_range)
        image = adjust_brightness(image, factor)
    
    # Contrast (image only)
    if random.random() < config.contrast_prob:
        factor = random.uniform(*config.contrast_range)
        image = adjust_contrast(image, factor)
    
    return image, masks


def augment_batch(
    batch: Dict,
    config: AugmentationConfig,
) -> Dict:
    """
    Apply augmentation to a batch from either dataloader.
    
    Handles both formats:
    - data_loader.py: {'images': [B,C,H,W], 'masks': List[Tensor]}
    - saco_loader.py: {'images': [B,C,H,W], 'masks': List[Tensor]}
    
    Args:
        batch: Batch dict from dataloader
        config: AugmentationConfig
        
    Returns:
        Augmented batch
    """
    if not config.enabled:
        return batch
    
    images = batch['images']  # [B, C, H, W]
    masks_list = batch['masks']  # List of [N_i, H, W]
    
    augmented_images = []
    augmented_masks = []
    
    for i in range(images.shape[0]):
        img = images[i]
        msk = masks_list[i]
        
        aug_img, aug_msk = apply_augmentation(img, msk, config)
        augmented_images.append(aug_img)
        augmented_masks.append(aug_msk)
    
    # Reconstruct batch
    augmented_batch = batch.copy()
    augmented_batch['images'] = torch.stack(augmented_images, dim=0)
    augmented_batch['masks'] = augmented_masks
    
    return augmented_batch


class AugmentedDataLoader:
    """
    Wrapper that applies augmentation to any dataloader.
    
    Can optionally multiply the dataset by yielding both original
    and augmented versions.
    """
    
    def __init__(
        self,
        base_loader: DataLoader,
        config: AugmentationConfig = None,
    ):
        self.base_loader = base_loader
        self.config = config or AugmentationConfig()
    
    def __iter__(self) -> Iterator[Dict]:
        for batch in self.base_loader:
            if self.config.include_original:
                # Yield original batch
                yield batch
            
            if self.config.enabled:
                # Yield augmented version(s)
                for _ in range(self.config.augmentation_multiplier):
                    yield augment_batch(batch, self.config)
    
    def __len__(self) -> int:
        base_len = len(self.base_loader)
        if self.config.include_original:
            return base_len * (1 + self.config.augmentation_multiplier)
        return base_len * self.config.augmentation_multiplier


def create_augmented_dataloader(
    base_loader: DataLoader,
    enabled: bool = True,
    include_original: bool = True,
    augmentation_multiplier: int = 1,
    **aug_kwargs,
) -> AugmentedDataLoader:
    """
    Convenience function to create augmented dataloader.
    
    Args:
        base_loader: Base dataloader to wrap
        enabled: Whether augmentation is enabled
        include_original: Whether to include non-augmented samples
        augmentation_multiplier: How many augmented versions per sample
        **aug_kwargs: Additional AugmentationConfig parameters
        
    Returns:
        AugmentedDataLoader
    """
    config = AugmentationConfig(
        enabled=enabled,
        include_original=include_original,
        augmentation_multiplier=augmentation_multiplier,
        **aug_kwargs,
    )
    return AugmentedDataLoader(base_loader, config)


# =============================================================================
# Integration helpers for existing code
# =============================================================================

def get_augmented_concept_dataloader(
    concept_data,  # SaCoConceptData or CategoryData
    batch_size: int = 1,
    image_size: int = 1008,
    max_images: int = None,
    shuffle: bool = False,
    augment: bool = True,
    aug_config: AugmentationConfig = None,
):
    """
    Create augmented dataloader that works with either SA-Co or COCO data.
    
    This is a drop-in replacement for get_concept_dataloader / get_category_dataloader
    with augmentation support.
    """
    # Import both loaders
    try:
        from saco_loader import get_concept_dataloader as saco_loader, SaCoConceptData
        from data_loader import get_category_dataloader as coco_loader, CategoryData
    except ImportError:
        # Handle partial imports
        SaCoConceptData = None
        CategoryData = None
    
    # Detect data type and create base loader
    data_type = type(concept_data).__name__
    
    if data_type == 'SaCoConceptData' or hasattr(concept_data, 'text_input'):
        from saco_loader import get_concept_dataloader
        base_loader = get_concept_dataloader(
            concept_data=concept_data,
            batch_size=batch_size,
            image_size=image_size,
            max_images=max_images,
            shuffle=shuffle,
        )
    elif data_type == 'CategoryData' or hasattr(concept_data, 'category_name'):
        from data_loader import get_category_dataloader
        base_loader = get_category_dataloader(
            category_data=concept_data,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle,
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    if not augment:
        return base_loader
    
    # Wrap with augmentation
    config = aug_config or AugmentationConfig()
    return AugmentedDataLoader(base_loader, config)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing augmentation functions...")
    
    # Create dummy image and masks
    image = torch.randn(3, 256, 256)
    masks = torch.randint(0, 2, (3, 256, 256)).bool()
    
    config = AugmentationConfig(enabled=True)
    
    # Test individual augmentations
    print("Horizontal flip...")
    img_flip, msk_flip = horizontal_flip(image, masks)
    assert img_flip.shape == image.shape
    assert msk_flip.shape == masks.shape
    
    print("Gaussian blur...")
    img_blur = gaussian_blur(image, kernel_size=5, sigma=1.0)
    assert img_blur.shape == image.shape
    
    print("Gaussian noise...")
    img_noise = gaussian_noise(image, std=0.05)
    assert img_noise.shape == image.shape
    
    print("Brightness adjustment...")
    img_bright = adjust_brightness(image, factor=1.2)
    assert img_bright.shape == image.shape
    
    print("Contrast adjustment...")
    img_contrast = adjust_contrast(image, factor=1.2)
    assert img_contrast.shape == image.shape
    
    print("Combined augmentation...")
    img_aug, msk_aug = apply_augmentation(image, masks, config)
    assert img_aug.shape == image.shape
    assert msk_aug.shape == masks.shape
    
    print("All augmentation tests passed!")