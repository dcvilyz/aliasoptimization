"""
Data loading utilities for COCO format datasets.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pycocotools.mask as mask_utils


@dataclass
class InstanceMask:
    """Single instance mask with metadata."""
    mask: np.ndarray  # Binary mask [H, W]
    bbox: List[float]  # [x, y, width, height]
    area: float
    category_id: int
    instance_id: int


@dataclass
class ImageAnnotation:
    """All annotations for a single image."""
    image_id: int
    image_path: str
    width: int
    height: int
    instances: List[InstanceMask]
    
    @property
    def num_instances(self) -> int:
        return len(self.instances)
    
    def get_combined_mask(self) -> np.ndarray:
        """Get union of all instance masks."""
        if not self.instances:
            return np.zeros((self.height, self.width), dtype=bool)
        combined = np.zeros((self.height, self.width), dtype=bool)
        for inst in self.instances:
            combined |= inst.mask.astype(bool)
        return combined
    
    def get_instance_masks(self) -> np.ndarray:
        """Get stacked instance masks [N, H, W]."""
        if not self.instances:
            return np.zeros((0, self.height, self.width), dtype=bool)
        return np.stack([inst.mask for inst in self.instances], axis=0)


@dataclass
class CategoryData:
    """All data for a single category/class."""
    category_id: int
    category_name: str
    annotations: List[ImageAnnotation]
    
    @property
    def num_images(self) -> int:
        return len(self.annotations)
    
    @property
    def num_instances(self) -> int:
        return sum(ann.num_instances for ann in self.annotations)
    
    def get_all_masks(self) -> List[Tuple[str, np.ndarray]]:
        """Get all (image_path, instance_masks) pairs."""
        return [(ann.image_path, ann.get_instance_masks()) for ann in self.annotations]


class COCODataset:
    """
    Load and manage COCO format dataset.
    
    Expected structure:
    dataset_path/
        train/
            images...
            _annotations.coco.json
        valid/
            images...
            _annotations.coco.json
        test/
            images...
            _annotations.coco.json
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        annotations_file: str = "_annotations.coco.json"
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.split_path = self.dataset_path / split
        self.annotations_path = self.split_path / annotations_file
        
        # Load COCO annotations
        with open(self.annotations_path, 'r') as f:
            self.coco_data = json.load(f)
            
        # Build indices
        self._build_indices()
        self._filter_empty_categories()
        self._reindex_categories()
        
    def _build_indices(self):
        """Build lookup indices from COCO data."""
        # Category index: id -> name
        self.categories = {
            cat['id']: cat['name'] 
            for cat in self.coco_data['categories']
        }
        self.category_ids = list(self.categories.keys())
        self.category_names = list(self.categories.values())
        
        # Image index: id -> image info
        self.images = {
            img['id']: img 
            for img in self.coco_data['images']
        }
        
        # Annotations grouped by image and category
        # {image_id: {category_id: [annotations]}}
        self.image_category_anns: Dict[int, Dict[int, List]] = {}
        
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            
            if img_id not in self.image_category_anns:
                self.image_category_anns[img_id] = {}
            if cat_id not in self.image_category_anns[img_id]:
                self.image_category_anns[img_id][cat_id] = []
                
            self.image_category_anns[img_id][cat_id].append(ann)
            
        # Category to images mapping
        # {category_id: [image_ids]}
        self.category_images: Dict[int, List[int]] = {
            cat_id: [] for cat_id in self.category_ids
        }
        for img_id, cat_anns in self.image_category_anns.items():
            for cat_id in cat_anns.keys():
                self.category_images[cat_id].append(img_id)
                
    def _decode_mask(self, ann: dict, height: int, width: int) -> np.ndarray:
        """Decode mask from COCO annotation."""
        if 'segmentation' not in ann:
            # No segmentation, create from bbox
            x, y, w, h = [int(v) for v in ann['bbox']]
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y:y+h, x:x+w] = 1
            return mask
            
        seg = ann['segmentation']
        
        if isinstance(seg, dict):
            # RLE format
            if isinstance(seg['counts'], list):
                # Uncompressed RLE
                rle = mask_utils.frPyObjects(seg, height, width)
            else:
                # Compressed RLE
                rle = seg
            mask = mask_utils.decode(rle)
        elif isinstance(seg, list):
            # Polygon format
            rle = mask_utils.frPyObjects(seg, height, width)
            mask = mask_utils.decode(rle)
            if mask.ndim == 3:
                mask = mask.any(axis=2)
        else:
            raise ValueError(f"Unknown segmentation format: {type(seg)}")
            
        return mask.astype(np.uint8)
    
    def get_category_data(self, category_id: int, max_images: Optional[int] = None) -> CategoryData:
        """Get all data for a specific category."""
        if category_id not in self.categories:
            raise ValueError(f"Category {category_id} not found")
            
        annotations = []
        for img_id in self.category_images[category_id]:
            img_info = self.images[img_id]
            img_path = str(self.split_path / img_info['file_name'])
            
            instances = []
            for ann in self.image_category_anns[img_id].get(category_id, []):
                mask = self._decode_mask(ann, img_info['height'], img_info['width'])
                instances.append(InstanceMask(
                    mask=mask,
                    bbox=ann['bbox'],
                    area=ann.get('area', mask.sum()),
                    category_id=category_id,
                    instance_id=ann['id']
                ))
                
            annotations.append(ImageAnnotation(
                image_id=img_id,
                image_path=img_path,
                width=img_info['width'],
                height=img_info['height'],
                instances=instances
            ))
            if max_images is not None and len(annotations) >= max_images:
                break
            
        return CategoryData(
            category_id=category_id,
            category_name=self.categories[category_id],
            annotations=annotations
        )
    
    def get_all_categories(self) -> List[CategoryData]:
        """Get data for all categories."""
        return [self.get_category_data(cat_id) for cat_id in self.category_ids]
    
    def summary(self) -> str:
        """Get dataset summary."""
        lines = [
            f"Dataset: {self.dataset_path.name}",
            f"Split: {self.split}",
            f"Images: {len(self.images)}",
            f"Categories: {len(self.categories)}",
            "",
            "Categories:"
        ]
        for cat_id, cat_name in self.categories.items():
            n_images = len(self.category_images[cat_id])
            n_instances = sum(
                len(self.image_category_anns[img_id].get(cat_id, []))
                for img_id in self.category_images[cat_id]
            )
            lines.append(f"  [{cat_id}] {cat_name}: {n_images} images, {n_instances} instances")
            
        return "\n".join(lines)

    def _filter_empty_categories(self):
        """Remove categories with no images or no instance annotations."""
        filtered_cats = {}
        filtered_cat_images = {}
        
        for cat_id, cat_name in self.categories.items():
            # How many images & instances?
            img_ids = self.category_images.get(cat_id, [])
            inst_count = sum(
                len(self.image_category_anns.get(img_id, {}).get(cat_id, []))
                for img_id in img_ids
            )

            if len(img_ids) == 0 or inst_count == 0:
                print(f"[dataset] Dropping category {cat_id} ({cat_name}) — no images or no annotations")
                continue
            
            # Keep it
            filtered_cats[cat_id] = cat_name
            filtered_cat_images[cat_id] = img_ids
        
        self.categories = filtered_cats
        self.category_images = filtered_cat_images

    def _reindex_categories(self):
        """Make category IDs dense (0..K-1) after filtering."""
        old_ids = sorted(self.categories.keys())
        mapping = {old_id: new_id for new_id, old_id in enumerate(old_ids)}

        # Remap categories dict
        self.categories = {
            mapping[old_id]: name
            for old_id, name in self.categories.items()
        }

        # Remap category_images
        self.category_images = {
            mapping[old_id]: img_ids
            for old_id, img_ids in self.category_images.items()
        }

        # Remap image_category_anns
        new_image_category_anns = {}
        for img_id, ann_dict in self.image_category_anns.items():
            new_ann_dict = {}
            for old_cat_id, anns in ann_dict.items():
                if old_cat_id in mapping:
                    new_ann_dict[mapping[old_cat_id]] = anns
            new_image_category_anns[img_id] = new_ann_dict
        self.image_category_anns = new_image_category_anns

        # Update category_ids list if you store one
        if hasattr(self, "category_ids"):
            self.category_ids = list(self.categories.keys())

        print("[dataset] Reindexed categories:", self.categories.keys())


class CategoryBatchDataset(Dataset):
    """
    PyTorch Dataset that yields batches of images for a single category.
    Used during optimization to evaluate token sequences.
    """
    
    def __init__(
        self,
        category_data: CategoryData,
        image_size: int = 1008,
        transform=None
    ):
        self.category_data = category_data
        self.image_size = image_size
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.category_data.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ann = self.category_data.annotations[idx]
        
        # Load image
        image = Image.open(ann.image_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        # Resize image
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Convert to tensor [C, H, W]
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Normalize (SAM3 normalization)
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        # Resize masks to match
        instance_masks = ann.get_instance_masks()  # [N, H, W]
        if instance_masks.shape[0] > 0:
            # Resize each mask
            resized_masks = []
            for mask in instance_masks:
                mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
                resized_masks.append(np.array(mask_pil) > 127)
            masks_tensor = torch.from_numpy(np.stack(resized_masks, axis=0))
        else:
            masks_tensor = torch.zeros((0, self.image_size, self.image_size), dtype=torch.bool)
            
        return {
            'image': image_tensor,
            'masks': masks_tensor,
            'num_instances': len(ann.instances),
            'image_id': ann.image_id,
            'original_size': torch.tensor(original_size),
        }


def collate_category_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for CategoryBatchDataset."""
    images = torch.stack([b['image'] for b in batch])
    
    # Masks have variable number of instances, keep as list
    masks = [b['masks'] for b in batch]
    
    num_instances = torch.tensor([b['num_instances'] for b in batch])
    image_ids = torch.tensor([b['image_id'] for b in batch])
    original_sizes = torch.stack([b['original_size'] for b in batch])
    
    return {
        'images': images,
        'masks': masks,  # List of [N_i, H, W] tensors
        'num_instances': num_instances,
        'image_ids': image_ids,
        'original_sizes': original_sizes,
    }


def get_category_dataloader(
    category_data: CategoryData,
    batch_size: int = 4,
    image_size: int = 1008,
    shuffle: bool = False,
    num_workers: int = 0,  # 0 for MPS compatibility
) -> DataLoader:
    """Create DataLoader for a category."""
    dataset = CategoryBatchDataset(category_data, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_category_batch,
        pin_memory=False,  # MPS doesn't support pinned memory
    )


# Testing
if __name__ == "__main__":
    # Example usage
    # Note: Set dataset_path to your COCO format dataset
    dataset_path = None  # Set to your dataset path
    
    if dataset_path is None:
        print("Please set dataset_path to your COCO format dataset")
        sys.exit(1)
    
    try:
        dataset = COCODataset(dataset_path, split="train")
        print(dataset.summary())
        
        # Get first category
        if dataset.category_ids:
            cat_data = dataset.get_category_data(dataset.category_ids[0])
            print(f"\nFirst category: {cat_data.category_name}")
            print(f"  Images: {cat_data.num_images}")
            print(f"  Instances: {cat_data.num_instances}")
            
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("This is expected in the container environment.")
