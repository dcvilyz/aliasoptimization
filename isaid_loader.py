"""
iSAID Dataset Loader - converts iSAID to SaCoDataset format for alias optimization.

iSAID Categories (15 classes):
- plane, ship, storage_tank
- baseball_diamond, tennis_court, basketball_court, Soccer_ball_field
- Ground_Track_Field, Roundabout, Swimming_pool
- Harbor, Bridge
- Large_Vehicle, Small_Vehicle
- Helicopter

Usage:
    from isaid_loader import load_isaid_dataset
    
    dataset = load_isaid_dataset(
        annotations_json='path/to/iSAID_val.json',
        images_dir='path/to/images/',
    )
    
    # Use like SaCoDataset
    concept = dataset['storage_tank']
    print(concept.num_positive_images)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from pycocotools import mask as mask_utils
from PIL import Image


@dataclass
class SaCoConceptData:
    """Compatible with saco_loader.SaCoConceptData"""
    text_input: str
    positive_image_paths: List[str]
    positive_masks: List[List[np.ndarray]]  # List of [masks per image]
    positive_bboxes: List[List[List[float]]]
    positive_pair_ids: List[str]
    negative_image_paths: List[str]
    negative_pair_ids: List[str]
    
    # For lazy loading - use field with default_factory
    _positive_polygons: Optional[List[List[dict]]] = None
    _image_sizes: Optional[List[tuple]] = None
    _masks_decoded: bool = True
    
    @property
    def num_positive_images(self) -> int:
        return len(self.positive_image_paths)
    
    @property
    def num_negative_images(self) -> int:
        return len(self.negative_image_paths)
    
    def get_masks(self, idx: int, target_size: int = 1008) -> List[np.ndarray]:
        """Get masks for image at index, decoding lazily if needed.
        
        Args:
            idx: Image index
            target_size: Decode masks at this resolution (matches SAM3 input size)
        """
        if self._masks_decoded:
            return self.positive_masks[idx]
        
        # Check if we have a cache
        if not hasattr(self, '_mask_cache'):
            self._mask_cache = {}
        
        # Return from cache if available
        if idx in self._mask_cache:
            return self._mask_cache[idx]
        
        # Decode at target resolution for speed
        polygons = self._positive_polygons[idx]
        orig_h, orig_w = self._image_sizes[idx]
        
        # Scale polygons to target size
        scale_x = target_size / orig_w
        scale_y = target_size / orig_h
        
        masks = []
        for poly in polygons:
            # Scale polygon coordinates
            scaled_poly = []
            for ring in poly:
                scaled_ring = []
                for i in range(0, len(ring), 2):
                    scaled_ring.append(ring[i] * scale_x)
                    scaled_ring.append(ring[i+1] * scale_y)
                scaled_poly.append(scaled_ring)
            
            mask = polygon_to_mask(scaled_poly, target_size, target_size)
            masks.append(mask)
        
        # Cache for future use
        self._mask_cache[idx] = masks
        return masks


class SaCoDataset:
    """Compatible with saco_loader.SaCoDataset"""
    def __init__(self, concepts: Dict[str, SaCoConceptData]):
        self.concepts = concepts
    
    @property
    def concept_names(self) -> List[str]:
        return list(self.concepts.keys())
    
    def __getitem__(self, concept_name: str) -> SaCoConceptData:
        return self.concepts[concept_name]
    
    def __len__(self) -> int:
        return len(self.concepts)


def polygon_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    """Convert COCO polygon segmentation to binary mask."""
    from pycocotools import mask as mask_utils
    
    if isinstance(segmentation, list):
        # Polygon format
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    else:
        # RLE format
        rle = segmentation
    
    mask = mask_utils.decode(rle)
    return mask.astype(np.uint8)


def load_isaid_dataset(
    annotations_json: str,
    images_dir: str,
    min_instances_per_concept: int = 5,
    include_negatives: bool = True,
) -> SaCoDataset:
    """
    Load iSAID dataset and convert to SaCoDataset format.
    
    Args:
        annotations_json: Path to iSAID_train.json or iSAID_val.json
        images_dir: Path to directory containing P*.png images
        min_instances_per_concept: Minimum instances to include a concept
        include_negatives: Whether to generate negative examples
        
    Returns:
        SaCoDataset with all iSAID categories as concepts
    """
    print(f"Loading iSAID annotations from: {annotations_json}")
    
    with open(annotations_json) as f:
        data = json.load(f)
    
    images_dir = Path(images_dir)
    
    # Build lookups
    images_by_id = {img['id']: img for img in data['images']}
    categories_by_id = {cat['id']: cat['name'] for cat in data['categories']}
    
    print(f"  Images: {len(images_by_id)}")
    print(f"  Annotations: {len(data['annotations'])}")
    print(f"  Categories: {list(categories_by_id.values())}")
    
    # Group annotations by (image_id, category_id)
    # Each image+category combo becomes one "positive example" with multiple masks
    annotations_grouped = {}  # (image_id, category_id) -> [annotations]
    
    for ann in data['annotations']:
        key = (ann['image_id'], ann['category_id'])
        if key not in annotations_grouped:
            annotations_grouped[key] = []
        annotations_grouped[key].append(ann)
    
    # Build concepts
    concepts = {}
    
    for cat_id, cat_name in categories_by_id.items():
        # Find all images with this category
        positive_image_paths = []
        positive_polygons = []  # Store raw polygons, not decoded masks
        positive_bboxes = []
        positive_pair_ids = []
        image_sizes = []
        
        # Get all (image_id, category_id) pairs for this category
        for (img_id, c_id), anns in annotations_grouped.items():
            if c_id != cat_id:
                continue
            
            img_info = images_by_id[img_id]
            img_path = images_dir / img_info['file_name']
            
            if not img_path.exists():
                continue
            
            # Store polygons for lazy decoding
            polygons = []
            bboxes = []
            
            for ann in anns:
                polygons.append(ann['segmentation'])
                bboxes.append(ann['bbox'])
            
            if polygons:
                positive_image_paths.append(str(img_path))
                positive_polygons.append(polygons)
                positive_bboxes.append(bboxes)
                positive_pair_ids.append(f"{img_info['file_name']}_{cat_name}")
                image_sizes.append((img_info['height'], img_info['width']))
        
        if len(positive_image_paths) < min_instances_per_concept:
            continue
        
        # Generate negatives: images that DON'T contain this category
        negative_image_paths = []
        negative_pair_ids = []
        
        if include_negatives:
            positive_img_ids = set()
            for (img_id, c_id), _ in annotations_grouped.items():
                if c_id == cat_id:
                    positive_img_ids.add(img_id)
            
            for img_id, img_info in images_by_id.items():
                if img_id not in positive_img_ids:
                    img_path = images_dir / img_info['file_name']
                    if img_path.exists():
                        negative_image_paths.append(str(img_path))
                        negative_pair_ids.append(f"{img_info['file_name']}_neg_{cat_name}")
        
        # Create concept with human-readable name
        readable_name = cat_name.lower().replace('_', ' ')
        
        concepts[readable_name] = SaCoConceptData(
            text_input=readable_name,
            positive_image_paths=positive_image_paths,
            positive_masks=[],  # Empty - will decode lazily
            positive_bboxes=positive_bboxes,
            positive_pair_ids=positive_pair_ids,
            negative_image_paths=negative_image_paths,
            negative_pair_ids=negative_pair_ids,
            _positive_polygons=positive_polygons,
            _image_sizes=image_sizes,
            _masks_decoded=False,
        )
        
        total_instances = sum(len(p) for p in positive_polygons)
        print(f"  {readable_name}: {len(positive_image_paths)} images, {total_instances} instances")
    
    dataset = SaCoDataset(concepts=concepts)
    
    print(f"\nLoaded iSAID dataset:")
    print(f"  Concepts: {len(dataset.concept_names)}")
    print(f"  Total positive images: {sum(dataset[c].num_positive_images for c in dataset.concept_names)}")
    
    return dataset


def print_isaid_stats(dataset: SaCoDataset):
    """Print statistics about loaded iSAID dataset."""
    print("\n" + "="*60)
    print("iSAID DATASET STATISTICS")
    print("="*60)
    
    print(f"\n{'Concept':<25} {'Pos Imgs':<10} {'Neg Imgs':<10} {'Instances':<12}")
    print("-"*60)
    
    for name in sorted(dataset.concept_names):
        concept = dataset[name]
        # Count instances from polygons if available, else from masks
        if concept._positive_polygons:
            total_instances = sum(len(polys) for polys in concept._positive_polygons)
        else:
            total_instances = sum(len(masks) for masks in concept.positive_masks)
        print(f"{name:<25} {concept.num_positive_images:<10} {concept.num_negative_images:<10} {total_instances:<12}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python isaid_loader.py <annotations_json> <images_dir>")
        sys.exit(1)
    
    dataset = load_isaid_dataset(sys.argv[1], sys.argv[2])
    print_isaid_stats(dataset)