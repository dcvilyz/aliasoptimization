"""
SA-Co Dataset Loader for Open-Vocabulary Segmentation.

SA-Co (Segment Anything Co-segmentation) uses a different structure than standard COCO:
- Each "image" entry is actually an (image, text_prompt) pair
- The text_input field contains arbitrary noun phrases
- Same physical image can appear multiple times with different prompts
- Categories field is ignored (open-vocabulary)

Reference: https://huggingface.co/datasets/facebook/SACo-Gold
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pycocotools.mask as mask_utils


@dataclass
class SaCoConceptData:
    """Data for a single concept (noun phrase) in SA-Co."""
    text_input: str  # The noun phrase (e.g., "golden retriever")
    
    # Positive samples (have annotations)
    positive_image_paths: List[str] = field(default_factory=list)
    positive_masks: List[List[np.ndarray]] = field(default_factory=list)  # List of mask lists per image
    positive_bboxes: List[List[List[float]]] = field(default_factory=list)  # List of bbox lists per image
    positive_pair_ids: List[str] = field(default_factory=list)  # Original pair IDs
    
    # Negative samples (no annotations for this prompt)
    negative_image_paths: List[str] = field(default_factory=list)
    negative_pair_ids: List[str] = field(default_factory=list)
    
    @property
    def num_positive_images(self) -> int:
        return len(self.positive_image_paths)
    
    @property
    def num_negative_images(self) -> int:
        return len(self.negative_image_paths)
    
    @property
    def num_total_images(self) -> int:
        return self.num_positive_images + self.num_negative_images
    
    @property
    def num_instances(self) -> int:
        """Total number of mask instances across all positive images."""
        return sum(len(masks) for masks in self.positive_masks)
    
    def __repr__(self):
        return (
            f"SaCoConceptData('{self.text_input}', "
            f"positive={self.num_positive_images}, "
            f"negative={self.num_negative_images}, "
            f"instances={self.num_instances})"
        )


class SaCoDataset:
    """
    Loader for SA-Co Gold/Silver datasets.
    
    Handles the open-vocabulary structure where each entry is an
    (image, text_prompt) pair rather than a fixed category.
    """
    def __init__(
        self,
        dataset_path: str,
        annotation_file: str = "annotations.json",
        images_dir: str = None,  # NEW: separate images path
        include_negatives: bool = False,#whether to include negatives
    ):
        self.dataset_path = Path(dataset_path)
        
        # Images can be in a separate location
        if images_dir is not None:
            self.images_dir = Path(images_dir)
        else:
            self.images_dir = self.dataset_path / "images"
        self.include_negatives = include_negatives
        
        # Load annotations
        annotation_path = self.dataset_path / annotation_file
        if not annotation_path.exists():
            # Try common alternatives
            for alt_name in ["saco_gold.json", "saco.json", "instances.json", 
                            "train.json", "val.json", "test.json"]:
                alt_path = self.dataset_path / alt_name
                if alt_path.exists():
                    annotation_path = alt_path
                    break
            else:
                # List available files to help user
                available = list(self.dataset_path.glob("*.json"))
                raise FileNotFoundError(
                    f"Annotation file not found at {annotation_path}\n"
                    f"Available JSON files: {available}"
                )
        
        print(f"Loading SA-Co annotations from: {annotation_path}")
        with open(annotation_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Parse the data
        self._parse_annotations()
        
        print(f"Loaded SA-Co dataset:")
        print(f"  Unique concepts: {len(self.concepts)}")
        print(f"  Total image-prompt pairs: {len(self.coco_data['images'])}")
        print(f"  Total annotations: {len(self.coco_data.get('annotations', []))}")
    
    def _parse_annotations(self):
        """Parse SA-Co format into concept-grouped structure."""
        
        # Build image-pair lookup: pair_id -> image info
        self.pair_lookup: Dict[str, dict] = {}
        for img_info in self.coco_data['images']:
            pair_id = str(img_info['id'])
            self.pair_lookup[pair_id] = img_info
        
        # Build annotation lookup: pair_id -> list of annotations
        self.annotation_lookup: Dict[str, List[dict]] = defaultdict(list)
        for ann in self.coco_data.get('annotations', []):
            pair_id = str(ann['image_id'])
            self.annotation_lookup[pair_id].append(ann)
        
        # Group by text_input (the noun phrase)
        concept_groups: Dict[str, List[str]] = defaultdict(list)  # text_input -> list of pair_ids
        for pair_id, img_info in self.pair_lookup.items():
            text_input = img_info['text_input']
            concept_groups[text_input].append(pair_id)
        
        # Build concept data objects
        self.concepts: Dict[str, SaCoConceptData] = {}
        
        for text_input, pair_ids in concept_groups.items():
            concept = SaCoConceptData(text_input=text_input)
            
            for pair_id in pair_ids:
                img_info = self.pair_lookup[pair_id]
                img_path = str(self.images_dir / img_info['file_name'])
                
                annotations = self.annotation_lookup.get(pair_id, [])
                
                if annotations:
                    # Positive sample - has masks
                    concept.positive_pair_ids.append(pair_id)
                    concept.positive_image_paths.append(img_path)
                    
                    # Decode masks
                    masks = []
                    bboxes = []
                    for ann in annotations:
                        # Decode RLE segmentation
                        if 'segmentation' in ann:
                            rle = ann['segmentation']
                            if isinstance(rle, dict):
                                # Already in RLE format
                                mask = mask_utils.decode(rle)
                            elif isinstance(rle, list):
                                # Polygon format - need to convert
                                # This is less common in SA-Co but handle it
                                h = img_info.get('height', 1008)
                                w = img_info.get('width', 1008)
                                rle = mask_utils.frPyObjects(rle, h, w)
                                mask = mask_utils.decode(rle)
                                if mask.ndim == 3:
                                    mask = mask.sum(axis=2) > 0
                            masks.append(mask.astype(np.uint8))
                        
                        if 'bbox' in ann:
                            bboxes.append(ann['bbox'])
                    
                    concept.positive_masks.append(masks)
                    concept.positive_bboxes.append(bboxes)
                    
                else:
                    # Negative sample - no masks for this prompt
                    concept.negative_pair_ids.append(pair_id)
                    concept.negative_image_paths.append(img_path)
            
            # Only include concepts that have at least one positive sample
            if concept.num_positive_images > 0:
                self.concepts[text_input] = concept
        
        # Create sorted list of concept names for consistent ordering
        self.concept_names = sorted(self.concepts.keys())
    
    def get_concept(self, text_input: str) -> SaCoConceptData:
        """Get data for a specific concept by its text."""
        if text_input not in self.concepts:
            raise KeyError(f"Concept '{text_input}' not found. "
                          f"Available: {self.concept_names[:5]}...")
        return self.concepts[text_input]
    
    def get_concept_by_index(self, index: int) -> SaCoConceptData:
        """Get concept by index in sorted concept list."""
        return self.concepts[self.concept_names[index]]
    
    def list_concepts(
        self, 
        min_positive_images: int = 1,
        min_instances: int = 1,
        sort_by: str = 'name',  # 'name', 'images', 'instances'
    ) -> List[Dict]:
        """List all concepts with their statistics."""
        concept_info = []
        
        for name in self.concept_names:
            concept = self.concepts[name]
            if (concept.num_positive_images >= min_positive_images and 
                concept.num_instances >= min_instances):
                concept_info.append({
                    'text_input': name,
                    'num_positive_images': concept.num_positive_images,
                    'num_negative_images': concept.num_negative_images,
                    'num_instances': concept.num_instances,
                })
        
        # Sort
        if sort_by == 'images':
            concept_info.sort(key=lambda x: x['num_positive_images'], reverse=True)
        elif sort_by == 'instances':
            concept_info.sort(key=lambda x: x['num_instances'], reverse=True)
        else:
            concept_info.sort(key=lambda x: x['text_input'])
        
        return concept_info
    
    def summary(self) -> str:
        """Return a summary string of the dataset."""
        total_positive = sum(c.num_positive_images for c in self.concepts.values())
        total_negative = sum(c.num_negative_images for c in self.concepts.values())
        total_instances = sum(c.num_instances for c in self.concepts.values())
        
        lines = [
            f"SA-Co Dataset: {self.dataset_path}",
            f"  Unique concepts: {len(self.concepts)}",
            f"  Positive image-pairs: {total_positive}",
            f"  Negative image-pairs: {total_negative}",
            f"  Total mask instances: {total_instances}",
        ]
        return "\n".join(lines)
    
    def __len__(self):
        return len(self.concepts)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get_concept_by_index(key)
        return self.get_concept(key)


class SaCoConceptDataset(Dataset):
    """
    PyTorch Dataset for a single SA-Co concept.
    
    Yields (image, masks) pairs for all positive samples of a concept.
    """
    
    def __init__(
        self,
        concept_data: SaCoConceptData,
        image_size: int = 1008,
        max_images: Optional[int] = None,
        transform=None,
    ):
        self.concept_data = concept_data
        self.image_size = image_size
        self.transform = transform
        
        # Limit number of images if specified
        self.num_images = concept_data.num_positive_images
        if max_images is not None:
            self.num_images = min(self.num_images, max_images)
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.concept_data.positive_image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Resize image
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Load and resize masks
        masks_np = self.concept_data.positive_masks[idx]
        masks_resized = []
        
        for mask in masks_np:
            # Resize mask
            mask_pil = Image.fromarray(mask * 255)
            mask_resized = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_resized) > 127).bool()
            masks_resized.append(mask_tensor)
        
        if masks_resized:
            masks_tensor = torch.stack(masks_resized, dim=0)  # [N, H, W]
        else:
            masks_tensor = torch.zeros((0, self.image_size, self.image_size), dtype=torch.bool)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return {
            'images': image_tensor,
            'masks': masks_tensor,
            'text_input': self.concept_data.text_input,
            'image_path': img_path,
            'num_instances': len(masks_np),
        }


def get_concept_dataloader(
    concept_data: SaCoConceptData,
    batch_size: int = 1,
    image_size: int = 1008,
    max_images: Optional[int] = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for a single concept."""
    
    dataset = SaCoConceptDataset(
        concept_data=concept_data,
        image_size=image_size,
        max_images=max_images,
    )
    
    def collate_fn(batch):
        """Custom collate to handle variable-size mask lists."""
        images = torch.stack([b['images'] for b in batch], dim=0)
        masks = [b['masks'] for b in batch]  # Keep as list, variable sizes
        text_inputs = [b['text_input'] for b in batch]
        image_paths = [b['image_path'] for b in batch]
        
        return {
            'images': images,
            'masks': masks,
            'text_inputs': text_inputs,
            'image_paths': image_paths,
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


# =============================================================================
# Convenience functions for validation
# =============================================================================

def load_saco_dataset(
    dataset_path: str,
    annotation_file: Optional[str] = None,
    images_dir: Optional[str] = None,  # ADD THIS
) -> SaCoDataset:
    """
    Load SA-Co dataset with automatic annotation file detection.
    
    Args:
        dataset_path: Path to SA-Co dataset directory
        annotation_file: Optional specific annotation file name
        
    Returns:
        SaCoDataset instance
    """
    path = Path(dataset_path)
    
    # Try to find annotation file
    if annotation_file:
        return SaCoDataset(dataset_path, annotation_file=annotation_file, images_dir=images_dir)
    
    # Auto-detect
    candidates = [
        "annotations.json",
        "saco_gold.json", 
        "saco_silver.json",
        "saco.json",
        "train.json",
        "val.json",
    ]
    
    for candidate in candidates:
        if (path / candidate).exists():
            return SaCoDataset(dataset_path, annotation_file=candidate,images_dir=images_dir)
    
    # Try to find any JSON file
    json_files = list(path.glob("*.json"))
    if json_files:
        return SaCoDataset(dataset_path, annotation_file=json_files[0].name, images_dir=images_dir)
    
    raise FileNotFoundError(f"No annotation file found in {dataset_path}")


def print_concept_table(
    dataset: SaCoDataset,
    min_images: int = 1,
    max_rows: int = 50,
    sort_by: str = 'instances',
):
    """Print a formatted table of concepts."""
    concepts = dataset.list_concepts(min_positive_images=min_images, sort_by=sort_by)
    
    print(f"\n{'Text Input (Noun Phrase)':<50} {'Images':>8} {'Instances':>10}")
    print("-" * 70)
    
    for i, c in enumerate(concepts[:max_rows]):
        text = c['text_input'][:48]
        print(f"{text:<50} {c['num_positive_images']:>8} {c['num_instances']:>10}")
    
    if len(concepts) > max_rows:
        print(f"... and {len(concepts) - max_rows} more concepts")
    
    print("-" * 70)
    print(f"Total: {len(concepts)} concepts")


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python saco_loader.py /path/to/saco-gold")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    # Load dataset
    dataset = load_saco_dataset(dataset_path)
    print(dataset.summary())
    
    # Print concept table
    print_concept_table(dataset, min_images=3, sort_by='instances')
    
    # Test loading a concept
    if dataset.concept_names:
        concept_name = dataset.concept_names[0]
        concept = dataset[concept_name]
        print(f"\nSample concept: {concept}")
        
        # Create dataloader
        loader = get_concept_dataloader(concept, batch_size=2, max_images=5)
        
        for batch in loader:
            print(f"  Batch: images={batch['images'].shape}, "
                  f"masks=[{', '.join(str(m.shape) for m in batch['masks'])}]")
            break
