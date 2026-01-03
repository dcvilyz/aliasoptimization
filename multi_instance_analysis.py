"""
Multi-Instance Analysis for SA-Co Dataset.
IMPROVED VERSION with:
1. Area filtering (exclude tiny fragment masks)
2. Coverage sanity checks (exclude texture/region concepts)
3. Scoring function for ranking
4. Support for loading ALL annotation files

Usage:
    from multi_instance_analysis import (
        analyze_multi_instance_concepts,
        get_recommended_test_concepts,
        load_all_saco_annotations,
    )
    
    # Load all annotation files
    dataset = load_all_saco_annotations(gt_annotations_dir, images_base_dir)
    
    # Get top concepts for testing
    recommended = get_recommended_test_concepts(dataset, top_k=20)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json
import os


@dataclass
class ConceptInstanceStats:
    """Statistics about instance counts for a concept."""
    concept_name: str
    
    # Image counts
    num_positive_images: int
    num_single_instance_images: int  # Images with exactly 1 valid instance
    num_multi_instance_images: int   # Images with 2+ valid instances
    
    # Instance counts (valid instances only)
    total_instances: int
    max_instances_per_image: int
    mean_instances_per_image: float
    
    # Instance distribution
    instance_distribution: Dict[int, int]  # {num_instances: count}
    
    # Multi-instance ratio
    multi_instance_ratio: float  # num_multi / num_positive
    
    # Coverage stats (for filtering texture concepts)
    mean_coverage: float  # Average total mask coverage per image
    median_coverage: float
    
    # Scoring
    score: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class ImageInstanceStats:
    """Statistics about a single image."""
    image_path: str
    concept_name: str
    pair_id: str
    num_instances: int
    num_valid_instances: int  # After area filtering
    mask_areas: List[int]
    total_mask_area: int
    image_coverage: float


def compute_mask_area(mask: np.ndarray) -> int:
    """Compute pixel area of a mask."""
    if mask is None:
        return 0
    return int(np.sum(mask > 0))


def is_valid_instance(
    mask: np.ndarray,
    image_area: int,
    min_area_px: int = 100,
    min_area_frac: float = 0.001,
) -> bool:
    """
    Check if a mask represents a valid instance (not a tiny fragment).
    
    Args:
        mask: Binary mask array
        image_area: Total image area in pixels
        min_area_px: Minimum absolute pixel area
        min_area_frac: Minimum fraction of image area
        
    Returns:
        True if mask is large enough to be a valid instance
    """
    if mask is None:
        return False
    
    area = compute_mask_area(mask)
    min_area = max(min_area_px, int(image_area * min_area_frac))
    return area >= min_area


def analyze_concept_instances(
    concept,
    min_area_px: int = 100,
    min_area_frac: float = 0.001,
    default_image_size: int = 1008,
) -> ConceptInstanceStats:
    """
    Analyze instance counts for a single concept with area filtering.
    
    Args:
        concept: SaCoConceptData
        min_area_px: Minimum absolute pixel area for valid instance
        min_area_frac: Minimum fraction of image area for valid instance
        default_image_size: Assumed image size if not known
        
    Returns:
        ConceptInstanceStats with detailed instance information
    """
    valid_instance_counts = []
    coverages = []
    image_area = default_image_size * default_image_size
    
    for masks in concept.positive_masks:
        if not masks:
            valid_instance_counts.append(0)
            coverages.append(0.0)
            continue
        
        # Count valid instances (filtering tiny fragments)
        valid_count = 0
        total_area = 0
        for mask in masks:
            area = compute_mask_area(mask)
            total_area += area
            if is_valid_instance(mask, image_area, min_area_px, min_area_frac):
                valid_count += 1
        
        valid_instance_counts.append(valid_count)
        coverages.append(total_area / image_area if image_area > 0 else 0)
    
    if not valid_instance_counts:
        return ConceptInstanceStats(
            concept_name=concept.text_input,
            num_positive_images=0,
            num_single_instance_images=0,
            num_multi_instance_images=0,
            total_instances=0,
            max_instances_per_image=0,
            mean_instances_per_image=0,
            instance_distribution={},
            multi_instance_ratio=0,
            mean_coverage=0,
            median_coverage=0,
            score=0,
        )
    
    # Compute statistics on valid instances
    counts = np.array(valid_instance_counts)
    num_single = int((counts == 1).sum())
    num_multi = int((counts >= 2).sum())
    
    # Build distribution
    distribution = {}
    for count in counts:
        distribution[int(count)] = distribution.get(int(count), 0) + 1
    
    multi_ratio = num_multi / len(counts) if len(counts) > 0 else 0
    
    # Coverage stats
    coverages = np.array(coverages)
    mean_cov = float(coverages.mean()) if len(coverages) > 0 else 0
    median_cov = float(np.median(coverages)) if len(coverages) > 0 else 0
    
    return ConceptInstanceStats(
        concept_name=concept.text_input,
        num_positive_images=len(counts),
        num_single_instance_images=num_single,
        num_multi_instance_images=num_multi,
        total_instances=int(counts.sum()),
        max_instances_per_image=int(counts.max()) if len(counts) > 0 else 0,
        mean_instances_per_image=float(counts.mean()) if len(counts) > 0 else 0,
        instance_distribution=distribution,
        multi_instance_ratio=multi_ratio,
        mean_coverage=mean_cov,
        median_coverage=median_cov,
        score=0,  # Computed later
    )


def compute_concept_score(
    stats: ConceptInstanceStats,
    max_instances_cap: int = 10,
) -> float:
    """
    Compute a ranking score for a concept.
    
    Score balances:
    - Representation (enough images)
    - Multi-instance pressure (images with 2+ instances)
    - Hardness (max instances, capped)
    
    Formula:
        score = sqrt(P) * (0.7 * M + 0.3 * R) * log1p(min(max_inst, cap))
        
    Where:
        P = num_positive_images
        M = num_multi_instance_images  
        R = multi_instance_ratio
    """
    P = stats.num_positive_images
    M = stats.num_multi_instance_images
    R = stats.multi_instance_ratio
    max_inst = min(stats.max_instances_per_image, max_instances_cap)
    
    if P == 0 or M == 0:
        return 0.0
    
    score = np.sqrt(P) * (0.7 * M + 0.3 * R * 100) * np.log1p(max_inst)
    return float(score)


def analyze_multi_instance_concepts(
    dataset,
    min_positive_images: int = 3,
    min_area_px: int = 100,
    min_area_frac: float = 0.001,
    sort_by: str = 'score',
) -> List[ConceptInstanceStats]:
    """
    Analyze all concepts for multi-instance statistics with area filtering.
    
    Args:
        dataset: SaCoDataset
        min_positive_images: Minimum positive images to include
        min_area_px: Minimum mask area in pixels
        min_area_frac: Minimum mask area as fraction of image
        sort_by: 'score', 'num_multi_instance_images', 'multi_instance_ratio', etc.
        
    Returns:
        List of ConceptInstanceStats sorted by specified field
    """
    results = []
    
    for concept_name in dataset.concept_names:
        concept = dataset[concept_name]
        
        if concept.num_positive_images < min_positive_images:
            continue
        
        stats = analyze_concept_instances(
            concept,
            min_area_px=min_area_px,
            min_area_frac=min_area_frac,
        )
        
        # Compute score
        stats.score = compute_concept_score(stats)
        results.append(stats)
    
    # Sort
    if sort_by == 'score':
        results.sort(key=lambda x: x.score, reverse=True)
    elif sort_by == 'num_multi_instance_images':
        results.sort(key=lambda x: x.num_multi_instance_images, reverse=True)
    elif sort_by == 'multi_instance_ratio':
        results.sort(key=lambda x: x.multi_instance_ratio, reverse=True)
    elif sort_by == 'max_instances_per_image':
        results.sort(key=lambda x: x.max_instances_per_image, reverse=True)
    else:
        results.sort(key=lambda x: x.score, reverse=True)
    
    return results


def get_recommended_test_concepts(
    dataset,
    top_k: int = 20,
    min_multi_instance_images: int = 3,
    min_max_instances: int = 2,
    min_positive_images: int = 10,
    max_coverage: float = 0.35,
    min_coverage: float = 0.005,
    min_area_px: int = 100,
    min_area_frac: float = 0.001,
) -> List[dict]:
    """
    Get recommended concepts for multi-instance testing.
    
    Filters for concepts that:
    1. Have enough positive images for train/test split
    2. Have enough multi-instance images
    3. Are not too texture-y (coverage sanity)
    4. Have valid-sized instances (not fragments)
    
    Then ranks by score and returns top_k.
    
    Args:
        dataset: SaCoDataset
        top_k: Number of concepts to return
        min_multi_instance_images: Minimum images with 2+ valid instances
        min_max_instances: Minimum max instances in any image
        min_positive_images: Minimum total positive images
        max_coverage: Max median coverage (reject texture concepts)
        min_coverage: Min median coverage (reject concepts with tiny masks)
        min_area_px: Minimum mask area in pixels
        min_area_frac: Minimum mask area as fraction of image
        
    Returns:
        List of top_k recommended concepts with their stats
    """
    all_stats = analyze_multi_instance_concepts(
        dataset,
        min_positive_images=min_positive_images,
        min_area_px=min_area_px,
        min_area_frac=min_area_frac,
        sort_by='score',
    )
    
    recommended = []
    for stats in all_stats:
        # Apply filters
        if stats.num_multi_instance_images < min_multi_instance_images:
            continue
        if stats.max_instances_per_image < min_max_instances:
            continue
        
        # Coverage sanity check
        if stats.median_coverage > max_coverage:
            continue  # Too texture-y
        if stats.median_coverage < min_coverage:
            continue  # Too tiny
        
        recommended.append({
            'concept': stats.concept_name,
            'score': stats.score,
            'num_positive_images': stats.num_positive_images,
            'num_multi_instance_images': stats.num_multi_instance_images,
            'multi_instance_ratio': stats.multi_instance_ratio,
            'max_instances': stats.max_instances_per_image,
            'mean_instances': stats.mean_instances_per_image,
            'total_instances': stats.total_instances,
            'median_coverage': stats.median_coverage,
            'mean_coverage': stats.mean_coverage,
            'distribution': stats.instance_distribution,
        })
        
        if len(recommended) >= top_k:
            break
    
    return recommended


def load_all_saco_annotations(
    gt_annotations_dir: str,
    images_base_dir: str,
) -> 'SaCoDataset':
    """
    Load ALL annotation files from gt-annotations directory into a single dataset.
    
    This merges all JSON files (gold_*.json) into one unified dataset.
    
    Args:
        gt_annotations_dir: Path to gt-annotations folder
        images_base_dir: Base path for images (will look in metaclip-images and sa1b-images)
        
    Returns:
        Merged SaCoDataset with all concepts
    """
    from saco_loader import SaCoDataset, SaCoConceptData, load_saco_dataset
    
    gt_dir = Path(gt_annotations_dir)
    annotation_files = list(gt_dir.glob("gold_*.json"))
    
    print(f"Found {len(annotation_files)} annotation files:")
    for f in annotation_files:
        print(f"  {f.name}")
    
    # Create merged dataset
    merged_concepts = {}  # concept_name -> SaCoConceptData
    
    for ann_file in annotation_files:
        print(f"\nLoading {ann_file.name}...")
        
        # Determine image directory based on annotation file name
        if 'metaclip' in ann_file.name:
            images_dir = Path(images_base_dir) / 'metaclip-images'
        elif 'sa1b' in ann_file.name:
            images_dir = Path(images_base_dir) / 'sa1b-images'
        else:
            # Default to metaclip
            images_dir = Path(images_base_dir) / 'metaclip-images'
        
        try:
            dataset = load_saco_dataset(
                str(gt_dir),
                annotation_file=ann_file.name,
                images_dir=str(images_dir),
            )
            
            print(f"  Loaded {len(dataset.concept_names)} concepts")
            
            # Merge concepts
            for concept_name in dataset.concept_names:
                concept = dataset[concept_name]
                
                if concept_name in merged_concepts:
                    # Append to existing
                    existing = merged_concepts[concept_name]
                    existing.positive_image_paths.extend(concept.positive_image_paths)
                    existing.positive_masks.extend(concept.positive_masks)
                    existing.positive_bboxes.extend(concept.positive_bboxes)
                    existing.positive_pair_ids.extend(concept.positive_pair_ids)
                    existing.negative_image_paths.extend(concept.negative_image_paths)
                    existing.negative_pair_ids.extend(concept.negative_pair_ids)
                else:
                    # New concept
                    merged_concepts[concept_name] = SaCoConceptData(
                        text_input=concept.text_input,
                        positive_image_paths=list(concept.positive_image_paths),
                        positive_masks=list(concept.positive_masks),
                        positive_bboxes=list(concept.positive_bboxes),
                        positive_pair_ids=list(concept.positive_pair_ids),
                        negative_image_paths=list(concept.negative_image_paths),
                        negative_pair_ids=list(concept.negative_pair_ids),
                    )
                    
        except Exception as e:
            print(f"  Error loading {ann_file.name}: {e}")
            continue
    
    # Create final merged dataset
    merged_dataset = SaCoDataset(concepts=merged_concepts)
    
    print(f"\n=== MERGED DATASET ===")
    print(f"Total unique concepts: {len(merged_dataset.concept_names)}")
    total_images = sum(merged_dataset[c].num_positive_images for c in merged_dataset.concept_names)
    print(f"Total image-concept pairs: {total_images}")
    
    return merged_dataset


def print_multi_instance_report(
    dataset,
    top_k: int = 20,
):
    """Print a summary report of multi-instance concepts."""
    print("=" * 90)
    print("MULTI-INSTANCE CONCEPT ANALYSIS (with area filtering)")
    print("=" * 90)
    
    print(f"\n--- Top {top_k} Concepts by Score ---")
    print(f"{'Concept':<35} {'Score':<8} {'Pos':<6} {'Multi':<6} {'Ratio':<8} {'Max':<5} {'MedCov':<8}")
    print("-" * 90)
    
    stats = analyze_multi_instance_concepts(
        dataset,
        min_positive_images=5,
        sort_by='score',
    )
    
    for s in stats[:top_k]:
        print(f"{s.concept_name[:34]:<35} {s.score:<8.1f} {s.num_positive_images:<6} "
              f"{s.num_multi_instance_images:<6} {s.multi_instance_ratio:<8.2%} "
              f"{s.max_instances_per_image:<5} {s.median_coverage:<8.3f}")
    
    # Recommended concepts
    print(f"\n--- Recommended Test Concepts (filtered) ---")
    recommended = get_recommended_test_concepts(dataset, top_k=top_k)
    
    print(f"\nFound {len(recommended)} concepts meeting criteria:")
    for i, r in enumerate(recommended, 1):
        print(f"  {i:2d}. {r['concept'][:40]:<40} score={r['score']:.1f} "
              f"pos={r['num_positive_images']} multi={r['num_multi_instance_images']} "
              f"max={r['max_instances']} cov={r['median_coverage']:.3f}")


def save_analysis(
    dataset,
    output_path: str,
    top_k: int = 50,
):
    """Save full analysis to JSON file."""
    
    # Get all stats
    all_stats = analyze_multi_instance_concepts(
        dataset,
        min_positive_images=5,
        sort_by='score',
    )
    
    # Get recommended
    recommended = get_recommended_test_concepts(dataset, top_k=top_k)
    
    analysis = {
        'total_concepts_analyzed': len(all_stats),
        'recommended_concepts': recommended,
        'all_concept_stats': [s.to_dict() for s in all_stats[:100]],  # Top 100
    }
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Analysis saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python multi_instance_analysis.py <gt_annotations_dir> <images_base_dir> [output_json]")
        sys.exit(1)
    
    gt_dir = sys.argv[1]
    images_dir = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "multi_instance_analysis.json"
    
    # Load all annotations
    dataset = load_all_saco_annotations(gt_dir, images_dir)
    
    # Print report
    print_multi_instance_report(dataset, top_k=30)
    
    # Save analysis
    save_analysis(dataset, output_path, top_k=50)