"""
Multi-Instance Analysis for SA-Co Dataset.

Finds concepts with multiple instances per image, which are critical
for testing automated segmentation pipelines that need to find ALL
instances of an object.

Usage:
    from multi_instance_analysis import (
        analyze_multi_instance_concepts,
        find_max_instance_images,
        get_recommended_test_concepts,
    )
    
    # Find concepts with most multi-instance images
    results = analyze_multi_instance_concepts(dataset)
    
    # Find images with most total instances (any concept)
    top_images = find_max_instance_images(dataset, top_k=50)
    
    # Get recommended concepts for multi-instance testing
    recommended = get_recommended_test_concepts(dataset, min_multi_instance_images=5)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json


@dataclass
class ConceptInstanceStats:
    """Statistics about instance counts for a concept."""
    concept_name: str
    
    # Image counts
    num_positive_images: int
    num_single_instance_images: int  # Images with exactly 1 instance
    num_multi_instance_images: int   # Images with 2+ instances
    
    # Instance counts
    total_instances: int
    max_instances_per_image: int
    mean_instances_per_image: float
    
    # Instance distribution
    instance_distribution: Dict[int, int]  # {num_instances: count}
    
    # Multi-instance ratio
    multi_instance_ratio: float  # num_multi / num_positive
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageInstanceStats:
    """Statistics about a single image."""
    image_path: str
    concept_name: str
    pair_id: str
    num_instances: int
    mask_areas: List[int]  # Pixel area of each mask
    total_mask_area: int
    image_coverage: float  # Total mask area / image area (if known)


def analyze_concept_instances(concept) -> ConceptInstanceStats:
    """
    Analyze instance counts for a single concept.
    
    Args:
        concept: SaCoConceptData
        
    Returns:
        ConceptInstanceStats with detailed instance information
    """
    instance_counts = []
    
    for masks in concept.positive_masks:
        # masks is a list of np.ndarray for this image
        num_instances = len(masks) if masks else 0
        instance_counts.append(num_instances)
    
    if not instance_counts:
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
        )
    
    # Compute statistics
    instance_counts = np.array(instance_counts)
    num_single = int((instance_counts == 1).sum())
    num_multi = int((instance_counts >= 2).sum())
    
    # Build distribution
    distribution = {}
    for count in instance_counts:
        distribution[int(count)] = distribution.get(int(count), 0) + 1
    
    multi_ratio = num_multi / len(instance_counts) if len(instance_counts) > 0 else 0
    
    return ConceptInstanceStats(
        concept_name=concept.text_input,
        num_positive_images=len(instance_counts),
        num_single_instance_images=num_single,
        num_multi_instance_images=num_multi,
        total_instances=int(instance_counts.sum()),
        max_instances_per_image=int(instance_counts.max()),
        mean_instances_per_image=float(instance_counts.mean()),
        instance_distribution=distribution,
        multi_instance_ratio=multi_ratio,
    )


def analyze_multi_instance_concepts(
    dataset,  # SaCoDataset
    min_positive_images: int = 3,
    sort_by: str = 'num_multi_instance_images',
) -> List[ConceptInstanceStats]:
    """
    Analyze all concepts for multi-instance statistics.
    
    Args:
        dataset: SaCoDataset
        min_positive_images: Minimum positive images to include concept
        sort_by: Field to sort by. Options:
            - 'num_multi_instance_images': Most images with 2+ instances
            - 'multi_instance_ratio': Highest % of images with 2+ instances
            - 'max_instances_per_image': Highest max instances in any single image
            - 'mean_instances_per_image': Highest average instances
            - 'total_instances': Most total instances across all images
            
    Returns:
        List of ConceptInstanceStats sorted by specified field
    """
    results = []
    
    for concept_name in dataset.concept_names:
        concept = dataset[concept_name]
        
        if concept.num_positive_images < min_positive_images:
            continue
        
        stats = analyze_concept_instances(concept)
        results.append(stats)
    
    # Sort
    if sort_by == 'num_multi_instance_images':
        results.sort(key=lambda x: x.num_multi_instance_images, reverse=True)
    elif sort_by == 'multi_instance_ratio':
        results.sort(key=lambda x: x.multi_instance_ratio, reverse=True)
    elif sort_by == 'max_instances_per_image':
        results.sort(key=lambda x: x.max_instances_per_image, reverse=True)
    elif sort_by == 'mean_instances_per_image':
        results.sort(key=lambda x: x.mean_instances_per_image, reverse=True)
    elif sort_by == 'total_instances':
        results.sort(key=lambda x: x.total_instances, reverse=True)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by}")
    
    return results


def find_max_instance_images(
    dataset,  # SaCoDataset
    top_k: int = 50,
    compute_coverage: bool = True,
) -> List[ImageInstanceStats]:
    """
    Find images with the most instances (of any concept).
    
    This finds the "hardest" images for instance segmentation -
    those with many objects to detect.
    
    Args:
        dataset: SaCoDataset
        top_k: Number of top images to return
        compute_coverage: Whether to load images and compute actual coverage
        
    Returns:
        List of ImageInstanceStats sorted by num_instances descending
    """
    from PIL import Image
    
    all_images = []
    
    for concept_name in dataset.concept_names:
        concept = dataset[concept_name]
        
        for idx in range(concept.num_positive_images):
            masks = concept.positive_masks[idx]
            image_path = concept.positive_image_paths[idx]
            pair_id = concept.positive_pair_ids[idx] if concept.positive_pair_ids else f"{idx}"
            
            num_instances = len(masks) if masks else 0
            
            # Compute mask areas
            mask_areas = []
            for mask in (masks or []):
                if isinstance(mask, np.ndarray):
                    mask_areas.append(int(mask.sum()))
                else:
                    mask_areas.append(0)
            
            total_area = sum(mask_areas)
            
            all_images.append(ImageInstanceStats(
                image_path=image_path,
                concept_name=concept_name,
                pair_id=pair_id,
                num_instances=num_instances,
                mask_areas=mask_areas,
                total_mask_area=total_area,
                image_coverage=0,  # Computed later for top images
            ))
    
    # Sort by num_instances descending
    all_images.sort(key=lambda x: x.num_instances, reverse=True)
    
    # Take top_k
    top_images = all_images[:top_k]
    
    # Now compute actual coverage for top images by loading them
    if compute_coverage:
        print(f"Computing coverage for top {len(top_images)} images...")
        seen_paths = {}  # Cache image sizes
        
        for i, img_stat in enumerate(top_images):
            path = img_stat.image_path
            
            # Check cache first
            if path in seen_paths:
                image_area = seen_paths[path]
            else:
                try:
                    with Image.open(path) as img:
                        w, h = img.size
                        image_area = w * h
                        seen_paths[path] = image_area
                except Exception as e:
                    print(f"  Warning: Could not load {path}: {e}")
                    image_area = None
            
            if image_area and image_area > 0:
                img_stat.image_coverage = img_stat.total_mask_area / image_area
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(top_images)}")
        
        print(f"  Done. Loaded {len(seen_paths)} unique images.")
    
    return top_images


def find_concepts_in_max_instance_images(
    dataset,  # SaCoDataset
    top_k_images: int = 100,
) -> Dict[str, dict]:
    """
    Find which concepts appear most frequently in high-instance images.
    
    This identifies concepts that are good for multi-instance testing
    because they tend to appear in images with many instances.
    
    Args:
        dataset: SaCoDataset
        top_k_images: Number of top instance images to consider
        
    Returns:
        Dict mapping concept_name to stats about its presence in top images
    """
    # Get top instance images
    top_images = find_max_instance_images(dataset, top_k=top_k_images)
    
    # Count concept occurrences in top images
    concept_counts = defaultdict(lambda: {
        'appearances': 0,
        'total_instances_in_top_images': 0,
        'images_with_max_instances': [],
    })
    
    for img in top_images:
        concept = img.concept_name
        concept_counts[concept]['appearances'] += 1
        concept_counts[concept]['total_instances_in_top_images'] += img.num_instances
        concept_counts[concept]['images_with_max_instances'].append({
            'image_path': img.image_path,
            'num_instances': img.num_instances,
        })
    
    # Sort by appearances
    sorted_concepts = sorted(
        concept_counts.items(),
        key=lambda x: x[1]['appearances'],
        reverse=True
    )
    
    return dict(sorted_concepts)


def get_recommended_test_concepts(
    dataset,  # SaCoDataset
    min_multi_instance_images: int = 5,
    min_max_instances: int = 3,
    min_positive_images: int = 10,
) -> List[dict]:
    """
    Get recommended concepts for multi-instance testing.
    
    Filters for concepts that have:
    1. Enough total positive images for train/test split
    2. Enough multi-instance images for meaningful evaluation
    3. High enough max instances to test edge cases
    
    Args:
        dataset: SaCoDataset
        min_multi_instance_images: Minimum images with 2+ instances
        min_max_instances: Minimum max instances in any image
        min_positive_images: Minimum total positive images
        
    Returns:
        List of recommended concepts with their stats
    """
    all_stats = analyze_multi_instance_concepts(
        dataset,
        min_positive_images=min_positive_images,
        sort_by='num_multi_instance_images',
    )
    
    recommended = []
    for stats in all_stats:
        if (stats.num_multi_instance_images >= min_multi_instance_images and
            stats.max_instances_per_image >= min_max_instances):
            
            recommended.append({
                'concept': stats.concept_name,
                'num_positive_images': stats.num_positive_images,
                'num_multi_instance_images': stats.num_multi_instance_images,
                'multi_instance_ratio': stats.multi_instance_ratio,
                'max_instances': stats.max_instances_per_image,
                'mean_instances': stats.mean_instances_per_image,
                'total_instances': stats.total_instances,
                'distribution': stats.instance_distribution,
            })
    
    return recommended


def print_multi_instance_report(
    dataset,  # SaCoDataset
    top_k: int = 20,
):
    """
    Print a summary report of multi-instance concepts.
    """
    print("=" * 80)
    print("MULTI-INSTANCE CONCEPT ANALYSIS")
    print("=" * 80)
    
    # Top by number of multi-instance images
    print(f"\n--- Top {top_k} Concepts by # Multi-Instance Images ---")
    print(f"{'Concept':<30} {'Pos':<6} {'Multi':<6} {'Ratio':<8} {'Max':<5} {'Mean':<6}")
    print("-" * 70)
    
    stats = analyze_multi_instance_concepts(
        dataset,
        min_positive_images=3,
        sort_by='num_multi_instance_images',
    )
    
    for s in stats[:top_k]:
        print(f"{s.concept_name:<30} {s.num_positive_images:<6} {s.num_multi_instance_images:<6} "
              f"{s.multi_instance_ratio:<8.2%} {s.max_instances_per_image:<5} {s.mean_instances_per_image:<6.2f}")
    
    # Top by max instances in single image
    print(f"\n--- Top {top_k} Concepts by Max Instances in Single Image ---")
    print(f"{'Concept':<30} {'Max':<5} {'Pos':<6} {'Multi':<6} {'Distribution':<30}")
    print("-" * 80)
    
    stats = analyze_multi_instance_concepts(
        dataset,
        min_positive_images=3,
        sort_by='max_instances_per_image',
    )
    
    for s in stats[:top_k]:
        dist_str = str(dict(sorted(s.instance_distribution.items())))[:30]
        print(f"{s.concept_name:<30} {s.max_instances_per_image:<5} {s.num_positive_images:<6} "
              f"{s.num_multi_instance_images:<6} {dist_str:<30}")
    
    # Top images by instance count
    print(f"\n--- Top {top_k} Images by Instance Count ---")
    print(f"{'Concept':<25} {'Instances':<10} {'Image Path':<40}")
    print("-" * 80)
    
    top_images = find_max_instance_images(dataset, top_k=top_k)
    for img in top_images:
        path_short = str(img.image_path)[-40:] if len(str(img.image_path)) > 40 else str(img.image_path)
        print(f"{img.concept_name:<25} {img.num_instances:<10} {path_short:<40}")
    
    # Concepts dominating high-instance images
    print(f"\n--- Concepts Most Present in Top 100 High-Instance Images ---")
    concept_presence = find_concepts_in_max_instance_images(dataset, top_k_images=100)
    
    print(f"{'Concept':<30} {'Appearances':<12} {'Total Instances':<15}")
    print("-" * 60)
    
    for concept, info in list(concept_presence.items())[:top_k]:
        print(f"{concept:<30} {info['appearances']:<12} {info['total_instances_in_top_images']:<15}")
    
    # Recommended test concepts
    print(f"\n--- Recommended Concepts for Multi-Instance Testing ---")
    recommended = get_recommended_test_concepts(dataset)
    
    if recommended:
        print(f"Found {len(recommended)} concepts meeting criteria:")
        print(f"  - min_multi_instance_images >= 5")
        print(f"  - max_instances >= 3")
        print(f"  - min_positive_images >= 10")
        print()
        
        for r in recommended[:10]:
            print(f"  {r['concept']}: {r['num_multi_instance_images']} multi-instance images, "
                  f"max={r['max_instances']}, mean={r['mean_instances']:.2f}")
    else:
        print("No concepts meet the default criteria. Try relaxing thresholds.")
    
    print("=" * 80)


def save_analysis_report(
    dataset,  # SaCoDataset
    output_path: str,
):
    """
    Save comprehensive analysis to JSON file.
    """
    report = {
        'dataset_info': {
            'num_concepts': len(dataset.concept_names),
            'path': str(dataset.dataset_path),
        },
        'concept_stats': [],
        'top_images': [],
        'concepts_in_top_images': {},
        'recommended_test_concepts': [],
    }
    
    # All concept stats
    all_stats = analyze_multi_instance_concepts(dataset, min_positive_images=1, sort_by='num_multi_instance_images')
    report['concept_stats'] = [s.to_dict() for s in all_stats]
    
    # Top images
    top_images = find_max_instance_images(dataset, top_k=200)
    report['top_images'] = [asdict(img) for img in top_images]
    
    # Concepts in top images
    report['concepts_in_top_images'] = find_concepts_in_max_instance_images(dataset, top_k_images=100)
    
    # Recommended
    report['recommended_test_concepts'] = get_recommended_test_concepts(dataset)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Analysis saved to: {output_path}")


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze SA-Co for multi-instance concepts')
    parser.add_argument('--dataset', type=str, required=True, help='Path to SA-Co dataset')
    parser.add_argument('--images_dir', type=str, default=None, help='Path to images')
    parser.add_argument('--output', type=str, default=None, help='Save JSON report to this path')
    parser.add_argument('--top_k', type=int, default=20, help='Number of top results to show')
    
    args = parser.parse_args()
    
    from saco_loader import load_saco_dataset
    
    print(f"Loading dataset from: {args.dataset}")
    dataset = load_saco_dataset(args.dataset, images_dir=args.images_dir)
    print(f"Loaded {len(dataset.concept_names)} concepts")
    
    print_multi_instance_report(dataset, top_k=args.top_k)
    
    if args.output:
        save_analysis_report(dataset, args.output)