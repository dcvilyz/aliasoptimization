#!/usr/bin/env python3
"""
Alias Optimization for SAM3 - Main Entry Point

Given a dataset with images and masks (COCO format), finds optimal
token sequences that maximize SAM3's mask reconstruction fidelity.

Usage:
    python main.py --dataset /path/to/dataset --split train
    
    # With custom config
    python main.py --dataset /path/to/dataset --output results/ --soft-steps 100
"""

import argparse
import sys
import torch
from pathlib import Path

# Add parent directory for imports when running as script
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_config
from data_loader import COCODataset
from optimizer import AliasOptimizer, save_all_results, print_results_summary
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize aliases for SAM3 concept segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python main.py --dataset /path/to/coco/dataset
    
    # Specific split
    python main.py --dataset /path/to/dataset --split valid
    
    # Custom optimization parameters
    python main.py --dataset /path/to/dataset --soft-steps 200 --generations 150
    
    # Optimize single category
    python main.py --dataset /path/to/dataset --category-id 1
        """
    )
    
    # Required
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to COCO format dataset",
    )
    
    # Optional
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use (default: mps for Apple Silicon)",
    )
    parser.add_argument(
        "--category-id",
        type=int,
        default=None,
        help="Optimize only this category ID (default: all categories)",
    )
    
    # Model
    parser.add_argument(
        "--sam3-checkpoint",
        type=str,
        default=None,
        help="Path to SAM3 checkpoint (uses default if not specified)",
    )
    
    # Optimization parameters
    parser.add_argument(
        "--soft-steps",
        type=int,
        default=200,
        help="Number of soft prompt optimization steps (default: 200)",
    )
    parser.add_argument(
        "--soft-lr",
        type=float,
        default=0.15,
        help="Learning rate for soft prompt optimization (default: 0.1)",
    )
    parser.add_argument(
        "--soft-length",
        type=int,
        default=5,
        help="Soft prompt sequence length (default: 5)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of evolutionary search generations (default: 100)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=50,
        help="Evolutionary search population size (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation (default: 4)",
    )
    
    # Misc
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Don't compare against class name baseline",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="Random seed (default: 99)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )
    
    return parser.parse_args()


def load_sam3_model(checkpoint_path: str = None, device: str = 'mps'):
    """
    Load SAM3 model.
    
    Note: This function needs to be adapted to your SAM3 installation.
    """
    """
    Load SAM3 image model on the given device.

    - If checkpoint_path is None: use the default HF checkpoint via build_sam3_image_model().
    - If checkpoint_path is set: try to load that .pt file as a state_dict.
    """
    print(f"Loading SAM3 model on device: {device}")

    try:
        import torch
        from sam3.model_builder import build_sam3_image_model
    except ImportError as e:
        print(f"Error importing SAM3: {e}")
        print(
            "\nMake sure the SAM3 repo is installed, e.g.\n"
            "  cd /path/to/sam3 && pip install -e ."
        )
        raise

    # Build the base model (this will pull weights from HF if you’re authenticated)
    model = build_sam3_image_model()

    # Optionally override weights from a local checkpoint
    if checkpoint_path is not None:
        print(f"Loading SAM3 weights from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")

        # Handle either a raw state_dict or a dict with a 'model' key
        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  [warn] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()

    return model


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create config
    config = get_config(
        dataset_path=args.dataset,
        device=args.device,
        output_dir=args.output,
        seed=args.seed,
    )
    
    # Override config with command line args
    config.optimization.soft_steps = args.soft_steps
    config.optimization.soft_lr = args.soft_lr
    config.optimization.soft_prompt_length = args.soft_length
    config.optimization.evolution_generations = args.generations
    config.optimization.population_size = args.population
    config.optimization.soft_batch_size = args.batch_size
    
    verbose = args.verbose and not args.quiet
    
    if verbose:
        print("=" * 60)
        print("SAM3 ALIAS OPTIMIZATION")
        print("=" * 60)
        print(f"\nDataset: {args.dataset}")
        print(f"Split: {args.split}")
        print(f"Device: {args.device}")
        print(f"Output: {args.output}")
        print(f"\nOptimization settings:")
        print(f"  Soft prompt steps: {args.soft_steps}")
        print(f"  Soft prompt length: {args.soft_length}")
        print(f"  Evolution generations: {args.generations}")
        print(f"  Population size: {args.population}")
        print("=" * 60)
    
    # Load dataset
    if verbose:
        print("\nLoading dataset...")
    
    try:
        dataset = COCODataset(
            dataset_path=args.dataset,
            split=args.split,
        )
        if verbose:
            print(dataset.summary())
    except FileNotFoundError as e:
        print(f"Error: Could not load dataset: {e}")
        sys.exit(1)
    
    # Load model
    if verbose:
        print("\nLoading SAM3 model...")
    
    try:
        model = load_sam3_model(
            checkpoint_path=args.sam3_checkpoint,
            device=args.device,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create optimizer
    optimizer = AliasOptimizer(
        model=model,
        config=config,
    )
    
    # Run optimization
    if args.category_id is not None:
        # Single category
        if args.category_id not in dataset.category_ids:
            print(f"Error: Category ID {args.category_id} not found in dataset")
            print(f"Available categories: {dataset.category_ids}")
            sys.exit(1)
        
        category_data = dataset.get_category_data(args.category_id, max_images=10)
        baseline_text = category_data.category_name if not args.no_baseline else None
        
        result = optimizer.optimize_category(
            category_data=category_data,
            baseline_text=baseline_text,
            verbose=verbose,
        )
        
        # Save result
        output_path = Path(args.output) / f"result_cat_{args.category_id}.json"
        result.save(str(output_path))
        
        if verbose:
            print(f"\nResult saved to: {output_path}")
        
    else:
        # All categories
        results = optimizer.optimize_dataset(
            dataset=dataset,
            use_class_names_as_baseline=not args.no_baseline,
            verbose=verbose,
        )
        
        # Save all results
        output_path = Path(args.output) / "all_results.json"
        save_all_results(results, str(output_path))
        
        # Print summary
        if verbose:
            print_results_summary(results)
    
    if verbose:
        print("\nDone!")


if __name__ == "__main__":
    main()
