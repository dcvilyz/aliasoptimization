"""
Evaluation Pipeline: Baseline SAM3 vs Adversarial Prompt Search

Protocol:
1. Split SA-Co concepts into train/test images
2. Baseline: SAM3 + ground truth labels evaluated on test
3. Ours: Search prompts on train, evaluate on test
4. Compare fitness, log compute time

Usage:
    python eval_pipeline.py --dataset /path/to/saco --concepts "draft weapon,cyclocomputer" --train_ratio 0.7
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import random


# =============================================================================
# PRESTART CHECKS
# =============================================================================

class PrestartCheckError(Exception):
    """Raised when prestart checks fail."""
    pass


def check_imports() -> Dict[str, bool]:
    """Check all required imports are available."""
    results = {}
    
    required_modules = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('psutil', 'psutil (for hardware info)'),
        ('PIL', 'Pillow (for image loading)'),
    ]
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            results[display_name] = True
        except ImportError:
            results[display_name] = False
    
    # Check project modules
    project_modules = [
        ('saco_loader', 'SA-Co data loader'),
        ('data_loader', 'COCO data loader'),
        ('augmentation', 'Augmentation module'),
        ('augmentation_integration', 'Augmentation integration'),
        ('discrete_search', 'Discrete search optimizer'),
        ('config', 'Config module'),
    ]
    
    for module_name, display_name in project_modules:
        try:
            __import__(module_name)
            results[display_name] = True
        except ImportError:
            results[display_name] = False
    
    return results


def check_hardware(device: str) -> Tuple[bool, Dict, List[str]]:
    """
    Check hardware is accessible and collect info.
    
    Returns:
        (success, hardware_info_dict, error_messages)
    """
    errors = []
    hw_info = {}
    
    # Basic system info
    try:
        import platform
        import psutil
        hw_info['cpu_name'] = platform.processor() or "Unknown"
        hw_info['cpu_cores'] = psutil.cpu_count(logical=False)
        hw_info['ram_gb'] = psutil.virtual_memory().total / (1024**3)
    except Exception as e:
        errors.append(f"Failed to get CPU/RAM info: {e}")
    
    # Device-specific checks
    if device == 'cuda':
        if not torch.cuda.is_available():
            errors.append("CUDA requested but torch.cuda.is_available() = False")
        else:
            try:
                hw_info['device_name'] = torch.cuda.get_device_name(0)
                hw_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                major, minor = torch.cuda.get_device_capability(0)
                hw_info['compute_capability'] = f"{major}.{minor}"
                
                # Test allocation
                test_tensor = torch.zeros(1000, 1000, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                errors.append(f"CUDA device error: {e}")
                
    elif device == 'mps':
        if not torch.backends.mps.is_available():
            errors.append("MPS requested but torch.backends.mps.is_available() = False")
        else:
            try:
                # Test allocation
                test_tensor = torch.zeros(1000, 1000, device='mps')
                del test_tensor
                
                # Get Apple Silicon info
                import subprocess
                result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                       capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    if 'Chip' in line:
                        hw_info['device_name'] = line.split(':')[-1].strip()
                    if 'Memory' in line and 'GB' in line:
                        try:
                            mem_str = line.split(':')[-1].strip()
                            hw_info['gpu_memory_gb'] = float(mem_str.replace('GB', '').strip())
                        except:
                            pass
            except subprocess.TimeoutExpired:
                errors.append("Timeout getting Apple Silicon info")
            except Exception as e:
                errors.append(f"MPS device error: {e}")
    
    elif device == 'cpu':
        try:
            test_tensor = torch.zeros(1000, 1000, device='cpu')
            del test_tensor
            hw_info['device_name'] = 'CPU'
        except Exception as e:
            errors.append(f"CPU tensor error: {e}")
    
    success = len(errors) == 0
    return success, hw_info, errors


def check_model(device: str) -> Tuple[bool, List[str]]:
    """
    Check model can be loaded and run.
    
    Returns:
        (success, error_messages)
    """
    errors = []
    
    try:
        from sam3.model_builder import build_sam3_image_model
    except ImportError as e:
        errors.append(f"Cannot import SAM3: {e}")
        return False, errors
    
    try:
        model = build_sam3_image_model()
        model.to(device)
        model.eval()
    except Exception as e:
        errors.append(f"Failed to load/move model: {e}")
        return False, errors
    
    # Test forward pass
    try:
        with torch.no_grad():
            dummy_image = torch.randn(1, 3, 1008, 1008, device=device)
            # Just test image encoding
            _ = model.backbone.forward_image(dummy_image)
    except Exception as e:
        errors.append(f"Model forward pass failed: {e}")
        return False, errors
    
    # Test tokenizer
    try:
        tokenizer = model.backbone.language_backbone.tokenizer
        tokens = tokenizer.encode("test prompt")
        decoded = tokenizer.decode(tokens)
    except Exception as e:
        errors.append(f"Tokenizer error: {e}")
        return False, errors
    
    # Cleanup
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return True, errors


def check_dataset(dataset_path: str, images_dir: Optional[str] = None) -> Tuple[bool, Dict, List[str]]:
    """
    Check dataset is accessible and valid.
    
    Returns:
        (success, dataset_info, error_messages)
    """
    errors = []
    info = {}
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        errors.append(f"Dataset path does not exist: {dataset_path}")
        return False, info, errors
    
    # Try to load dataset
    try:
        from saco_loader import load_saco_dataset
        dataset = load_saco_dataset(str(dataset_path), images_dir=images_dir)
        info['num_concepts'] = len(dataset.concepts)
        info['concept_names'] = dataset.concept_names[:5]  # First 5
    except Exception as e:
        errors.append(f"Failed to load dataset: {e}")
        return False, info, errors
    
    # Check at least one concept has images
    if info['num_concepts'] == 0:
        errors.append("Dataset has no concepts")
        return False, info, errors
    
    # Try to load one image
    try:
        first_concept = dataset[dataset.concept_names[0]]
        if first_concept.num_positive_images == 0:
            errors.append(f"First concept '{first_concept.text_input}' has no images")
            return False, info, errors
        
        first_image_path = first_concept.positive_image_paths[0]
        if not Path(first_image_path).exists():
            errors.append(f"Image file not found: {first_image_path}")
            return False, info, errors
        
        # Try to open it
        from PIL import Image
        img = Image.open(first_image_path)
        info['sample_image_size'] = img.size
        img.close()
        
    except Exception as e:
        errors.append(f"Failed to load sample image: {e}")
        return False, info, errors
    
    return True, info, errors


def check_vocab_embeddings() -> Tuple[bool, Dict, List[str]]:
    """
    Check vocabulary embeddings are available.
    
    Returns:
        (success, info, error_messages)
    """
    errors = []
    info = {}
    
    try:
        from embeddings import load_both_vocab_embeddings
        vocab_256, vocab_1024 = load_both_vocab_embeddings()
        info['vocab_256_shape'] = (vocab_256.vocab_size, vocab_256.embed_dim)
        info['vocab_1024_shape'] = (vocab_1024.vocab_size, vocab_1024.embed_dim)
    except ImportError:
        # Try alternative loading
        try:
            from embeddings import VocabularyEmbeddings
            # Check if cache files exist
            cache_256 = Path('cache/token_activation_map_256d.pt')
            cache_1024 = Path('cache/vocab_embeddings_1024d.pt')
            
            if not cache_256.exists():
                errors.append(f"256-dim embeddings not found at {cache_256}")
            if not cache_1024.exists():
                errors.append(f"1024-dim embeddings not found at {cache_1024}")
                
            if errors:
                return False, info, errors
                
            info['vocab_256_path'] = str(cache_256)
            info['vocab_1024_path'] = str(cache_1024)
        except Exception as e:
            errors.append(f"Failed to check embeddings: {e}")
            return False, info, errors
    except Exception as e:
        errors.append(f"Failed to load embeddings: {e}")
        return False, info, errors
    
    return True, info, errors


def check_disk_space(output_dir: str, min_gb: float = 1.0) -> Tuple[bool, Dict, List[str]]:
    """
    Check sufficient disk space for outputs.
    
    Returns:
        (success, info, error_messages)
    """
    import shutil
    errors = []
    info = {}
    
    output_path = Path(output_dir)
    
    # Get disk usage for the partition containing output_dir
    try:
        # Create output dir if it doesn't exist (to check its partition)
        output_path.mkdir(parents=True, exist_ok=True)
        
        usage = shutil.disk_usage(output_path)
        info['total_gb'] = usage.total / (1024**3)
        info['free_gb'] = usage.free / (1024**3)
        info['used_percent'] = (usage.used / usage.total) * 100
        
        if info['free_gb'] < min_gb:
            errors.append(f"Insufficient disk space: {info['free_gb']:.1f} GB free, need {min_gb} GB")
            
    except Exception as e:
        errors.append(f"Failed to check disk space: {e}")
        return False, info, errors
    
    return len(errors) == 0, info, errors


def run_prestart_checks(
    device: str,
    dataset_path: str,
    images_dir: Optional[str],
    output_dir: str,
    verbose: bool = True,
) -> Tuple[bool, Dict]:
    """
    Run all prestart checks.
    
    Returns:
        (all_passed, full_report)
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'checks': {},
        'all_passed': False,
    }
    
    all_passed = True
    
    if verbose:
        print("\n" + "="*70)
        print("PRESTART CHECKS")
        print("="*70)
    
    # 1. Imports
    if verbose:
        print("\n[1/6] Checking imports...")
    import_results = check_imports()
    report['checks']['imports'] = import_results
    imports_ok = all(import_results.values())
    if verbose:
        for name, ok in import_results.items():
            status = "✓" if ok else "✗"
            print(f"  {status} {name}")
    if not imports_ok:
        all_passed = False
        if verbose:
            print("  FAILED: Missing required imports")
    
    # 2. Hardware
    if verbose:
        print(f"\n[2/6] Checking hardware ({device})...")
    hw_ok, hw_info, hw_errors = check_hardware(device)
    report['checks']['hardware'] = {'success': hw_ok, 'info': hw_info, 'errors': hw_errors}
    if verbose:
        if hw_ok:
            print(f"  ✓ Device: {hw_info.get('device_name', 'Unknown')}")
            if 'gpu_memory_gb' in hw_info:
                print(f"  ✓ GPU Memory: {hw_info['gpu_memory_gb']:.1f} GB")
            print(f"  ✓ CPU: {hw_info.get('cpu_name', 'Unknown')} ({hw_info.get('cpu_cores', '?')} cores)")
            print(f"  ✓ RAM: {hw_info.get('ram_gb', 0):.1f} GB")
        else:
            for err in hw_errors:
                print(f"  ✗ {err}")
    if not hw_ok:
        all_passed = False
    
    # 3. Model
    if verbose:
        print(f"\n[3/6] Checking model loading...")
    model_ok, model_errors = check_model(device)
    report['checks']['model'] = {'success': model_ok, 'errors': model_errors}
    if verbose:
        if model_ok:
            print("  ✓ Model loads successfully")
            print("  ✓ Forward pass works")
            print("  ✓ Tokenizer works")
        else:
            for err in model_errors:
                print(f"  ✗ {err}")
    if not model_ok:
        all_passed = False
    
    # 4. Dataset
    if verbose:
        print(f"\n[4/6] Checking dataset...")
    dataset_ok, dataset_info, dataset_errors = check_dataset(dataset_path, images_dir)
    report['checks']['dataset'] = {'success': dataset_ok, 'info': dataset_info, 'errors': dataset_errors}
    if verbose:
        if dataset_ok:
            print(f"  ✓ Dataset loaded: {dataset_info.get('num_concepts', 0)} concepts")
            print(f"  ✓ Sample concepts: {dataset_info.get('concept_names', [])}")
            print(f"  ✓ Images accessible")
        else:
            for err in dataset_errors:
                print(f"  ✗ {err}")
    if not dataset_ok:
        all_passed = False
    
    # 5. Vocab embeddings
    if verbose:
        print(f"\n[5/6] Checking vocabulary embeddings...")
    vocab_ok, vocab_info, vocab_errors = check_vocab_embeddings()
    report['checks']['vocab_embeddings'] = {'success': vocab_ok, 'info': vocab_info, 'errors': vocab_errors}
    if verbose:
        if vocab_ok:
            if 'vocab_256_shape' in vocab_info:
                print(f"  ✓ 256-dim embeddings: {vocab_info['vocab_256_shape']}")
                print(f"  ✓ 1024-dim embeddings: {vocab_info['vocab_1024_shape']}")
            else:
                print(f"  ✓ Embedding files found")
        else:
            for err in vocab_errors:
                print(f"  ✗ {err}")
    if not vocab_ok:
        all_passed = False
    
    # 6. Disk space
    if verbose:
        print(f"\n[6/6] Checking disk space...")
    disk_ok, disk_info, disk_errors = check_disk_space(output_dir)
    report['checks']['disk_space'] = {'success': disk_ok, 'info': disk_info, 'errors': disk_errors}
    if verbose:
        if disk_ok:
            print(f"  ✓ Free space: {disk_info.get('free_gb', 0):.1f} GB")
            print(f"  ✓ Output dir: {output_dir}")
        else:
            for err in disk_errors:
                print(f"  ✗ {err}")
    if not disk_ok:
        all_passed = False
    
    # Summary
    report['all_passed'] = all_passed
    
    if verbose:
        print("\n" + "="*70)
        if all_passed:
            print("✓ ALL PRESTART CHECKS PASSED")
        else:
            print("✗ PRESTART CHECKS FAILED")
            print("  Fix the errors above before running evaluation.")
        print("="*70 + "\n")
    
    return all_passed, report


# =============================================================================
# HARDWARE INFO & COMPUTE METRICS
# =============================================================================


@dataclass
class HardwareInfo:
    """Hardware specification for reproducibility."""
    device_type: str  # 'mps', 'cuda', 'cpu'
    device_name: str  # e.g., 'Apple M4 Pro', 'NVIDIA A100', 'AMD EPYC 7742'
    gpu_memory_gb: Optional[float]  # GPU memory if applicable
    compute_capability: Optional[str]  # CUDA compute capability if applicable
    cpu_name: str
    cpu_cores: int
    ram_gb: float
    
    
@dataclass
class ComputeMetrics:
    """Detailed compute metrics for a single run."""
    # Time
    wall_time_seconds: float
    
    # Forward passes
    num_forward_passes: int
    
    # FLOPs (estimated for SAM3)
    # SAM3 image encoder: ~XXX GFLOPs per image
    # SAM3 text encoder: ~XXX GFLOPs per text
    # SAM3 decoder: ~XXX GFLOPs per mask
    flops_per_forward_pass: int  # estimated FLOPs for one eval
    total_flops: int
    
    # Throughput
    forward_passes_per_second: float
    flops_per_second: float  # effective FLOP/s achieved
    
    
def get_hardware_info(device: str) -> HardwareInfo:
    """Collect hardware information."""
    import platform
    import psutil
    
    cpu_name = platform.processor() or "Unknown"
    cpu_cores = psutil.cpu_count(logical=False) or 0
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    device_name = "Unknown"
    gpu_memory_gb = None
    compute_capability = None
    
    if device == 'cuda':
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                major, minor = torch.cuda.get_device_capability(0)
                compute_capability = f"{major}.{minor}"
        except:
            pass
    elif device == 'mps':
        # Apple Silicon - get chip name from system
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                   capture_output=True, text=True)
            cpu_info = result.stdout.strip()
            # Try to identify Apple Silicon
            if 'Apple' in cpu_info or platform.machine() == 'arm64':
                # Get chip info
                result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                       capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if 'Chip' in line:
                        device_name = line.split(':')[-1].strip()
                        break
                # Get unified memory
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                       capture_output=True, text=True)
                gpu_memory_gb = int(result.stdout.strip()) / (1024**3)  # Unified memory
        except:
            device_name = "Apple Silicon (unknown)"
    
    return HardwareInfo(
        device_type=device,
        device_name=device_name,
        gpu_memory_gb=gpu_memory_gb,
        compute_capability=compute_capability,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
    )


def estimate_sam3_flops(image_size: int = 1008, num_images: int = 1) -> int:
    """
    Estimate FLOPs for one SAM3 forward pass.
    
    This is a rough estimate based on model architecture.
    SAM3 uses:
    - Image encoder: ViT-H/16 (~632 GFLOPs for 1024x1024)
    - Text encoder: CLIP text transformer (~1-2 GFLOPs)
    - Mask decoder: Lightweight (~0.5 GFLOPs)
    
    Returns FLOPs as integer.
    """
    # ViT-H image encoder estimate (dominant cost)
    # Roughly 632 GFLOPs for 1024x1024 image
    # Scale by image size ratio squared
    scale = (image_size / 1024) ** 2
    image_encoder_flops = int(632e9 * scale)
    
    # Text encoder (~1.5 GFLOPs)
    text_encoder_flops = int(1.5e9)
    
    # Mask decoder (~0.5 GFLOPs)
    decoder_flops = int(0.5e9)
    
    # Total per image
    per_image = image_encoder_flops + text_encoder_flops + decoder_flops
    
    return per_image * num_images


def compute_metrics(
    wall_time: float, 
    num_forward_passes: int,
    image_size: int = 1008,
    images_per_pass: int = 1,
) -> ComputeMetrics:
    """Compute detailed metrics from wall time and forward pass count."""
    
    flops_per_pass = estimate_sam3_flops(image_size, images_per_pass)
    total_flops = flops_per_pass * num_forward_passes
    
    fps = num_forward_passes / wall_time if wall_time > 0 else 0
    flops_per_sec = total_flops / wall_time if wall_time > 0 else 0
    
    return ComputeMetrics(
        wall_time_seconds=wall_time,
        num_forward_passes=num_forward_passes,
        flops_per_forward_pass=flops_per_pass,
        total_flops=total_flops,
        forward_passes_per_second=fps,
        flops_per_second=flops_per_sec,
    )


@dataclass
class EvalResult:
    """Result for a single concept evaluation."""
    concept: str
    
    # Data split info
    num_train_images: int
    num_test_images: int
    num_train_negatives: int
    num_test_negatives: int
    
    # Search config
    token_count: int  # Number of tokens used in search
    
    # Baseline (GT labels)
    baseline_test_fitness: float
    baseline_test_iou: float
    baseline_eval_time: float  # seconds
    baseline_eval_count: int   # forward passes for baseline eval
    
    # Ours (searched prompts)
    search_train_fitness: float  # fitness on train during search
    search_test_fitness: float   # fitness on held-out test
    search_test_iou: float
    optimized_tokens: List[int]
    optimized_decoded: str
    search_time: float           # seconds for search
    search_evals: int            # number of unique forward passes during search
    eval_time: float             # seconds for test eval
    
    # Comparison
    improvement: float           # search_test - baseline_test
    relative_improvement: float  # improvement / baseline_test
    search_wins: bool
    
    # Compute summary
    total_search_forward_passes: int  # search_evals
    total_time_seconds: float         # search_time + eval_time
    time_per_eval_ms: float           # average time per forward pass


@dataclass 
class EvalSummary:
    """Summary across all concepts."""
    timestamp: str
    
    # Hardware
    hardware: dict  # HardwareInfo as dict
    
    # Experiment config
    num_concepts: int
    train_ratio: float
    image_size: int
    
    # Aggregate performance metrics
    baseline_mean_fitness: float
    search_mean_fitness: float
    mean_improvement: float
    win_rate: float
    
    # Compute costs (measured)
    total_wall_time_seconds: float
    total_forward_passes: int
    total_flops: int  # estimated
    avg_forward_passes_per_concept: float
    avg_wall_time_per_concept: float
    avg_time_per_forward_pass_ms: float
    effective_tflops_per_second: float  # achieved throughput
    
    # Individual results
    results: List[dict]


def split_concept_data(concept, train_ratio: float = 0.7, seed: int = 42):
    """
    Split a concept's images into train/test sets.
    
    Splits BOTH positive and negative images with the same ratio.
    
    Args:
        concept: SaCoConceptData
        train_ratio: Fraction for training
        seed: Random seed for reproducibility
        
    Returns:
        train_concept, test_concept (modified copies with both positives and negatives)
    """
    from saco_loader import SaCoConceptData
    
    random.seed(seed)
    
    # Split positive images
    n_pos = concept.num_positive_images
    pos_indices = list(range(n_pos))
    random.shuffle(pos_indices)
    
    n_pos_train = int(n_pos * train_ratio)
    pos_train_indices = pos_indices[:n_pos_train]
    pos_test_indices = pos_indices[n_pos_train:]
    
    # Split negative images
    n_neg = concept.num_negative_images
    neg_indices = list(range(n_neg))
    random.shuffle(neg_indices)
    
    n_neg_train = int(n_neg * train_ratio)
    neg_train_indices = neg_indices[:n_neg_train]
    neg_test_indices = neg_indices[n_neg_train:]
    
    # Create train concept
    train_concept = SaCoConceptData(text_input=concept.text_input)
    
    # Add positive images to train
    for i in pos_train_indices:
        train_concept.positive_image_paths.append(concept.positive_image_paths[i])
        train_concept.positive_masks.append(concept.positive_masks[i])
        if concept.positive_bboxes:
            train_concept.positive_bboxes.append(concept.positive_bboxes[i])
        train_concept.positive_pair_ids.append(concept.positive_pair_ids[i])
    
    # Add negative images to train
    for i in neg_train_indices:
        train_concept.negative_image_paths.append(concept.negative_image_paths[i])
        train_concept.negative_pair_ids.append(concept.negative_pair_ids[i])
    
    # Create test concept
    test_concept = SaCoConceptData(text_input=concept.text_input)
    
    # Add positive images to test
    for i in pos_test_indices:
        test_concept.positive_image_paths.append(concept.positive_image_paths[i])
        test_concept.positive_masks.append(concept.positive_masks[i])
        if concept.positive_bboxes:
            test_concept.positive_bboxes.append(concept.positive_bboxes[i])
        test_concept.positive_pair_ids.append(concept.positive_pair_ids[i])
    
    # Add negative images to test
    for i in neg_test_indices:
        test_concept.negative_image_paths.append(concept.negative_image_paths[i])
        test_concept.negative_pair_ids.append(concept.negative_pair_ids[i])
    
    return train_concept, test_concept


def evaluate_tokens_on_concept(
    model,
    tokens: List[int],
    concept,  # SaCoConceptData
    config,
    use_augmentation: bool = False,
    include_negatives: bool = True,
    return_detailed: bool = False,
) -> Tuple[float, dict, float]:
    """
    Evaluate token sequence on a concept's images.
    
    Now includes negative images and per-image tracking.
    
    Args:
        model: SAM3 model
        tokens: Token sequence to evaluate
        concept: SaCoConceptData with positive and optionally negative images
        config: Config object
        use_augmentation: Whether to apply augmentation
        include_negatives: Whether to evaluate on negative images
        return_detailed: Whether to return full ConceptEvalResult
        
    Returns:
        fitness, metrics_dict, eval_time
        If return_detailed=True, metrics_dict contains 'detailed_result' key
    """
    from detailed_eval import DetailedEvaluator
    
    start_time = time.time()
    
    evaluator = DetailedEvaluator(model, config)
    result = evaluator.evaluate_concept(
        tokens=tokens,
        concept=concept,
        include_negatives=include_negatives,
        use_augmentation=use_augmentation,
    )
    
    eval_time = time.time() - start_time
    
    # Build metrics dict
    metrics_dict = {
        'positive_mean_iou': result.positive_mean_iou,
        'positive_mean_quality_iou': result.positive_mean_quality_iou,
        'positive_recall': result.positive_recall,
        'positive_precision': result.positive_precision,
        'negative_false_positive_rate': result.negative_false_positive_rate,
        'negative_clean_rate': result.negative_clean_rate,
        'num_positive_images': result.num_positive_images,
        'num_negative_images': result.num_negative_images,
    }
    
    if return_detailed:
        metrics_dict['detailed_result'] = result
    
    return result.fitness, metrics_dict, eval_time


def run_search_on_concept(
    model,
    concept,  # SaCoConceptData (train split)
    config,
    vocab_embeddings,
    use_augmentation: bool = True,
    include_negatives: bool = True,
    token_count: int = None,  # Override config.optimization.soft_prompt_length
) -> Tuple[List[int], str, float, int, float]:
    """
    Run discrete search on train images.
    
    Args:
        model: SAM3 model
        concept: SaCoConceptData (train split)
        config: Config object
        vocab_embeddings: Vocabulary embeddings
        use_augmentation: Whether to use augmentation
        include_negatives: Whether to include negative images in search
        token_count: Number of tokens to search for (default: config.optimization.soft_prompt_length)
    
    Returns:
        best_tokens, decoded, train_fitness, num_evals, search_time
    """
    from detailed_eval import DetailedEvaluator
    from discrete_search import DiscreteSearchOptimizer
    
    start_time = time.time()
    
    # Create detailed evaluator
    evaluator = DetailedEvaluator(model, config)
    
    # Create evaluation function with counter
    eval_counter = {'count': 0}
    
    def counting_eval_fn(tokens):
        eval_counter['count'] += 1
        result = evaluator.evaluate_concept(
            tokens=tokens,
            concept=concept,
            include_negatives=include_negatives,
            use_augmentation=use_augmentation,
            # Use subset for speed during search
            max_positive_images=10,
            max_negative_images=5,
        )
        return result.fitness, result
    
    # Initialize random tokens
    tokenizer = model.backbone.language_backbone.tokenizer
    special_tokens = [49406, 49407]
    seq_length = token_count if token_count is not None else config.optimization.soft_prompt_length
    
    random.seed(42)  # Reproducible
    start_tokens = []
    while len(start_tokens) < seq_length:
        t = random.randint(0, vocab_embeddings.vocab_size - 1)
        if t not in special_tokens:
            start_tokens.append(t)
    
    # Run discrete search
    discrete_optimizer = DiscreteSearchOptimizer(
        vocab_embeddings=vocab_embeddings,
        evaluate_fn=counting_eval_fn,
        config=config,
        special_token_ids=special_tokens,
    )
    
    best = discrete_optimizer.optimize(seed_tokens=start_tokens, verbose=True)
    
    search_time = time.time() - start_time
    
    # Get actual eval count from cache (unique evaluations)
    num_unique_evals = len(discrete_optimizer.eval_cache)
    num_total_calls = eval_counter['count']
    
    print(f"  Search stats: {num_unique_evals} unique evals, {num_total_calls} total calls, {search_time:.1f}s")
    
    return best.tokens, tokenizer.decode(best.tokens), best.fitness, num_unique_evals, search_time


def evaluate_concept(
    model,
    concept,  # SaCoConceptData
    config,
    vocab_embeddings,
    train_ratio: float = 0.7,
    use_augmentation: bool = True,
    include_negatives: bool = True,
    save_detailed: bool = True,
    output_dir: str = 'eval_results',
    token_count: int = None,  # Override default token count
    verbose: bool = True,
) -> EvalResult:
    """
    Full evaluation for a single concept.
    
    Args:
        model: SAM3 model
        concept: SaCoConceptData
        config: Config object
        vocab_embeddings: Vocabulary embeddings for search
        train_ratio: Train/test split ratio
        use_augmentation: Whether to use augmentation
        include_negatives: Whether to include negative images
        save_detailed: Whether to save per-image results to JSON
        output_dir: Directory to save detailed results
        token_count: Number of tokens for search (default: config.optimization.soft_prompt_length)
        verbose: Print progress
    """
    tokenizer = model.backbone.language_backbone.tokenizer
    text_input = concept.text_input
    
    # Determine token count
    actual_token_count = token_count if token_count is not None else config.optimization.soft_prompt_length
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {text_input} (token_count={actual_token_count})")
        print(f"{'='*60}")
    
    # Split data (now includes negatives)
    train_concept, test_concept = split_concept_data(concept, train_ratio)
    
    if verbose:
        print(f"  Train: {train_concept.num_positive_images} positive, {train_concept.num_negative_images} negative")
        print(f"  Test:  {test_concept.num_positive_images} positive, {test_concept.num_negative_images} negative")
    
    # 1. Baseline: GT labels on test (with augmentation for fair comparison)
    if verbose:
        print(f"\n[1/3] Evaluating baseline on test...")
    
    baseline_tokens = tokenizer.encode(text_input)
    baseline_fitness, baseline_metrics, baseline_time = evaluate_tokens_on_concept(
        model, baseline_tokens, test_concept, config, 
        use_augmentation=use_augmentation,
        include_negatives=include_negatives,
        return_detailed=save_detailed,
    )
    
    if verbose:
        print(f"  Baseline fitness: {baseline_fitness:.4f}")
        print(f"  Positive IoU: {baseline_metrics.get('positive_mean_iou', 0):.4f}")
        if include_negatives:
            print(f"  Negative FP rate: {baseline_metrics.get('negative_false_positive_rate', 0):.4f}")
        print(f"  Eval time: {baseline_time:.1f}s")
    
    # Save baseline detailed results
    if save_detailed and 'detailed_result' in baseline_metrics:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        baseline_detail_path = output_path / f"{text_input.replace(' ', '_')}_baseline_detailed.json"
        baseline_metrics['detailed_result'].save(str(baseline_detail_path))
        if verbose:
            print(f"  Saved detailed results: {baseline_detail_path}")
    
    # 2. Search on train
    if verbose:
        print(f"\n[2/3] Running search on train (token_count={actual_token_count})...")
    
    best_tokens, decoded, train_fitness, num_evals, search_time = run_search_on_concept(
        model, train_concept, config, vocab_embeddings, use_augmentation,
        include_negatives=include_negatives,
        token_count=actual_token_count,
    )
    
    if verbose:
        print(f"  Train fitness: {train_fitness:.4f}")
        print(f"  Best prompt: '{decoded}'")
        print(f"  Search time: {search_time:.1f}s ({num_evals} evals)")
    
    # 3. Evaluate searched prompt on test (with augmentation for fair comparison)
    if verbose:
        print(f"\n[3/3] Evaluating searched prompt on test...")
    
    search_fitness, search_metrics, eval_time = evaluate_tokens_on_concept(
        model, best_tokens, test_concept, config, 
        use_augmentation=use_augmentation,
        include_negatives=include_negatives,
        return_detailed=save_detailed,
    )
    
    if verbose:
        print(f"  Test fitness: {search_fitness:.4f}")
        print(f"  Positive IoU: {search_metrics.get('positive_mean_iou', 0):.4f}")
        if include_negatives:
            print(f"  Negative FP rate: {search_metrics.get('negative_false_positive_rate', 0):.4f}")
        print(f"  Eval time: {eval_time:.1f}s")
    
    # Save search detailed results
    if save_detailed and 'detailed_result' in search_metrics:
        search_detail_path = output_path / f"{text_input.replace(' ', '_')}_tc{actual_token_count}_search_detailed.json"
        search_metrics['detailed_result'].save(str(search_detail_path))
        if verbose:
            print(f"  Saved detailed results: {search_detail_path}")
    
    # Compare
    improvement = search_fitness - baseline_fitness
    relative_imp = improvement / baseline_fitness if baseline_fitness > 0 else 0
    search_wins = search_fitness >= baseline_fitness
    
    # Compute summary
    total_time = search_time + eval_time
    time_per_eval = (search_time * 1000) / num_evals if num_evals > 0 else 0
    
    if verbose:
        winner = "✓ SEARCH" if search_wins else "✗ BASELINE"
        print(f"\n  RESULT: {winner}")
        print(f"    Baseline: {baseline_fitness:.4f}")
        print(f"    Search:   {search_fitness:.4f}")
        print(f"    Δ = {improvement:+.4f} ({relative_imp*100:+.1f}%)")
        print(f"\n  COMPUTE:")
        print(f"    Token count: {actual_token_count}")
        print(f"    Search evals: {num_evals}")
        print(f"    Search time: {search_time:.1f}s")
        print(f"    Time/eval: {time_per_eval:.1f}ms")
    
    return EvalResult(
        concept=text_input,
        num_train_images=train_concept.num_positive_images,
        num_test_images=test_concept.num_positive_images,
        num_train_negatives=train_concept.num_negative_images,
        num_test_negatives=test_concept.num_negative_images,
        token_count=actual_token_count,
        baseline_test_fitness=baseline_fitness,
        baseline_test_iou=baseline_metrics.get('positive_mean_iou', 0),
        baseline_eval_time=baseline_time,
        baseline_eval_count=1,  # Single forward pass for baseline
        search_train_fitness=train_fitness,
        search_test_fitness=search_fitness,
        search_test_iou=search_metrics.get('positive_mean_iou', 0),
        optimized_tokens=best_tokens,
        optimized_decoded=decoded,
        search_time=search_time,
        search_evals=num_evals,
        eval_time=eval_time,
        improvement=improvement,
        relative_improvement=relative_imp,
        search_wins=search_wins,
        total_search_forward_passes=num_evals,
        total_time_seconds=total_time,
        time_per_eval_ms=time_per_eval,
    )


def run_full_evaluation(
    model,
    dataset,  # SaCoDataset
    concept_names: List[str],
    config,
    vocab_embeddings,
    train_ratio: float = 0.7,
    use_augmentation: bool = True,
    include_negatives: bool = True,
    token_counts: List[int] = None,  # Token count sweep: e.g., [3, 5, 7]
    output_dir: str = 'eval_results',
    verbose: bool = True,
) -> EvalSummary:
    """
    Run full evaluation across multiple concepts with optional token count sweep.
    
    Args:
        model: SAM3 model
        dataset: SaCoDataset
        concept_names: List of concept names to evaluate
        config: Config object
        vocab_embeddings: Vocabulary embeddings
        train_ratio: Train/test split ratio
        use_augmentation: Whether to use augmentation
        include_negatives: Whether to include negative images
        token_counts: List of token counts to sweep (default: [config default])
        output_dir: Output directory
        verbose: Print progress
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Default token counts
    if token_counts is None:
        token_counts = [config.optimization.soft_prompt_length]
    
    all_results = []
    
    # Run evaluation for each concept x token_count combination
    total_runs = len(concept_names) * len(token_counts)
    run_idx = 0
    
    for name in concept_names:
        for tc in token_counts:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] {name} (token_count={tc})")
            
            try:
                concept = dataset[name]
                result = evaluate_concept(
                    model=model,
                    concept=concept,
                    config=config,
                    vocab_embeddings=vocab_embeddings,
                    train_ratio=train_ratio,
                    use_augmentation=use_augmentation,
                    include_negatives=include_negatives,
                    save_detailed=True,
                    output_dir=output_dir,
                    token_count=tc,
                    verbose=verbose,
                )
                all_results.append(result)
                
                # Save intermediate
                safe_name = name.replace(' ', '_').replace('/', '_')
                with open(output_path / f"result_{run_idx:03d}_{safe_name}_tc{tc}.json", 'w') as f:
                    json.dump(asdict(result), f, indent=2, default=str)
                    
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Get hardware info
    hw_info = get_hardware_info(config.device)
    
    # Compute summary (aggregating all token counts)
    if all_results:
        baseline_fitnesses = [r.baseline_test_fitness for r in all_results]
        search_fitnesses = [r.search_test_fitness for r in all_results]
        improvements = [r.improvement for r in all_results]
        wins = [r.search_wins for r in all_results]
        search_times = [r.search_time for r in all_results]
        search_evals = [r.search_evals for r in all_results]
        time_per_evals = [r.time_per_eval_ms for r in all_results]
        
        total_wall_time = np.sum(search_times)
        total_evals = int(np.sum(search_evals))
        avg_time_per_eval = np.mean(time_per_evals) if time_per_evals else 0
        
        # Compute FLOPs
        flops_per_pass = estimate_sam3_flops(config.data.image_size)
        total_flops = flops_per_pass * total_evals
        effective_tflops = (total_flops / total_wall_time / 1e12) if total_wall_time > 0 else 0
        
        summary = EvalSummary(
            timestamp=datetime.now().isoformat(),
            hardware=asdict(hw_info),
            num_concepts=len(concept_names),
            train_ratio=train_ratio,
            image_size=config.data.image_size,
            baseline_mean_fitness=np.mean(baseline_fitnesses),
            search_mean_fitness=np.mean(search_fitnesses),
            mean_improvement=np.mean(improvements),
            win_rate=np.mean(wins),
            total_wall_time_seconds=total_wall_time,
            total_forward_passes=total_evals,
            total_flops=total_flops,
            avg_forward_passes_per_concept=np.mean(search_evals),
            avg_wall_time_per_concept=np.mean(search_times),
            avg_time_per_forward_pass_ms=avg_time_per_eval,
            effective_tflops_per_second=effective_tflops,
            results=[asdict(r) for r in all_results],
        )
    else:
        summary = EvalSummary(
            timestamp=datetime.now().isoformat(),
            hardware=asdict(hw_info),
            num_concepts=0,
            train_ratio=train_ratio,
            image_size=config.data.image_size,
            baseline_mean_fitness=0,
            search_mean_fitness=0,
            mean_improvement=0,
            win_rate=0,
            total_wall_time_seconds=0,
            total_forward_passes=0,
            total_flops=0,
            avg_forward_passes_per_concept=0,
            avg_wall_time_per_concept=0,
            avg_time_per_forward_pass_ms=0,
            effective_tflops_per_second=0,
            results=[],
        )
    
    # Save summary
    with open(output_path / 'eval_summary.json', 'w') as f:
        json.dump(asdict(summary), f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print(f"\n--- HARDWARE ---")
    hw = summary.hardware
    print(f"Device: {hw['device_type']} ({hw['device_name']})")
    if hw['gpu_memory_gb']:
        print(f"GPU Memory: {hw['gpu_memory_gb']:.1f} GB")
    if hw['compute_capability']:
        print(f"CUDA Compute: {hw['compute_capability']}")
    print(f"CPU: {hw['cpu_name']} ({hw['cpu_cores']} cores)")
    print(f"RAM: {hw['ram_gb']:.1f} GB")
    
    print(f"\n--- EXPERIMENT ---")
    print(f"Concepts evaluated: {summary.num_concepts}")
    print(f"Token counts: {token_counts}")
    print(f"Total runs: {len(all_results)}")
    print(f"Train/test ratio: {summary.train_ratio:.0%}/{1-summary.train_ratio:.0%}")
    print(f"Image size: {summary.image_size}x{summary.image_size}")
    
    print(f"\n--- PERFORMANCE (all runs) ---")
    print(f"Baseline mean fitness: {summary.baseline_mean_fitness:.4f}")
    print(f"Search mean fitness:   {summary.search_mean_fitness:.4f}")
    print(f"Mean improvement:      {summary.mean_improvement:+.4f}")
    print(f"Win rate:              {summary.win_rate:.1%}")
    
    print(f"\n--- COMPUTE (measured) ---")
    print(f"Total wall time:       {summary.total_wall_time_seconds:.1f}s ({summary.total_wall_time_seconds/60:.1f} min)")
    print(f"Total forward passes:  {summary.total_forward_passes:,}")
    print(f"Total FLOPs:           {summary.total_flops:.2e} ({summary.total_flops/1e12:.1f} TFLOPs)")
    print(f"Avg passes/run:        {summary.avg_forward_passes_per_concept:.0f}")
    print(f"Avg time/run:          {summary.avg_wall_time_per_concept:.1f}s")
    print(f"Avg time/pass:         {summary.avg_time_per_forward_pass_ms:.1f}ms")
    print(f"Effective throughput:  {summary.effective_tflops_per_second:.2f} TFLOP/s")
    print("="*70)
    
    # Run post-hoc analysis
    if all_results:
        print("\nRunning post-hoc analysis...")
        analysis = run_posthoc_analysis(all_results, output_path)
        print("Post-hoc analysis complete.")
    
    return summary


# =============================================================================
# POST-HOC ANALYSIS
# =============================================================================

def run_posthoc_analysis(
    results: List[EvalResult],
    output_path: Path,
    multi_instance_analysis_path: str = None,
) -> dict:
    """
    Run post-hoc analysis on evaluation results.
    
    Slices results by:
    1. Token count (if sweep was done)
    2. Multi-instance vs single-instance concepts
    3. Concept difficulty buckets
    4. Performance on high-instance images
    
    Args:
        results: List of EvalResult from evaluation
        output_path: Directory to save analysis
        multi_instance_analysis_path: Path to multi_instance_analysis.json for subset analysis
        
    Returns:
        Analysis dict
    """
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'num_results': len(results),
    }
    
    # 1. Analysis by token count
    token_count_analysis = analyze_by_token_count(results)
    analysis['by_token_count'] = token_count_analysis
    
    # 2. Analysis by concept (best token count per concept)
    concept_analysis = analyze_by_concept(results)
    analysis['by_concept'] = concept_analysis
    
    # 3. Load multi-instance data if available and analyze
    if multi_instance_analysis_path and Path(multi_instance_analysis_path).exists():
        with open(multi_instance_analysis_path, 'r') as f:
            mi_data = json.load(f)
        
        multi_instance_analysis = analyze_multi_instance_performance(results, mi_data)
        analysis['multi_instance'] = multi_instance_analysis
    
    # 4. Summary statistics
    analysis['summary'] = compute_summary_statistics(results)
    
    # Save analysis
    with open(output_path / 'posthoc_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Print key findings
    print_posthoc_summary(analysis)
    
    return analysis


def analyze_by_token_count(results: List[EvalResult]) -> dict:
    """Analyze performance grouped by token count."""
    from collections import defaultdict
    
    by_tc = defaultdict(list)
    for r in results:
        by_tc[r.token_count].append(r)
    
    tc_analysis = {}
    for tc, tc_results in sorted(by_tc.items()):
        fitnesses = [r.search_test_fitness for r in tc_results]
        baselines = [r.baseline_test_fitness for r in tc_results]
        improvements = [r.improvement for r in tc_results]
        wins = [r.search_wins for r in tc_results]
        search_times = [r.search_time for r in tc_results]
        search_evals = [r.search_evals for r in tc_results]
        
        tc_analysis[tc] = {
            'num_results': len(tc_results),
            'baseline_mean': float(np.mean(baselines)),
            'search_mean': float(np.mean(fitnesses)),
            'search_std': float(np.std(fitnesses)),
            'improvement_mean': float(np.mean(improvements)),
            'improvement_std': float(np.std(improvements)),
            'win_rate': float(np.mean(wins)),
            'avg_search_time': float(np.mean(search_times)),
            'avg_search_evals': float(np.mean(search_evals)),
            'best_concept': max(tc_results, key=lambda x: x.improvement).concept,
            'worst_concept': min(tc_results, key=lambda x: x.improvement).concept,
        }
    
    # Find optimal token count
    if tc_analysis:
        optimal_tc = max(tc_analysis.keys(), key=lambda tc: tc_analysis[tc]['improvement_mean'])
        tc_analysis['optimal_token_count'] = optimal_tc
        tc_analysis['optimal_improvement'] = tc_analysis[optimal_tc]['improvement_mean']
    
    return tc_analysis


def analyze_by_concept(results: List[EvalResult]) -> dict:
    """Analyze performance grouped by concept, finding best token count per concept."""
    from collections import defaultdict
    
    by_concept = defaultdict(list)
    for r in results:
        by_concept[r.concept].append(r)
    
    concept_analysis = {}
    for concept, concept_results in by_concept.items():
        # Find best result for this concept
        best = max(concept_results, key=lambda x: x.search_test_fitness)
        
        concept_analysis[concept] = {
            'num_runs': len(concept_results),
            'baseline_fitness': best.baseline_test_fitness,
            'best_search_fitness': best.search_test_fitness,
            'best_token_count': best.token_count,
            'best_improvement': best.improvement,
            'best_prompt': best.optimized_decoded,
            'search_wins': best.search_wins,
            # All token count results
            'all_results': [
                {
                    'token_count': r.token_count,
                    'search_fitness': r.search_test_fitness,
                    'improvement': r.improvement,
                }
                for r in sorted(concept_results, key=lambda x: x.token_count)
            ],
        }
    
    return concept_analysis


def analyze_multi_instance_performance(
    results: List[EvalResult],
    mi_data: dict,
) -> dict:
    """
    Analyze performance on multi-instance vs single-instance concepts.
    
    Args:
        results: Evaluation results
        mi_data: Multi-instance analysis data from saco_multi_instance_analysis.json
    """
    # Build lookup of multi-instance concepts
    mi_concepts = set()
    single_concepts = set()
    
    for stat in mi_data.get('concept_stats', []):
        if stat['num_multi_instance_images'] >= 5:
            mi_concepts.add(stat['concept_name'])
        elif stat['num_multi_instance_images'] == 0:
            single_concepts.add(stat['concept_name'])
    
    # Also get the recommended test concepts
    recommended = set(r['concept'] for r in mi_data.get('recommended_test_concepts', []))
    
    # Separate results
    mi_results = [r for r in results if r.concept in mi_concepts]
    single_results = [r for r in results if r.concept in single_concepts]
    recommended_results = [r for r in results if r.concept in recommended]
    
    def compute_stats(result_list, name):
        if not result_list:
            return {'num_results': 0, 'note': f'No {name} concepts in results'}
        
        return {
            'num_results': len(result_list),
            'concepts': list(set(r.concept for r in result_list)),
            'baseline_mean': float(np.mean([r.baseline_test_fitness for r in result_list])),
            'search_mean': float(np.mean([r.search_test_fitness for r in result_list])),
            'improvement_mean': float(np.mean([r.improvement for r in result_list])),
            'win_rate': float(np.mean([r.search_wins for r in result_list])),
        }
    
    return {
        'multi_instance': compute_stats(mi_results, 'multi-instance'),
        'single_instance': compute_stats(single_results, 'single-instance'),
        'recommended_test': compute_stats(recommended_results, 'recommended'),
        'mi_concept_count': len(mi_concepts),
        'single_concept_count': len(single_concepts),
        'recommended_count': len(recommended),
    }


def compute_summary_statistics(results: List[EvalResult]) -> dict:
    """Compute overall summary statistics."""
    if not results:
        return {}
    
    improvements = [r.improvement for r in results]
    baselines = [r.baseline_test_fitness for r in results]
    searches = [r.search_test_fitness for r in results]
    wins = [r.search_wins for r in results]
    
    return {
        'total_runs': len(results),
        'unique_concepts': len(set(r.concept for r in results)),
        'unique_token_counts': sorted(set(r.token_count for r in results)),
        
        'baseline': {
            'mean': float(np.mean(baselines)),
            'std': float(np.std(baselines)),
            'min': float(np.min(baselines)),
            'max': float(np.max(baselines)),
        },
        'search': {
            'mean': float(np.mean(searches)),
            'std': float(np.std(searches)),
            'min': float(np.min(searches)),
            'max': float(np.max(searches)),
        },
        'improvement': {
            'mean': float(np.mean(improvements)),
            'std': float(np.std(improvements)),
            'min': float(np.min(improvements)),
            'max': float(np.max(improvements)),
            'positive_rate': float(np.mean([i > 0 for i in improvements])),
        },
        'win_rate': float(np.mean(wins)),
        
        # Top performers
        'top_improvements': [
            {'concept': r.concept, 'token_count': r.token_count, 'improvement': r.improvement}
            for r in sorted(results, key=lambda x: x.improvement, reverse=True)[:10]
        ],
        'worst_improvements': [
            {'concept': r.concept, 'token_count': r.token_count, 'improvement': r.improvement}
            for r in sorted(results, key=lambda x: x.improvement)[:10]
        ],
    }


def print_posthoc_summary(analysis: dict):
    """Print key findings from post-hoc analysis."""
    print("\n" + "="*70)
    print("POST-HOC ANALYSIS")
    print("="*70)
    
    # Token count analysis
    if 'by_token_count' in analysis:
        tc_analysis = analysis['by_token_count']
        print("\n--- BY TOKEN COUNT ---")
        for tc in sorted(k for k in tc_analysis.keys() if isinstance(k, int)):
            stats = tc_analysis[tc]
            print(f"  tc={tc}: search={stats['search_mean']:.4f} (±{stats['search_std']:.4f}), "
                  f"Δ={stats['improvement_mean']:+.4f}, win_rate={stats['win_rate']:.1%}")
        
        if 'optimal_token_count' in tc_analysis:
            print(f"\n  OPTIMAL: token_count={tc_analysis['optimal_token_count']} "
                  f"(Δ={tc_analysis['optimal_improvement']:+.4f})")
    
    # Multi-instance analysis
    if 'multi_instance' in analysis:
        mi = analysis['multi_instance']
        print("\n--- MULTI-INSTANCE ANALYSIS ---")
        
        if mi['multi_instance']['num_results'] > 0:
            mi_stats = mi['multi_instance']
            print(f"  Multi-instance ({mi_stats['num_results']} runs): "
                  f"Δ={mi_stats['improvement_mean']:+.4f}, win_rate={mi_stats['win_rate']:.1%}")
        
        if mi['single_instance']['num_results'] > 0:
            single_stats = mi['single_instance']
            print(f"  Single-instance ({single_stats['num_results']} runs): "
                  f"Δ={single_stats['improvement_mean']:+.4f}, win_rate={single_stats['win_rate']:.1%}")
        
        if mi['recommended_test']['num_results'] > 0:
            rec_stats = mi['recommended_test']
            print(f"  Recommended test ({rec_stats['num_results']} runs): "
                  f"Δ={rec_stats['improvement_mean']:+.4f}, win_rate={rec_stats['win_rate']:.1%}")
    
    # Summary
    if 'summary' in analysis:
        summary = analysis['summary']
        print("\n--- OVERALL ---")
        print(f"  Total runs: {summary['total_runs']}")
        print(f"  Unique concepts: {summary['unique_concepts']}")
        print(f"  Win rate: {summary['win_rate']:.1%}")
        print(f"  Mean improvement: {summary['improvement']['mean']:+.4f} (±{summary['improvement']['std']:.4f})")
        print(f"  Positive improvement rate: {summary['improvement']['positive_rate']:.1%}")
        
        print("\n  Top 5 improvements:")
        for item in summary['top_improvements'][:5]:
            print(f"    {item['concept']} (tc={item['token_count']}): Δ={item['improvement']:+.4f}")
    
    print("="*70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline vs adversarial prompt search')
    parser.add_argument('--dataset', type=str, required=True, help='Path to SA-Co dataset')
    parser.add_argument('--images_dir', type=str, default=None, help='Path to images (if separate)')
    parser.add_argument('--annotation_file', type=str, default=None, help='Specific annotation file name')
    parser.add_argument('--concepts', type=str, default=None, 
                        help='Comma-separated concept names (default: all with >=5 images)')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Output directory')
    parser.add_argument('--device', type=str, default='mps', help='Device (mps/cuda/cpu)')
    parser.add_argument('--no_augmentation', action='store_true', help='Disable augmentation')
    parser.add_argument('--no_negatives', action='store_true', help='Disable negative images')
    parser.add_argument('--skip_prestart', action='store_true', help='Skip prestart checks (not recommended)')
    
    # Token count sweep
    parser.add_argument('--token_counts', type=str, default='5',
                        help='Comma-separated token counts to sweep (e.g., "3,5,7")')
    
    # Multi-instance analysis
    parser.add_argument('--multi_instance_analysis', type=str, default=None,
                        help='Path to saco_multi_instance_analysis.json for subset analysis')
    parser.add_argument('--multi_instance_only', action='store_true',
                        help='Only evaluate recommended multi-instance concepts')
    
    args = parser.parse_args()
    
    # Parse token counts
    token_counts = [int(tc.strip()) for tc in args.token_counts.split(',')]
    print(f"Token counts to evaluate: {token_counts}")
    
    # ==========================================================================
    # PRESTART CHECKS
    # ==========================================================================
    if not args.skip_prestart:
        checks_passed, prestart_report = run_prestart_checks(
            device=args.device,
            dataset_path=args.dataset,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            verbose=True,
        )
        
        # Save prestart report
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / 'prestart_report.json', 'w') as f:
            json.dump(prestart_report, f, indent=2, default=str)
        
        if not checks_passed:
            print("\nAborting due to failed prestart checks.")
            print(f"Report saved to: {output_path / 'prestart_report.json'}")
            return 1
        
        print("Prestart checks passed. Starting evaluation...\n")
    else:
        print("WARNING: Skipping prestart checks (--skip_prestart)")
    
    # ==========================================================================
    # LOAD RESOURCES
    # ==========================================================================
    
    # Load model
    print("Loading SAM3 model...")
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model()
    model.to(args.device)
    model.eval()
    
    # Load dataset
    print(f"Loading SA-Co dataset from: {args.dataset}")
    from saco_loader import load_saco_dataset
    dataset = load_saco_dataset(args.dataset, annotation_file=args.annotation_file, images_dir=args.images_dir)
    
    # Load config
    from config import get_config
    config = get_config(dataset_path=args.dataset, device=args.device, output_dir=args.output_dir)
    
    # Load vocab embeddings
    print("Loading vocabulary embeddings...")
    from embeddings import load_both_vocab_embeddings
    vocab_256, vocab_1024 = load_both_vocab_embeddings(model=model, device=args.device)
    
    # Select concepts
    if args.multi_instance_only and args.multi_instance_analysis:
        # Use recommended multi-instance concepts
        with open(args.multi_instance_analysis, 'r') as f:
            mi_data = json.load(f)
        concept_names = [r['concept'] for r in mi_data.get('recommended_test_concepts', [])]
        print(f"Using {len(concept_names)} recommended multi-instance concepts")
    elif args.concepts:
        concept_names = [c.strip() for c in args.concepts.split(',')]
    else:
        concept_info = dataset.list_concepts(min_positive_images=5, sort_by='instances')
        concept_names = [c['text_input'] for c in concept_info[:20]]  # Top 20
    
    print(f"Evaluating {len(concept_names)} concepts × {len(token_counts)} token counts = {len(concept_names) * len(token_counts)} runs")
    
    # ==========================================================================
    # RUN EVALUATION
    # ==========================================================================
    
    summary = run_full_evaluation(
        model=model,
        dataset=dataset,
        concept_names=concept_names,
        config=config,
        vocab_embeddings=vocab_1024,
        train_ratio=args.train_ratio,
        use_augmentation=not args.no_augmentation,
        include_negatives=not args.no_negatives,
        token_counts=token_counts,
        output_dir=args.output_dir,
    )
    
    # Run additional post-hoc analysis with multi-instance data
    if args.multi_instance_analysis:
        output_path = Path(args.output_dir)
        results = [EvalResult(**r) for r in summary.results]
        run_posthoc_analysis(results, output_path, args.multi_instance_analysis)
    
    print(f"\nResults saved to: {args.output_dir}/")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
