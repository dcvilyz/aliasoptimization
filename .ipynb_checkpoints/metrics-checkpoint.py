"""
Evaluation metrics for alias optimization.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import time


@dataclass
class MetricResult:
    """Container for metric results."""
    mean_iou: float
    instance_recall: float
    instance_precision: float
    f1_score: float
    
    # Per-threshold metrics
    recall_at_50: float
    recall_at_75: float
    
    # Additional
    num_predictions: int
    num_ground_truth: int
    num_matched: int
    
    # Quality-adjusted metrics
    mean_quality_iou: float = 0.0
    num_filtered: int = 0  # Masks filtered due to size
    
    # Coverage tracking (for false negative penalty)
    num_images: int = 0
    num_images_with_predictions: int = 0
    
    @property
    def coverage(self) -> float:
        """Fraction of images that got at least one prediction."""
        if self.num_images == 0:
            return 0.0
        return self.num_images_with_predictions / self.num_images
    
    def __repr__(self):
        return (
            f"MetricResult(\n"
            f"  mean_iou={self.mean_iou:.4f},\n"
            f"  mean_quality_iou={self.mean_quality_iou:.4f},\n"
            f"  recall={self.instance_recall:.4f},\n"
            f"  precision={self.instance_precision:.4f},\n"
            f"  f1={self.f1_score:.4f},\n"
            f"  recall@50={self.recall_at_50:.4f},\n"
            f"  recall@75={self.recall_at_75:.4f},\n"
            f"  matched={self.num_matched}/{self.num_ground_truth},\n"
            f"  filtered={self.num_filtered},\n"
            f"  coverage={self.coverage:.2%}\n"
            f")"
        )
    
    @property
    def matched_str(self) -> str:
        return f"{self.num_matched}/{self.num_ground_truth}"
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'mean_iou': self.mean_iou,
            'mean_quality_iou': self.mean_quality_iou,
            'instance_recall': self.instance_recall,
            'instance_precision': self.instance_precision,
            'f1_score': self.f1_score,
            'recall_at_50': self.recall_at_50,
            'recall_at_75': self.recall_at_75,
            'num_predictions': self.num_predictions,
            'num_ground_truth': self.num_ground_truth,
            'num_matched': self.num_matched,
            'num_filtered': self.num_filtered,
            'num_images': self.num_images,
            'num_images_with_predictions': self.num_images_with_predictions,
            'coverage': self.coverage,
        }


@dataclass
class SearchStep:
    """Single step in the search trajectory."""
    step_id: int
    tokens: List[int]
    fitness: float
    threshold: float
    parent_id: Optional[int]
    method: str  # 'init', 'local_search', 'mutation', 'beam', 'evolution'
    timestamp: float
    metrics: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'step_id': self.step_id,
            'tokens': self.tokens,
            'fitness': self.fitness,
            'threshold': self.threshold,
            'parent_id': self.parent_id,
            'method': self.method,
            'timestamp': self.timestamp,
            'metrics': self.metrics,
        }


@dataclass 
class SearchTrajectory:
    """
    Tracks the complete search trajectory for visualization.
    
    Logs every evaluation during optimization so we can later
    visualize the path through token embedding space.
    """
    category_name: str = ""
    config: Dict = field(default_factory=dict)
    steps: List[SearchStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    def log(
        self,
        tokens: List[int],
        fitness: float,
        threshold: float,
        parent_id: Optional[int] = None,
        method: str = 'local_search',
        metrics: Optional[MetricResult] = None,
    ) -> int:
        """Log a search step and return its ID."""
        step_id = len(self.steps)
        step = SearchStep(
            step_id=step_id,
            tokens=list(tokens),  # Copy to avoid mutation issues
            fitness=fitness,
            threshold=threshold,
            parent_id=parent_id,
            method=method,
            timestamp=time.time() - self.start_time,
            metrics=metrics.to_dict() if metrics else None,
        )
        self.steps.append(step)
        return step_id
    
    def get_best_at_threshold(self, threshold: float) -> Optional[SearchStep]:
        """Get best step at a specific threshold."""
        relevant = [s for s in self.steps if s.threshold == threshold]
        if not relevant:
            return None
        return max(relevant, key=lambda s: s.fitness)
    
    def get_trajectory_to(self, step_id: int) -> List[SearchStep]:
        """Get the path from root to a specific step."""
        path = []
        current = self.steps[step_id]
        while current is not None:
            path.append(current)
            if current.parent_id is not None:
                current = self.steps[current.parent_id]
            else:
                current = None
        return list(reversed(path))
    
    def save(self, path: str):
        """Save trajectory to JSON for later visualization."""
        data = {
            'category_name': self.category_name,
            'config': self.config,
            'total_steps': len(self.steps),
            'duration_seconds': time.time() - self.start_time,
            'steps': [s.to_dict() for s in self.steps],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SearchTrajectory':
        """Load trajectory from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        trajectory = cls(
            category_name=data['category_name'],
            config=data['config'],
        )
        for step_data in data['steps']:
            step = SearchStep(
                step_id=step_data['step_id'],
                tokens=step_data['tokens'],
                fitness=step_data['fitness'],
                threshold=step_data['threshold'],
                parent_id=step_data['parent_id'],
                method=step_data['method'],
                timestamp=step_data['timestamp'],
                metrics=step_data.get('metrics'),
            )
            trajectory.steps.append(step)
        return trajectory
    
    def summary(self) -> str:
        """Print summary of the trajectory."""
        if not self.steps:
            return "Empty trajectory"
        
        thresholds = sorted(set(s.threshold for s in self.steps))
        lines = [
            f"Search Trajectory: {self.category_name}",
            f"Total steps: {len(self.steps)}",
            f"Duration: {self.steps[-1].timestamp:.1f}s",
            f"Thresholds: {thresholds}",
            "",
            "Best per threshold:",
        ]
        for t in thresholds:
            best = self.get_best_at_threshold(t)
            if best:
                lines.append(f"  {t}: fitness={best.fitness:.4f}, tokens={best.tokens}")
        
        return "\n".join(lines)


def filter_oversized_masks(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    max_size_ratio: float = 2.0,
) -> Tuple[torch.Tensor, int]:
    """
    Filter out predicted masks that are too large relative to GT.
    
    This removes "segment everything" masks that would have high FP rate.
    
    Args:
        pred_masks: [N, H, W] predicted masks
        gt_masks: [M, H, W] ground truth masks
        max_size_ratio: Maximum allowed ratio of pred size to avg GT size
        
    Returns:
        Filtered pred_masks and count of how many were filtered
    """
    if pred_masks.shape[0] == 0 or gt_masks.shape[0] == 0:
        return pred_masks, 0
    
    # Compute sizes
    pred_sizes = pred_masks.sum(dim=(-1, -2)).float()  # [N]
    gt_sizes = gt_masks.sum(dim=(-1, -2)).float()  # [M]
    avg_gt_size = gt_sizes.mean()
    
    # Filter predictions that are too large
    max_allowed_size = max_size_ratio * avg_gt_size
    valid_mask = pred_sizes <= max_allowed_size
    
    num_filtered = (~valid_mask).sum().item()
    filtered_preds = pred_masks[valid_mask]
    
    return filtered_preds, num_filtered


def compute_mask_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute IoU between predicted and ground truth masks.
    
    Args:
        pred_mask: [H, W] or [N, H, W] predicted mask(s)
        gt_mask: [H, W] or [M, H, W] ground truth mask(s)
        eps: Small value to avoid division by zero
        
    Returns:
        IoU score(s). Shape depends on inputs:
        - [H,W] vs [H,W] -> scalar
        - [N,H,W] vs [M,H,W] -> [N, M] pairwise IoU matrix
    """
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    
    if pred_mask.dim() == 2 and gt_mask.dim() == 2:
        # Single mask vs single mask
        intersection = (pred_mask & gt_mask).sum().float()
        union = (pred_mask | gt_mask).sum().float()
        return intersection / (union + eps)
    
    elif pred_mask.dim() == 3 and gt_mask.dim() == 3:
        # Batch of masks: compute pairwise IoU
        N, H, W = pred_mask.shape
        M = gt_mask.shape[0]
        
        pred_flat = pred_mask.view(N, 1, H * W).float()  # [N, 1, H*W]
        gt_flat = gt_mask.view(1, M, H * W).float()  # [1, M, H*W]
        
        intersection = (pred_flat * gt_flat).sum(dim=2)  # [N, M]
        union = pred_flat.sum(dim=2) + gt_flat.sum(dim=2) - intersection  # [N, M]
        
        return intersection / (union + eps)
    
    else:
        raise ValueError(f"Unexpected mask dimensions: pred={pred_mask.shape}, gt={gt_mask.shape}")


def compute_false_positive_rate(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    r"""
    Compute false positive rate: |C \ G| / |C|
    
    This measures what fraction of the prediction is NOT in the ground truth.
    High values indicate over-segmentation (predicting too much).
    
    Args:
        pred_mask: [H, W] or [N, H, W] predicted mask(s)
        gt_mask: [H, W] or [M, H, W] ground truth mask(s)
        
    Returns:
        FP rate(s). Same shape logic as compute_mask_iou.
    """
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    
    if pred_mask.dim() == 2 and gt_mask.dim() == 2:
        pred_area = pred_mask.sum().float()
        if pred_area < eps:
            return torch.tensor(0.0)
        false_positive = (pred_mask & ~gt_mask).sum().float()
        return false_positive / (pred_area + eps)
    
    elif pred_mask.dim() == 3 and gt_mask.dim() == 3:
        N, H, W = pred_mask.shape
        M = gt_mask.shape[0]
        
        pred_flat = pred_mask.view(N, 1, H * W).float()  # [N, 1, H*W]
        gt_flat = gt_mask.view(1, M, H * W).float()  # [1, M, H*W]
        
        # False positives: pred AND NOT gt
        false_positive = (pred_flat * (1 - gt_flat)).sum(dim=2)  # [N, M]
        pred_area = pred_flat.sum(dim=2)  # [N, 1]
        
        return false_positive / (pred_area + eps)
    
    else:
        raise ValueError(f"Unexpected mask dimensions: pred={pred_mask.shape}, gt={gt_mask.shape}")


def compute_quality_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    fp_penalty_weight: float = 1.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute quality-adjusted IoU that penalizes over-segmentation.
    
    Quality = IoU * (1 - FP_rate)
    
    This ensures that masks which cover too much area (beyond the GT)
    get penalized even if they technically have decent IoU.
    
    A "segment everything" mask will have:
    - IoU ~ 0.1 (small intersection, huge union)  
    - FP_rate ~ 0.95 (95% of prediction is junk)
    - Quality ~ 0.1 * 0.05 = 0.005
    
    A tight, accurate mask will have:
    - IoU ~ 0.8
    - FP_rate ~ 0.1
    - Quality ~ 0.8 * 0.9 = 0.72
    
    Args:
        pred_mask: Predicted mask(s)
        gt_mask: Ground truth mask(s)
        fp_penalty_weight: How harshly to penalize FP (1.0 = full penalty)
        eps: Small value for numerical stability
        
    Returns:
        Quality-adjusted IoU score(s)
    """
    iou = compute_mask_iou(pred_mask, gt_mask, eps)
    fp_rate = compute_false_positive_rate(pred_mask, gt_mask, eps)
    
    # Quality = IoU * (1 - fp_penalty_weight * FP_rate)
    quality = iou * (1 - fp_penalty_weight * fp_rate)
    
    return quality


def compute_mask_iou_batch(
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    eps: float = 1e-6
) -> List[torch.Tensor]:
    """
    Compute IoU matrices for batch of images.
    
    Args:
        pred_masks: List of [N_i, H, W] predicted masks per image
        gt_masks: List of [M_i, H, W] ground truth masks per image
        
    Returns:
        List of [N_i, M_i] IoU matrices per image
    """
    iou_matrices = []
    for pred, gt in zip(pred_masks, gt_masks):
        if pred.shape[0] == 0 or gt.shape[0] == 0:
            # Handle empty predictions or ground truth
            iou_matrices.append(torch.zeros(pred.shape[0], gt.shape[0]))
        else:
            iou_matrices.append(compute_mask_iou(pred, gt, eps))
    return iou_matrices


def hungarian_matching(
    iou_matrix: torch.Tensor,
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
    """
    Greedy matching based on IoU (simpler than Hungarian for our use case).
    
    Args:
        iou_matrix: [N, M] IoU matrix (predictions x ground truth)
        iou_threshold: Minimum IoU to consider a match
        
    Returns:
        matches: List of (pred_idx, gt_idx) pairs
        matched_ious: IoU values for matched pairs
    """
    if iou_matrix.numel() == 0:
        return [], torch.tensor([])
        
    N, M = iou_matrix.shape
    matches = []
    matched_ious = []
    
    # Track which gt masks have been matched
    gt_matched = set()
    pred_matched = set()
    
    # Sort all IoU values in descending order
    flat_iou = iou_matrix.flatten()
    sorted_indices = torch.argsort(flat_iou, descending=True)
    
    for idx in sorted_indices:
        iou_val = flat_iou[idx].item()
        if iou_val < iou_threshold:
            break
            
        pred_idx = idx.item() // M
        gt_idx = idx.item() % M
        
        if pred_idx not in pred_matched and gt_idx not in gt_matched:
            matches.append((pred_idx, gt_idx))
            matched_ious.append(iou_val)
            pred_matched.add(pred_idx)
            gt_matched.add(gt_idx)
            
    return matches, torch.tensor(matched_ious) if matched_ious else torch.tensor([])


def compute_metrics_single_image(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    iou_threshold: float = 0.5,
    use_quality_iou: bool = True,
    fp_penalty_weight: float = 1.0,
    filter_oversized: bool = True,
    max_size_ratio: float = 2.0,
) -> Dict[str, float]:
    """
    Compute metrics for a single image.
    
    Args:
        pred_masks: [N, H, W] predicted masks
        gt_masks: [M, H, W] ground truth masks
        iou_threshold: Threshold for matching
        use_quality_iou: If True, use quality-adjusted IoU that penalizes over-segmentation
        fp_penalty_weight: Weight for false positive penalty (only if use_quality_iou=True)
        filter_oversized: If True, filter out masks larger than max_size_ratio * avg GT size
        max_size_ratio: Maximum allowed ratio of pred size to avg GT size
        
    Returns:
        Dictionary of metrics
    """
    N_original = pred_masks.shape[0]
    M = gt_masks.shape[0]
    num_filtered = 0
    
    # Filter oversized masks if enabled
    if filter_oversized and pred_masks.shape[0] > 0 and gt_masks.shape[0] > 0:
        pred_masks, num_filtered = filter_oversized_masks(pred_masks, gt_masks, max_size_ratio)
    
    N = pred_masks.shape[0]
    
    if M == 0:
        # No ground truth - can't compute meaningful metrics
        return {
            'mean_iou': 0.0,
            'mean_quality_iou': 0.0,
            'recall': 0.0,
            'precision': 1.0 if N == 0 else 0.0,
            'f1': 0.0,
            'num_matched': 0,
            'num_predictions': N,
            'num_predictions_original': N_original,
            'num_ground_truth': 0,
            'num_filtered': num_filtered,
        }
        
    if N == 0:
        # No predictions (or all filtered)
        return {
            'mean_iou': 0.0,
            'mean_quality_iou': 0.0,
            'recall': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'num_matched': 0,
            'num_predictions': 0,
            'num_predictions_original': N_original,
            'num_ground_truth': M,
            'num_filtered': num_filtered,
        }
    
    # Compute both regular IoU and quality-adjusted IoU
    iou_matrix = compute_mask_iou(pred_masks, gt_masks)
    
    if use_quality_iou:
        # Use quality-adjusted IoU for matching
        quality_matrix = compute_quality_iou(pred_masks, gt_masks, fp_penalty_weight)
        matches, matched_qualities = hungarian_matching(quality_matrix, iou_threshold)
        
        # Also get the raw IoU values for matched pairs (for reporting)
        matched_ious = torch.tensor([iou_matrix[p, g].item() for p, g in matches]) if matches else torch.tensor([])
    else:
        matches, matched_ious = hungarian_matching(iou_matrix, iou_threshold)
        matched_qualities = matched_ious
    
    num_matched = len(matches)
    
    # Compute metrics
    mean_iou = matched_ious.mean().item() if num_matched > 0 else 0.0
    mean_quality_iou = matched_qualities.mean().item() if num_matched > 0 else 0.0
    recall = num_matched / M
    precision = num_matched / N
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'mean_iou': mean_iou,
        'mean_quality_iou': mean_quality_iou,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'num_matched': num_matched,
        'num_predictions': N,
        'num_predictions_original': N_original,
        'num_ground_truth': M,
        'num_filtered': num_filtered,
    }


def compute_metrics_batch(
    pred_masks_batch: List[torch.Tensor],
    gt_masks_batch: List[torch.Tensor],
    iou_threshold: float = 0.5,
    use_quality_iou: bool = True,
    fp_penalty_weight: float = 1.0,
    filter_oversized: bool = True,
    max_size_ratio: float = 2.0,
) -> MetricResult:
    """
    Compute aggregate metrics over a batch of images.
    
    Args:
        pred_masks_batch: List of [N_i, H, W] predicted masks
        gt_masks_batch: List of [M_i, H, W] ground truth masks
        iou_threshold: Threshold for matching
        use_quality_iou: If True, use quality-adjusted IoU that penalizes over-segmentation
        fp_penalty_weight: Weight for false positive penalty
        filter_oversized: If True, filter out masks larger than max_size_ratio * avg GT size
        max_size_ratio: Maximum allowed ratio of pred size to avg GT size
        
    Returns:
        MetricResult with aggregated metrics
    """
    all_metrics = []
    total_matched = 0
    total_predictions = 0
    total_gt = 0
    total_filtered = 0
    
    # Coverage tracking
    num_images = 0
    num_images_with_predictions = 0
    
    # Also track recall at different thresholds
    matched_at_50 = 0
    matched_at_75 = 0
    
    for pred_masks, gt_masks in zip(pred_masks_batch, gt_masks_batch):
        num_images += 1
        
        metrics = compute_metrics_single_image(
            pred_masks, gt_masks, iou_threshold, 
            use_quality_iou=use_quality_iou,
            fp_penalty_weight=fp_penalty_weight,
            filter_oversized=filter_oversized,
            max_size_ratio=max_size_ratio,
        )
        all_metrics.append(metrics)
        
        total_matched += metrics['num_matched']
        total_predictions += metrics['num_predictions']
        total_gt += metrics['num_ground_truth']
        total_filtered += metrics.get('num_filtered', 0)
        
        # Track coverage
        if metrics['num_predictions'] > 0:
            num_images_with_predictions += 1
        
        # Compute matches at different thresholds (on filtered masks)
        # Need to re-filter for threshold computation
        if filter_oversized and pred_masks.shape[0] > 0 and gt_masks.shape[0] > 0:
            pred_masks_filtered, _ = filter_oversized_masks(pred_masks, gt_masks, max_size_ratio)
        else:
            pred_masks_filtered = pred_masks
            
        if pred_masks_filtered.shape[0] > 0 and gt_masks.shape[0] > 0:
            if use_quality_iou:
                score_matrix = compute_quality_iou(pred_masks_filtered, gt_masks, fp_penalty_weight)
            else:
                score_matrix = compute_mask_iou(pred_masks_filtered, gt_masks)
            matches_50, _ = hungarian_matching(score_matrix, 0.5)
            matches_75, _ = hungarian_matching(score_matrix, 0.75)
            matched_at_50 += len(matches_50)
            matched_at_75 += len(matches_75)
    
    # Aggregate metrics
    if total_gt > 0:
        mean_iou = np.mean([m['mean_iou'] for m in all_metrics if m['num_ground_truth'] > 0])
        mean_quality_iou = np.mean([m['mean_quality_iou'] for m in all_metrics if m['num_ground_truth'] > 0])
        instance_recall = total_matched / total_gt
        recall_at_50 = matched_at_50 / total_gt
        recall_at_75 = matched_at_75 / total_gt
    else:
        mean_iou = 0.0
        mean_quality_iou = 0.0
        instance_recall = 0.0
        recall_at_50 = 0.0
        recall_at_75 = 0.0
        
    if total_predictions > 0:
        instance_precision = total_matched / total_predictions
    else:
        instance_precision = 0.0
        
    if instance_precision + instance_recall > 0:
        f1_score = 2 * instance_precision * instance_recall / (instance_precision + instance_recall)
    else:
        f1_score = 0.0
    
    return MetricResult(
        mean_iou=float(mean_iou),
        instance_recall=instance_recall,
        instance_precision=instance_precision,
        f1_score=f1_score,
        recall_at_50=recall_at_50,
        recall_at_75=recall_at_75,
        num_predictions=total_predictions,
        num_ground_truth=total_gt,
        num_matched=total_matched,
        mean_quality_iou=float(mean_quality_iou),
        num_filtered=total_filtered,
        num_images=num_images,
        num_images_with_predictions=num_images_with_predictions,
    )


def compute_combined_score(
    metrics: MetricResult,
    iou_weight: float = 0.4,
    recall_weight: float = 0.4,
    precision_weight: float = 0.2,
    use_quality_iou: bool = True,
    min_coverage: float = 0.8,
) -> float:
    """
    Compute weighted combined score for optimization.
    
    We emphasize recall because we want to find ALL instances.
    Using quality_iou penalizes oversized/sloppy masks.
    Coverage penalty ensures predictions on most images.
    
    Args:
        metrics: MetricResult object
        iou_weight: Weight for IoU term
        recall_weight: Weight for recall term
        precision_weight: Weight for precision term
        use_quality_iou: Use quality-adjusted IoU instead of raw IoU
        min_coverage: Minimum fraction of images that should have predictions
    """
    if use_quality_iou:
        iou_term = iou_weight * metrics.mean_quality_iou
    else:
        iou_term = iou_weight * metrics.mean_iou
    
    base_score = (
        iou_term +
        recall_weight * metrics.instance_recall +
        precision_weight * metrics.instance_precision
    )
    
    # Coverage penalty: if we're missing predictions on too many images, penalize
    coverage = metrics.coverage if hasattr(metrics, 'coverage') else 1.0
    if coverage < min_coverage:
        # Proportional penalty
        coverage_penalty = coverage / min_coverage
        base_score *= coverage_penalty
    
    return base_score


class MetricTracker:
    """Track metrics over optimization run."""
    
    def __init__(self):
        self.history = []
        self.best_score = -float('inf')
        self.best_metrics = None
        self.best_tokens = None
        self.patience_counter = 0
        
    def update(
        self,
        metrics: MetricResult,
        tokens: List[int],
        score: Optional[float] = None
    ) -> bool:
        """
        Update tracker with new metrics.
        
        Returns:
            True if this is a new best score
        """
        if score is None:
            score = compute_combined_score(metrics)
            
        self.history.append({
            'metrics': metrics.to_dict(),
            'tokens': tokens.copy(),
            'score': score,
        })
        
        is_best = score > self.best_score
        if is_best:
            self.best_score = score
            self.best_metrics = metrics
            self.best_tokens = tokens.copy()
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        return is_best
    
    def should_stop(self, patience: int) -> bool:
        """Check if we should early stop."""
        return self.patience_counter >= patience
    
    def get_summary(self) -> Dict:
        """Get summary of optimization run."""
        return {
            'best_score': self.best_score,
            'best_metrics': self.best_metrics.to_dict() if self.best_metrics else None,
            'best_tokens': self.best_tokens,
            'num_iterations': len(self.history),
        }


# Loss functions for gradient-based optimization

def soft_iou_loss(
    pred_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Differentiable IoU loss using soft masks.
    
    Args:
        pred_logits: [N, H, W] predicted logits (before sigmoid)
        gt_masks: [M, H, W] ground truth masks
        
    Returns:
        Loss value (1 - mean best IoU)
    """
    pred_soft = torch.sigmoid(pred_logits)  # [N, H, W]
    
    if pred_soft.shape[0] == 0 or gt_masks.shape[0] == 0:
        return torch.tensor(1.0, device=pred_logits.device)
    
    N, H, W = pred_soft.shape
    M = gt_masks.shape[0]
    
    # Compute soft IoU for all pairs
    pred_flat = pred_soft.view(N, 1, H * W)  # [N, 1, H*W]
    gt_flat = gt_masks.float().view(1, M, H * W)  # [1, M, H*W]
    
    intersection = (pred_flat * gt_flat).sum(dim=2)  # [N, M]
    union = pred_flat.sum(dim=2) + gt_flat.sum(dim=2) - intersection  # [N, M]
    
    iou_matrix = intersection / (union + eps)  # [N, M]
    
    # For each GT mask, find best matching prediction
    best_iou_per_gt = iou_matrix.max(dim=0)[0]  # [M]
    
    # Loss is 1 - mean IoU
    return 1.0 - best_iou_per_gt.mean()


def dice_loss(
    pred_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Dice loss for mask prediction.
    """
    pred_soft = torch.sigmoid(pred_logits)
    
    if pred_soft.shape[0] == 0 or gt_masks.shape[0] == 0:
        return torch.tensor(1.0, device=pred_logits.device)
    
    N, H, W = pred_soft.shape
    M = gt_masks.shape[0]
    
    pred_flat = pred_soft.view(N, 1, H * W)
    gt_flat = gt_masks.float().view(1, M, H * W)
    
    intersection = (pred_flat * gt_flat).sum(dim=2)
    dice = (2 * intersection) / (pred_flat.sum(dim=2) + gt_flat.sum(dim=2) + eps)
    
    best_dice_per_gt = dice.max(dim=0)[0]
    
    return 1.0 - best_dice_per_gt.mean()


def combined_mask_loss(
    pred_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    iou_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Combined IoU and Dice loss."""
    return (
        iou_weight * soft_iou_loss(pred_logits, gt_masks) +
        dice_weight * dice_loss(pred_logits, gt_masks)
    )