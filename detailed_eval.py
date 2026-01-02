"""
Detailed Evaluation Module with Per-Image Tracking and Negative Handling.
OPTIMIZED VERSION: Batched image processing + cached text encoding.

This module provides:
1. Per-image result tracking (image path, augmentations, predictions, GT, metrics)
2. Negative image handling (penalize any prediction on negatives)
3. Combined fitness that accounts for both positive and negative performance
4. BATCHED inference for 5-10x speedup on GPU

Usage:
    from detailed_eval import DetailedEvaluator, ImageResult, ConceptEvalResult
    
    evaluator = DetailedEvaluator(model, config)
    result = evaluator.evaluate_concept(tokens, concept_data, include_negatives=True)
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
import time


@dataclass
class AugmentationRecord:
    """Record of augmentations applied to an image."""
    horizontal_flip: bool = False
    gaussian_blur: bool = False
    blur_sigma: Optional[float] = None
    gaussian_noise: bool = False
    noise_std: Optional[float] = None
    brightness_adjusted: bool = False
    brightness_factor: Optional[float] = None
    contrast_adjusted: bool = False
    contrast_factor: Optional[float] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def none(cls) -> 'AugmentationRecord':
        """Return record indicating no augmentation."""
        return cls()
    
    def __str__(self) -> str:
        augs = []
        if self.horizontal_flip:
            augs.append("flip")
        if self.gaussian_blur:
            augs.append(f"blur({self.blur_sigma:.2f})")
        if self.gaussian_noise:
            augs.append(f"noise({self.noise_std:.3f})")
        if self.brightness_adjusted:
            augs.append(f"bright({self.brightness_factor:.2f})")
        if self.contrast_adjusted:
            augs.append(f"contrast({self.contrast_factor:.2f})")
        return ", ".join(augs) if augs else "none"


@dataclass
class ImageResult:
    """Detailed result for a single image evaluation."""
    # Identity
    image_path: str
    image_index: int
    is_positive: bool
    augmentation: AugmentationRecord
    
    # Predictions
    num_predictions: int
    prediction_confidences: List[float]
    prediction_areas: List[int]  # Pixel counts
    
    # Ground truth
    num_ground_truth: int
    ground_truth_areas: List[int]
    
    # Matching (for positives)
    num_matched: int  # GT masks that matched a prediction (IoU > 0.5)
    matched_ious: List[float]
    
    # Aggregate metrics for this image
    mean_iou: float  # Mean IoU of matched masks
    false_positive_rate: float  # For negatives: prediction_area / image_area
    quality_iou: float  # mean_iou * recall (rewards both quality and quantity)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['augmentation'] = self.augmentation.to_dict()
        return d
    
    @property
    def negative_penalty(self) -> float:
        """Penalty for this image if it's a negative."""
        return self.false_positive_rate if not self.is_positive else 0.0


@dataclass
class ConceptEvalResult:
    """Full evaluation result for a concept."""
    concept_name: str
    tokens: List[int]
    decoded_prompt: str
    
    # Counts
    num_positive_images: int
    num_negative_images: int
    
    # Per-image results
    image_results: List[ImageResult]
    
    # Aggregate positive metrics
    positive_mean_iou: float
    positive_mean_quality_iou: float
    positive_recall: float
    positive_precision: float
    
    # Aggregate negative metrics
    negative_false_positive_rate: float
    negative_clean_rate: float  # Fraction of negatives with zero predictions
    
    # Combined fitness
    fitness: float
    
    # Timing
    eval_time_seconds: float
    
    def to_dict(self) -> dict:
        d = {
            'concept_name': self.concept_name,
            'tokens': self.tokens,
            'decoded_prompt': self.decoded_prompt,
            'num_positive_images': self.num_positive_images,
            'num_negative_images': self.num_negative_images,
            'positive_mean_iou': float(self.positive_mean_iou),
            'positive_mean_quality_iou': float(self.positive_mean_quality_iou),
            'positive_recall': float(self.positive_recall),
            'positive_precision': float(self.positive_precision),
            'negative_false_positive_rate': float(self.negative_false_positive_rate),
            'negative_clean_rate': float(self.negative_clean_rate),
            'fitness': float(self.fitness),
            'eval_time_seconds': float(self.eval_time_seconds),
            'image_results': [r.to_dict() for r in self.image_results],
        }
        return d
    
    def save(self, path: str):
        """Save result to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ConceptEvalResult':
        """Load result from JSON file."""
        with open(path, 'r') as f:
            d = json.load(f)
        
        image_results = []
        for ir_dict in d['image_results']:
            aug_dict = ir_dict.pop('augmentation')
            aug = AugmentationRecord(**aug_dict)
            ir = ImageResult(augmentation=aug, **ir_dict)
            image_results.append(ir)
        
        d['image_results'] = image_results
        return cls(**d)


class DetailedEvaluator:
    """
    Evaluator that tracks per-image results and handles negatives.
    OPTIMIZED: Uses batched inference and cached text encoding.
    """
    
    def __init__(
        self,
        model,
        config,
        device: str = None,
    ):
        self.model = model
        self.config = config
        self.device = device or config.device
        self.tokenizer = model.backbone.language_backbone.tokenizer
    
    def evaluate_concept(
        self,
        tokens: List[int],
        concept,  # SaCoConceptData
        include_negatives: bool = True,
        use_augmentation: bool = False,  # Disabled by default for speed
        max_positive_images: int = None,
        max_negative_images: int = None,
        confidence_threshold: float = 0.5,
        batch_size: int = 8,  # Process images in batches
    ) -> ConceptEvalResult:
        """
        Evaluate tokens on a concept with detailed per-image tracking.
        
        OPTIMIZED:
        - Text encoding cached (done once, not per image)
        - Batched image processing
        
        Args:
            tokens: Token sequence to evaluate
            concept: SaCoConceptData with positive and negative images
            include_negatives: Whether to evaluate on negative images
            use_augmentation: Whether to apply augmentation (default False for speed)
            max_positive_images: Limit on positive images (None = all)
            max_negative_images: Limit on negative images (None = all)
            confidence_threshold: Threshold for mask predictions
            batch_size: Number of images to process at once
            
        Returns:
            ConceptEvalResult with per-image details
        """
        from PIL import Image
        
        start_time = time.time()
        
        decoded_prompt = self.tokenizer.decode(tokens)
        
        # Cache text encoding ONCE for all images
        with torch.no_grad():
            text_out = self.model.backbone.forward_text([decoded_prompt], device=self.device)
        
        # Collect all images to evaluate
        eval_items = []  # List of (image_path, gt_masks, is_positive, image_index)
        
        # Positive images
        n_pos = concept.num_positive_images
        if max_positive_images:
            n_pos = min(n_pos, max_positive_images)
        
        for idx in range(n_pos):
            img_path = concept.positive_image_paths[idx]
            gt_masks = concept.positive_masks[idx]
            eval_items.append((img_path, gt_masks, True, idx))
        
        # Negative images
        if include_negatives:
            n_neg = concept.num_negative_images
            if max_negative_images:
                n_neg = min(n_neg, max_negative_images)
            
            for idx in range(n_neg):
                img_path = concept.negative_image_paths[idx]
                eval_items.append((img_path, [], False, idx))
        
        # Process in batches
        image_results = []
        for batch_start in range(0, len(eval_items), batch_size):
            batch_items = eval_items[batch_start:batch_start + batch_size]
            batch_results = self._evaluate_batch(
                batch_items=batch_items,
                text_out=text_out,
                confidence_threshold=confidence_threshold,
            )
            image_results.extend(batch_results)
        
        # Aggregate metrics
        positive_results = [r for r in image_results if r.is_positive]
        negative_results = [r for r in image_results if not r.is_positive]
        
        # Positive aggregates
        if positive_results:
            positive_mean_iou = np.mean([r.mean_iou for r in positive_results])
            positive_mean_quality_iou = np.mean([r.quality_iou for r in positive_results])
            total_matched = sum(r.num_matched for r in positive_results)
            total_gt = sum(r.num_ground_truth for r in positive_results)
            total_pred = sum(r.num_predictions for r in positive_results)
            positive_recall = total_matched / total_gt if total_gt > 0 else 0
            positive_precision = total_matched / total_pred if total_pred > 0 else 0
        else:
            positive_mean_iou = 0
            positive_mean_quality_iou = 0
            positive_recall = 0
            positive_precision = 0
        
        # Negative aggregates
        if negative_results:
            negative_false_positive_rate = np.mean([r.false_positive_rate for r in negative_results])
            negative_clean_rate = np.mean([1.0 if r.num_predictions == 0 else 0.0 for r in negative_results])
        else:
            negative_false_positive_rate = 0
            negative_clean_rate = 1.0
        
        # Combined fitness
        fitness = self._compute_combined_fitness(
            positive_quality_iou=positive_mean_quality_iou,
            positive_recall=positive_recall,
            positive_precision=positive_precision,
            negative_fp_rate=negative_false_positive_rate,
            has_negatives=len(negative_results) > 0,
        )
        
        eval_time = time.time() - start_time
        
        return ConceptEvalResult(
            concept_name=concept.text_input,
            tokens=tokens,
            decoded_prompt=decoded_prompt,
            num_positive_images=len([r for r in image_results if r.is_positive]),
            num_negative_images=len([r for r in image_results if not r.is_positive]),
            image_results=image_results,
            positive_mean_iou=positive_mean_iou,
            positive_mean_quality_iou=positive_mean_quality_iou,
            positive_recall=positive_recall,
            positive_precision=positive_precision,
            negative_false_positive_rate=negative_false_positive_rate,
            negative_clean_rate=negative_clean_rate,
            fitness=fitness,
            eval_time_seconds=eval_time,
        )
    
    def _evaluate_batch(
        self,
        batch_items: List[tuple],  # List of (image_path, gt_masks, is_positive, image_index)
        text_out: dict,  # Cached text encoding
        confidence_threshold: float,
    ) -> List[ImageResult]:
        """Evaluate a batch of images with cached text encoding."""
        from PIL import Image as PILImage
        
        image_size = self.config.data.image_size
        results = []
        
        # Load and preprocess all images in batch
        image_tensors = []
        batch_metadata = []
        
        for img_path, gt_masks, is_positive, image_index in batch_items:
            # Load image
            image = PILImage.open(img_path).convert('RGB')
            image = image.resize((image_size, image_size), PILImage.BILINEAR)
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image_tensors.append(image_tensor)
            
            # Process GT masks
            gt_tensors = []
            for mask in gt_masks:
                if isinstance(mask, np.ndarray):
                    mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                    mask_resized = mask_pil.resize((image_size, image_size), PILImage.NEAREST)
                    mask_tensor = torch.from_numpy(np.array(mask_resized) > 127).bool()
                    gt_tensors.append(mask_tensor)
                else:
                    gt_tensors.append(mask)
            
            batch_metadata.append({
                'gt_masks': gt_tensors,
                'is_positive': is_positive,
                'image_index': image_index,
                'image_path': img_path,
            })
        
        # Stack into batch tensor
        images_batch = torch.stack(image_tensors).to(self.device)  # [B, C, H, W]
        batch_size = images_batch.shape[0]
        
        with torch.no_grad():
            # Forward pass on image batch
            backbone_out = self.model.backbone.forward_image(images_batch)
            
            # Expand text encoding to match batch size
            # Handle different tensor shapes - some may be [1, ...], some may be [max_batch, ...]
            expanded_text_out = {}
            for key, value in text_out.items():
                if isinstance(value, torch.Tensor):
                    if value.shape[0] == 1:
                        # Expand from [1, ...] to [batch_size, ...]
                        expanded_text_out[key] = value.expand(batch_size, *value.shape[1:]).contiguous()
                    elif value.shape[0] >= batch_size:
                        # Already large enough, just slice
                        expanded_text_out[key] = value[:batch_size].contiguous()
                    else:
                        # Repeat to match batch size
                        repeats = (batch_size + value.shape[0] - 1) // value.shape[0]
                        expanded_text_out[key] = value.repeat(repeats, *([1] * (len(value.shape) - 1)))[:batch_size].contiguous()
                else:
                    expanded_text_out[key] = value
            
            backbone_out.update(expanded_text_out)
            
            from sam3.model.data_misc import FindStage
            find_input = FindStage(
                img_ids=torch.arange(batch_size, device=self.device),
                text_ids=torch.zeros(batch_size, dtype=torch.long, device=self.device),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            
            geometric_prompt = self.model._get_dummy_prompt(num_prompts=batch_size)
            
            out = self.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=None,
                geometric_prompt=geometric_prompt,
            )
            
            pred_masks = out.get('pred_masks')  # [B, num_masks, H, W]
            pred_logits = out.get('pred_logits')  # [B, num_masks, 1]
        
        # Process predictions for each image in batch
        for b_idx, meta in enumerate(batch_metadata):
            result = self._process_single_prediction(
                pred_masks=pred_masks[b_idx] if pred_masks is not None else None,
                pred_logits=pred_logits[b_idx] if pred_logits is not None else None,
                gt_masks=meta['gt_masks'],
                is_positive=meta['is_positive'],
                image_index=meta['image_index'],
                image_path=meta['image_path'],
                confidence_threshold=confidence_threshold,
                image_size=image_size,
            )
            results.append(result)
        
        return results
    
    def _process_single_prediction(
        self,
        pred_masks,  # [num_masks, H, W] or None
        pred_logits,  # [num_masks, 1] or None
        gt_masks: List[torch.Tensor],
        is_positive: bool,
        image_index: int,
        image_path: str,
        confidence_threshold: float,
        image_size: int,
    ) -> ImageResult:
        """Process predictions for a single image."""
        image_area = image_size * image_size
        
        prediction_confidences = []
        prediction_areas = []
        pred_binary_masks = []
        
        if pred_masks is not None and pred_logits is not None:
            probs = pred_logits.sigmoid().squeeze(-1)  # [num_masks]
            
            keep = probs > confidence_threshold
            kept_masks = pred_masks[keep]
            kept_probs = probs[keep]
            
            for i in range(kept_masks.shape[0]):
                mask_logits = kept_masks[i]
                
                # Resize if needed
                if mask_logits.shape[-2:] != (image_size, image_size):
                    mask_logits = F.interpolate(
                        mask_logits.unsqueeze(0).unsqueeze(0).float(),
                        size=(image_size, image_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                
                mask_prob = mask_logits.sigmoid()
                mask_binary = mask_prob > 0.5
                
                prediction_confidences.append(kept_probs[i].item())
                prediction_areas.append(mask_binary.sum().item())
                pred_binary_masks.append(mask_binary)
        
        num_predictions = len(pred_binary_masks)
        num_ground_truth = len(gt_masks)
        ground_truth_areas = [m.sum().item() for m in gt_masks]
        
        # Matching (for positives)
        num_matched = 0
        matched_ious = []
        matched_pred_indices = set()
        
        if is_positive and num_predictions > 0 and num_ground_truth > 0:
            for gt_idx, gt_mask in enumerate(gt_masks):
                gt_mask_device = gt_mask.to(self.device)
                best_iou = 0
                best_pred_idx = -1
                
                for pred_idx, pred_mask in enumerate(pred_binary_masks):
                    if pred_idx in matched_pred_indices:
                        continue
                    
                    intersection = (pred_mask & gt_mask_device).sum().item()
                    union = (pred_mask | gt_mask_device).sum().item()
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = pred_idx
                
                if best_iou > 0.5:
                    num_matched += 1
                    matched_ious.append(best_iou)
                    matched_pred_indices.add(best_pred_idx)
        
        # Compute metrics
        mean_iou = np.mean(matched_ious) if matched_ious else 0.0
        
        if is_positive:
            recall = num_matched / num_ground_truth if num_ground_truth > 0 else 0
            quality_iou = mean_iou * recall
            false_positive_rate = 0.0
        else:
            recall = 0.0
            quality_iou = 0.0
            total_pred_area = sum(prediction_areas)
            false_positive_rate = total_pred_area / image_area if image_area > 0 else 0
        
        return ImageResult(
            image_path=image_path,
            image_index=image_index,
            is_positive=is_positive,
            augmentation=AugmentationRecord.none(),
            num_predictions=num_predictions,
            prediction_confidences=prediction_confidences,
            prediction_areas=prediction_areas,
            num_ground_truth=num_ground_truth,
            ground_truth_areas=ground_truth_areas,
            num_matched=num_matched,
            matched_ious=matched_ious,
            mean_iou=float(mean_iou),
            false_positive_rate=float(false_positive_rate),
            quality_iou=float(quality_iou),
        )
    
    def _compute_combined_fitness(
        self,
        positive_quality_iou: float,
        positive_recall: float,
        positive_precision: float,
        negative_fp_rate: float,
        has_negatives: bool,
    ) -> float:
        """
        Compute combined fitness score.
        
        Args:
            positive_quality_iou: Quality-adjusted IoU on positives
            positive_recall: Recall on positives
            positive_precision: Precision on positives
            negative_fp_rate: False positive rate on negatives (0-1)
            has_negatives: Whether we evaluated on negatives
            
        Returns:
            Combined fitness score (higher is better)
        """
        # Positive component (same as before)
        positive_score = (
            0.4 * positive_quality_iou +
            0.4 * positive_recall +
            0.2 * positive_precision
        )
        
        if not has_negatives:
            return positive_score
        
        # Negative penalty: reduce score based on false positive rate on negatives
        # If FP rate is 0, no penalty. If FP rate is 1, halve the score.
        negative_penalty = 0.5 * negative_fp_rate
        
        return positive_score * (1 - negative_penalty)