"""
Detailed Evaluation Module with Per-Image Tracking and Negative Handling.

This module provides:
1. Per-image result tracking (image path, augmentations, predictions, GT, metrics)
2. Negative image handling (penalize any prediction on negatives)
3. Combined fitness that accounts for both positive and negative performance

Usage:
    from detailed_eval import DetailedEvaluator, ImageResult, ConceptEvalResult
    
    evaluator = DetailedEvaluator(model, config)
    result = evaluator.evaluate_concept(tokens, concept_data, include_negatives=True)
    
    # Access per-image results
    for img_result in result.image_results:
        print(f"{img_result.image_path}: IoU={img_result.iou}, FP={img_result.false_positive_rate}")
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
        return "+".join(augs) if augs else "none"


@dataclass
class ImageResult:
    """Detailed result for a single image evaluation."""
    # Image identification
    image_path: str
    image_index: int
    is_positive: bool  # True if this is a positive (has GT masks), False if negative
    
    # Augmentation applied
    augmentation: AugmentationRecord
    
    # Predictions
    num_predictions: int
    prediction_confidences: List[float]  # Confidence scores for each prediction
    prediction_areas: List[int]  # Pixel area of each predicted mask
    
    # Ground truth (only for positives)
    num_ground_truth: int
    ground_truth_areas: List[int]
    
    # Matching (only for positives)
    num_matched: int
    matched_ious: List[float]  # IoU for each matched pair
    
    # Aggregate metrics for this image
    mean_iou: float  # Average IoU of matched pairs (0 if no matches)
    false_positive_rate: float  # Fraction of image covered by unmatched predictions
    quality_iou: float  # mean_iou * (1 - false_positive_rate)
    
    # For negatives: any prediction is a false positive
    negative_penalty: float  # For negatives: area of predictions / image area
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['augmentation'] = self.augmentation.to_dict()
        return d


@dataclass
class ConceptEvalResult:
    """Complete evaluation result for a concept."""
    concept_name: str
    tokens: List[int]
    decoded_prompt: str
    
    # Split info
    num_positive_images: int
    num_negative_images: int
    
    # Per-image results
    image_results: List[ImageResult]
    
    # Aggregate metrics - positives
    positive_mean_iou: float
    positive_mean_quality_iou: float
    positive_recall: float  # fraction of GT instances matched
    positive_precision: float  # fraction of predictions that matched
    
    # Aggregate metrics - negatives
    negative_false_positive_rate: float  # avg prediction area on negatives
    negative_clean_rate: float  # fraction of negatives with zero predictions
    
    # Combined fitness
    fitness: float  # Combined score accounting for both
    
    # Timing
    eval_time_seconds: float
    
    def to_dict(self) -> dict:
        d = {
            'concept_name': self.concept_name,
            'tokens': self.tokens,
            'decoded_prompt': self.decoded_prompt,
            'num_positive_images': self.num_positive_images,
            'num_negative_images': self.num_negative_images,
            'positive_mean_iou': self.positive_mean_iou,
            'positive_mean_quality_iou': self.positive_mean_quality_iou,
            'positive_recall': self.positive_recall,
            'positive_precision': self.positive_precision,
            'negative_false_positive_rate': self.negative_false_positive_rate,
            'negative_clean_rate': self.negative_clean_rate,
            'fitness': self.fitness,
            'eval_time_seconds': self.eval_time_seconds,
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
        
        # Reconstruct image results
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
        use_augmentation: bool = True,
        max_positive_images: int = None,
        max_negative_images: int = None,
        confidence_threshold: float = 0.5,
    ) -> ConceptEvalResult:
        """
        Evaluate tokens on a concept with detailed per-image tracking.
        
        Args:
            tokens: Token sequence to evaluate
            concept: SaCoConceptData with positive and negative images
            include_negatives: Whether to evaluate on negative images
            use_augmentation: Whether to apply augmentation
            max_positive_images: Limit on positive images (None = all)
            max_negative_images: Limit on negative images (None = all)
            confidence_threshold: Threshold for mask predictions
            
        Returns:
            ConceptEvalResult with per-image details
        """
        from augmentation import AugmentationConfig, apply_augmentation
        from PIL import Image
        import random
        
        start_time = time.time()
        
        decoded_prompt = self.tokenizer.decode(tokens)
        image_results = []
        
        # Prepare augmentation config
        if use_augmentation:
            aug_config = AugmentationConfig(
                enabled=True,
                include_original=True,
                augmentation_multiplier=2,
            )
        else:
            aug_config = None
        
        # Process positive images
        n_pos = concept.num_positive_images
        if max_positive_images:
            n_pos = min(n_pos, max_positive_images)
        
        for idx in range(n_pos):
            img_path = concept.positive_image_paths[idx]
            gt_masks = concept.positive_masks[idx]
            
            # Evaluate original
            result = self._evaluate_single_image(
                tokens=tokens,
                image_path=img_path,
                gt_masks=gt_masks,
                image_index=idx,
                is_positive=True,
                augmentation=AugmentationRecord.none(),
                confidence_threshold=confidence_threshold,
            )
            image_results.append(result)
            
            # Evaluate augmented versions
            if aug_config and aug_config.enabled:
                for aug_idx in range(aug_config.augmentation_multiplier):
                    aug_record, aug_image, aug_masks = self._apply_random_augmentation(
                        img_path, gt_masks, aug_config
                    )
                    result = self._evaluate_single_image_tensor(
                        tokens=tokens,
                        image_tensor=aug_image,
                        gt_masks=aug_masks,
                        image_path=img_path,
                        image_index=idx,
                        is_positive=True,
                        augmentation=aug_record,
                        confidence_threshold=confidence_threshold,
                    )
                    image_results.append(result)
        
        # Process negative images
        if include_negatives:
            n_neg = concept.num_negative_images
            if max_negative_images:
                n_neg = min(n_neg, max_negative_images)
            
            for idx in range(n_neg):
                img_path = concept.negative_image_paths[idx]
                
                # Evaluate original
                result = self._evaluate_single_image(
                    tokens=tokens,
                    image_path=img_path,
                    gt_masks=[],  # No ground truth for negatives
                    image_index=idx,
                    is_positive=False,
                    augmentation=AugmentationRecord.none(),
                    confidence_threshold=confidence_threshold,
                )
                image_results.append(result)
                
                # Evaluate augmented versions
                if aug_config and aug_config.enabled:
                    for aug_idx in range(aug_config.augmentation_multiplier):
                        aug_record, aug_image, _ = self._apply_random_augmentation(
                            img_path, [], aug_config
                        )
                        result = self._evaluate_single_image_tensor(
                            tokens=tokens,
                            image_tensor=aug_image,
                            gt_masks=[],
                            image_path=img_path,
                            image_index=idx,
                            is_positive=False,
                            augmentation=aug_record,
                            confidence_threshold=confidence_threshold,
                        )
                        image_results.append(result)
        
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
            negative_false_positive_rate = np.mean([r.negative_penalty for r in negative_results])
            negative_clean_rate = np.mean([1.0 if r.num_predictions == 0 else 0.0 for r in negative_results])
        else:
            negative_false_positive_rate = 0
            negative_clean_rate = 1.0
        
        # Combined fitness
        # Reward high quality IoU on positives, penalize predictions on negatives
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
    
    def _evaluate_single_image(
        self,
        tokens: List[int],
        image_path: str,
        gt_masks: List[np.ndarray],
        image_index: int,
        is_positive: bool,
        augmentation: AugmentationRecord,
        confidence_threshold: float,
    ) -> ImageResult:
        """Evaluate on a single image loaded from path."""
        from PIL import Image
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_size = self.config.data.image_size
        image = image.resize((image_size, image_size), Image.BILINEAR)
        
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dim
        
        # Process GT masks
        gt_tensors = []
        for mask in gt_masks:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_resized = mask_pil.resize((image_size, image_size), Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_resized) > 127).bool()
            gt_tensors.append(mask_tensor)
        
        return self._evaluate_single_image_tensor(
            tokens=tokens,
            image_tensor=image_tensor.squeeze(0),
            gt_masks=gt_tensors,
            image_path=image_path,
            image_index=image_index,
            is_positive=is_positive,
            augmentation=augmentation,
            confidence_threshold=confidence_threshold,
        )
    
    def _evaluate_single_image_tensor(
        self,
        tokens: List[int],
        image_tensor: torch.Tensor,  # [C, H, W]
        gt_masks: List[torch.Tensor],  # List of [H, W] bool tensors
        image_path: str,
        image_index: int,
        is_positive: bool,
        augmentation: AugmentationRecord,
        confidence_threshold: float,
    ) -> ImageResult:
        """Evaluate on a single image tensor."""
        self.model.eval()
        
        image_size = image_tensor.shape[-1]
        image_area = image_size * image_size
        
        # Prepare image batch
        images = image_tensor.unsqueeze(0).to(self.device)  # [1, C, H, W]
        
        # Decode tokens to string
        text_string = self.tokenizer.decode(tokens)
        
        with torch.no_grad():
            # Forward pass
            backbone_out = self.model.backbone.forward_image(images)
            text_out = self.model.backbone.forward_text([text_string], device=self.device)
            backbone_out.update(text_out)
            
            from sam3.model.data_misc import FindStage
            find_input = FindStage(
                img_ids=torch.tensor([0], device=self.device),
                text_ids=torch.tensor([0], device=self.device),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            
            geometric_prompt = self.model._get_dummy_prompt(num_prompts=1)
            
            out = self.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=None,
                geometric_prompt=geometric_prompt,
            )
            
            pred_masks = out.get('pred_masks')
            pred_logits = out.get('pred_logits')
        
        # Process predictions
        prediction_confidences = []
        prediction_areas = []
        pred_binary_masks = []
        
        if pred_masks is not None and pred_logits is not None:
            probs = pred_logits.sigmoid().squeeze(-1)  # [1, num_masks]
            
            keep = probs[0] > confidence_threshold
            kept_masks = pred_masks[0][keep]  # [num_kept, H, W] logits
            kept_probs = probs[0][keep]
            
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
                
                # Sigmoid and threshold
                mask_prob = mask_logits.sigmoid()
                mask_binary = mask_prob > 0.5
                
                prediction_confidences.append(kept_probs[i].item())
                prediction_areas.append(mask_binary.sum().item())
                pred_binary_masks.append(mask_binary)
        
        num_predictions = len(pred_binary_masks)
        
        # Ground truth processing
        num_ground_truth = len(gt_masks)
        ground_truth_areas = [m.sum().item() for m in gt_masks]
        
        # Matching (for positives)
        num_matched = 0
        matched_ious = []
        matched_pred_indices = set()
        
        if is_positive and num_ground_truth > 0 and num_predictions > 0:
            # Simple greedy matching by IoU
            for gt_mask in gt_masks:
                gt_mask = gt_mask.to(self.device)
                best_iou = 0
                best_pred_idx = -1
                
                for pred_idx, pred_mask in enumerate(pred_binary_masks):
                    if pred_idx in matched_pred_indices:
                        continue
                    
                    intersection = (gt_mask & pred_mask).sum().item()
                    union = (gt_mask | pred_mask).sum().item()
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = pred_idx
                
                if best_iou > 0.5:  # IoU threshold for matching
                    num_matched += 1
                    matched_ious.append(best_iou)
                    matched_pred_indices.add(best_pred_idx)
        
        # Compute metrics
        mean_iou = np.mean(matched_ious) if matched_ious else 0.0
        
        # False positive rate: area covered by unmatched predictions / image area
        unmatched_area = 0
        for pred_idx, pred_mask in enumerate(pred_binary_masks):
            if pred_idx not in matched_pred_indices:
                unmatched_area += pred_mask.sum().item()
        false_positive_rate = unmatched_area / image_area
        
        quality_iou = mean_iou * (1 - false_positive_rate)
        
        # Negative penalty: any prediction area / image area
        if not is_positive:
            total_pred_area = sum(prediction_areas)
            negative_penalty = min(1.0, total_pred_area / image_area)
        else:
            negative_penalty = 0.0
        
        return ImageResult(
            image_path=image_path,
            image_index=image_index,
            is_positive=is_positive,
            augmentation=augmentation,
            num_predictions=num_predictions,
            prediction_confidences=prediction_confidences,
            prediction_areas=prediction_areas,
            num_ground_truth=num_ground_truth,
            ground_truth_areas=ground_truth_areas,
            num_matched=num_matched,
            matched_ious=matched_ious,
            mean_iou=mean_iou,
            false_positive_rate=false_positive_rate,
            quality_iou=quality_iou,
            negative_penalty=negative_penalty,
        )
    
    def _apply_random_augmentation(
        self,
        image_path: str,
        gt_masks: List[np.ndarray],
        aug_config,
    ) -> Tuple[AugmentationRecord, torch.Tensor, List[torch.Tensor]]:
        """Apply random augmentation and return record + augmented data."""
        from PIL import Image
        from augmentation import (
            horizontal_flip, gaussian_blur, gaussian_noise,
            adjust_brightness, adjust_contrast
        )
        import random
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_size = self.config.data.image_size
        image = image.resize((image_size, image_size), Image.BILINEAR)
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Process GT masks
        mask_tensors = []
        for mask in gt_masks:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_resized = mask_pil.resize((image_size, image_size), Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_resized) > 127).bool()
            mask_tensors.append(mask_tensor)
        
        if mask_tensors:
            masks_stacked = torch.stack(mask_tensors)
        else:
            masks_stacked = torch.zeros((0, image_size, image_size), dtype=torch.bool)
        
        # Apply augmentations and track what was applied
        record = AugmentationRecord()
        
        # Horizontal flip
        if random.random() < aug_config.horizontal_flip_prob:
            image_tensor, masks_stacked = horizontal_flip(image_tensor, masks_stacked)
            record.horizontal_flip = True
        
        # Gaussian blur
        if random.random() < aug_config.gaussian_blur_prob:
            sigma = random.uniform(*aug_config.blur_sigma_range)
            image_tensor = gaussian_blur(image_tensor, aug_config.blur_kernel_size, sigma)
            record.gaussian_blur = True
            record.blur_sigma = sigma
        
        # Gaussian noise
        if random.random() < aug_config.gaussian_noise_prob:
            std = random.uniform(*aug_config.noise_std_range)
            image_tensor = gaussian_noise(image_tensor, std)
            record.gaussian_noise = True
            record.noise_std = std
        
        # Brightness
        if random.random() < aug_config.brightness_prob:
            factor = random.uniform(*aug_config.brightness_range)
            image_tensor = adjust_brightness(image_tensor, factor)
            record.brightness_adjusted = True
            record.brightness_factor = factor
        
        # Contrast
        if random.random() < aug_config.contrast_prob:
            factor = random.uniform(*aug_config.contrast_range)
            image_tensor = adjust_contrast(image_tensor, factor)
            record.contrast_adjusted = True
            record.contrast_factor = factor
        
        # Unstack masks back to list
        mask_list = [masks_stacked[i] for i in range(masks_stacked.shape[0])]
        
        return record, image_tensor, mask_list


def create_detailed_evaluate_fn(
    model,
    concept,  # SaCoConceptData
    config,
    include_negatives: bool = True,
    use_augmentation: bool = True,
    max_positive_images: int = None,
    max_negative_images: int = None,
):
    """
    Create evaluation function that returns detailed results.
    
    For use with discrete search - wraps DetailedEvaluator.
    """
    evaluator = DetailedEvaluator(model, config)
    
    def evaluate_fn(tokens: List[int]) -> Tuple[float, ConceptEvalResult]:
        result = evaluator.evaluate_concept(
            tokens=tokens,
            concept=concept,
            include_negatives=include_negatives,
            use_augmentation=use_augmentation,
            max_positive_images=max_positive_images,
            max_negative_images=max_negative_images,
        )
        return result.fitness, result
    
    return evaluate_fn
