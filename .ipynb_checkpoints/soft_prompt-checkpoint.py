"""
Soft Prompt Optimization for SAM3.

This module implements gradient-based optimization of continuous
"soft prompt" embeddings that are injected into SAM3's text encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam, AdamW
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from config import Config
from metrics import soft_iou_loss, dice_loss, compute_metrics_batch, MetricResult
from embeddings import VocabularyEmbeddings


class SoftPromptModule(nn.Module):
    """
    Learnable soft prompt embeddings.
    
    These embeddings are injected into SAM3's text encoder pathway,
    bypassing the discrete tokenization step.
    """
    
    def __init__(
        self,
        seq_length: int,
        embed_dim: int,
        init_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        
        if init_embeddings is not None:
            assert init_embeddings.shape == (seq_length, embed_dim)
            self.soft_embeddings = nn.Parameter(init_embeddings.clone())
        else:
            # Random initialization
            self.soft_embeddings = nn.Parameter(
                torch.randn(seq_length, embed_dim) * 0.02
            )
    
    def forward(self) -> torch.Tensor:
        """Return current soft embeddings."""
        return self.soft_embeddings
    
    def get_with_special_tokens(
        self,
        sot_embedding: torch.Tensor,
        eot_embedding: torch.Tensor,
        positional_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get soft embeddings with SOT/EOT tokens and positional encoding.
        
        Returns:
            embeddings: [1, seq_length+2, embed_dim] with SOT + soft + EOT
            attention_mask: [1, seq_length+2] attention mask
        """
        # Construct sequence: SOT + soft_embeddings + EOT
        full_seq = torch.cat([
            sot_embedding.unsqueeze(0),  # [1, embed_dim]
            self.soft_embeddings,  # [seq_length, embed_dim]
            eot_embedding.unsqueeze(0),  # [1, embed_dim]
        ], dim=0)  # [seq_length+2, embed_dim]
        
        # Add positional embeddings
        total_len = full_seq.shape[0]
        full_seq = full_seq + positional_embedding[:total_len]
        
        # Add batch dimension
        full_seq = full_seq.unsqueeze(0)  # [1, seq_length+2, embed_dim]
        
        # Create attention mask (all 1s = attend to all)
        attention_mask = torch.ones(1, total_len, dtype=torch.bool, device=full_seq.device)
        
        return full_seq, attention_mask


class SAM3SoftPromptWrapper(nn.Module):
    """
    Wrapper around SAM3 that allows soft prompt injection.
    
    This modifies the forward pass to use soft embeddings instead
    of discrete token embeddings.
    """
    
    def __init__(
        self,
        sam3_model: nn.Module,
        soft_prompt: SoftPromptModule,
    ):
        super().__init__()
        
        self.sam3 = sam3_model
        self.soft_prompt = soft_prompt
        
        # Freeze SAM3 parameters
        for param in self.sam3.parameters():
            param.requires_grad = False
            
        # Keep soft prompt trainable
        for param in self.soft_prompt.parameters():
            param.requires_grad = True
    
    def get_text_encoder_components(self):
        """Get necessary components from SAM3's text encoder."""
        text_encoder = self.sam3.backbone.language_backbone
        encoder = text_encoder.encoder
        
        return {
            'token_embedding': encoder.token_embedding,
            'positional_embedding': encoder.positional_embedding,
            'transformer': encoder.transformer,
            'ln_final': encoder.ln_final,
            'resizer': text_encoder.resizer,
            'attn_mask': encoder.attn_mask,
            'sot_id': text_encoder.tokenizer.sot_token_id,
            'eot_id': text_encoder.tokenizer.eot_token_id,
        }
    
    def forward_text_with_soft_prompt(
        self,
        device: str = 'mps',
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through text encoder using soft prompts.
        
        Returns dictionary compatible with SAM3's expected format.
        """
        components = self.get_text_encoder_components()
        
        # Get SOT and EOT embeddings
        sot_id = torch.tensor([components['sot_id']], device=device)
        eot_id = torch.tensor([components['eot_id']], device=device)
        
        sot_embed = components['token_embedding'](sot_id).squeeze(0)
        eot_embed = components['token_embedding'](eot_id).squeeze(0)
        pos_embed = components['positional_embedding'].to(device)
        
        # Get soft embeddings with special tokens
        embeddings, attn_mask = self.soft_prompt.get_with_special_tokens(
            sot_embed, eot_embed, pos_embed
        )
        
        # Pass through transformer
        seq_len = embeddings.shape[1]
        causal_mask = components['attn_mask']
        if causal_mask is not None:
            causal_mask = causal_mask[:seq_len, :seq_len].to(device)
        
        x = components['transformer'](embeddings, attn_mask=causal_mask)
        x = components['ln_final'](x)
        
        # Resize to model dimension
        text_memory = components['resizer'](x)  # [1, seq_len, d_model]
        
        # Transpose for SAM3 format (seq_first)
        text_memory = text_memory.transpose(0, 1)  # [seq_len, 1, d_model]
        
        # Create attention mask in SAM3 format (True = masked/ignored)
        # For soft prompts, we want to attend to all tokens
        text_attention_mask = ~attn_mask  # Invert: False = attend
        
        return {
            'language_features': text_memory,
            'language_mask': text_attention_mask,
            'language_embeds': embeddings.transpose(0, 1),  # For compatibility
        }
    
    def forward(
        self,
        images: torch.Tensor,
        return_masks: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with soft prompts.
        
        Args:
            images: [B, C, H, W] input images
            return_masks: Whether to compute masks
            
        Returns:
            Dictionary with predictions
        """
        device = images.device
        
        # Get image features
        backbone_out = self.sam3.backbone.forward_image(images)
        
        # Get soft prompt text features
        text_out = self.forward_text_with_soft_prompt(device=str(device))
        backbone_out.update(text_out)
        
        # Create dummy find_input for grounding
        from sam3.model.data_misc import FindStage
        
        batch_size = images.shape[0]
        find_input = FindStage(
            img_ids=torch.arange(batch_size, device=device),
            text_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        
        # Get dummy geometric prompt
        geometric_prompt = self.sam3._get_dummy_prompt(num_prompts=batch_size)
        
        # Run grounding
        out = self.sam3.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )
        
        return out


def optimize_soft_prompt(
    model: nn.Module,
    dataloader,
    init_embeddings: Optional[torch.Tensor] = None,
    config: Config = None,
    device: str = 'mps',
    embed_dim: int = 1024,
    vocab_embeddings: VocabularyEmbeddings = None,  # ADD THIS
) -> Tuple[SoftPromptModule, Dict]:
    """
    Optimize soft prompt embeddings using gradient descent.
    
    Args:
        model: SAM3 model (will be frozen)
        dataloader: DataLoader yielding images and masks
        init_embeddings: Initial soft prompt embeddings
        config: Configuration
        device: Device to use
        
    Returns:
        optimized_prompt: Optimized SoftPromptModule
        history: Optimization history
    """
    if config is None:
        config = Config()
    
    opt_config = config.optimization
    
    # Get text encoder dimensions
    #COMMENT OUT FOR PATCHED APPROACH
    #text_encoder = model.backbone.language_backbone
    #embed_dim = text_encoder.encoder.width
    
    # Create soft prompt module
    soft_prompt = SoftPromptModule(
        seq_length=opt_config.soft_prompt_length,
        embed_dim=embed_dim,
        init_embeddings=init_embeddings,
    ).to(device)
    
    # Create wrapper
    wrapper = SAM3SoftPromptWrapper(model, soft_prompt)
    wrapper.to(device)
    
    # Optimizer
    optimizer = AdamW(
        soft_prompt.parameters(),
        lr=opt_config.soft_lr,
        weight_decay=0.01,
    )
    
    # 15% warmup, linear decay to 2/3 LR through 65%, then cosine decay
    warmup_pct = 0.15
    linear_end_pct = 0.65
    linear_end_lr = 2/3  # Fraction of peak LR at end of linear phase
    
    warmup_steps = int(opt_config.soft_steps * warmup_pct)
    linear_end_step = int(opt_config.soft_steps * linear_end_pct)
    decay_steps = opt_config.soft_steps - linear_end_step
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Warmup: 0 -> 1
            return step / warmup_steps
        elif step < linear_end_step:
            # Linear decay: 1 -> 2/3
            progress = (step - warmup_steps) / (linear_end_step - warmup_steps)
            return 1.0 - progress * (1.0 - linear_end_lr)
        else:
            # Cosine decay: 2/3 -> 0
            progress = (step - linear_end_step) / decay_steps
            return linear_end_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
                
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    history = {
        'loss': [],
        'task_loss': [],
        'lr': [],
    }
    
    best_loss = float('inf')
    best_embeddings = None
    
    for step in tqdm(range(opt_config.soft_steps), desc="Soft prompt optimization"):
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            images = batch['images'].to(device)
            gt_masks = [m.to(device) for m in batch['masks']]
            
            optimizer.zero_grad()
            
            # Forward pass
            try:
                out = wrapper(images)
                pred_masks = out.get('pred_masks')
                
                if pred_masks is None:
                    continue
                
                # Compute task loss
                batch_loss = 0.0
                size_penalty = 0.0
                for b in range(len(gt_masks)):
                    if gt_masks[b].shape[0] == 0:
                        continue
                    
                    pred_b = pred_masks[b]
                    gt_b = gt_masks[b]
                    
                    if pred_b.shape[-2:] != gt_b.shape[-2:]:
                        pred_b = F.interpolate(
                            pred_b.unsqueeze(0),
                            size=gt_b.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                    
                    loss_b = soft_iou_loss(pred_b, gt_b)
                    batch_loss = batch_loss + loss_b
                    
                    # Size penalty: penalize masks that are too large relative to GT
                    # This discourages "segment everything" solutions
                    pred_sizes = pred_b.sigmoid().sum(dim=(-1, -2))  # [num_queries]
                    gt_avg_size = gt_b.float().sum(dim=(-1, -2)).mean()  # scalar
                    max_allowed_size = 2.0 * gt_avg_size
                    
                    # Penalize predictions exceeding 2x average GT size
                    size_excess = F.relu(pred_sizes - max_allowed_size)
                    size_penalty = size_penalty + size_excess.mean()
                
                if batch_loss > 0:
                    task_loss = batch_loss / len(gt_masks)
                    size_loss = 0.1 * size_penalty / len(gt_masks)  # Weight size penalty
                    
                    # Proximity regularization: keep embeddings near valid tokens
                    # This prevents drift into meaningless embedding space
                    soft_emb = soft_prompt.soft_embeddings
                    soft_norm = F.normalize(soft_emb, dim=-1)
                    
                    # We don't have vocab embeddings in this scope, so skip for now
                    # TODO: Pass vocab_embeddings to enable proximity loss
                    total_loss = task_loss + size_loss
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(soft_prompt.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    epoch_task_loss += task_loss.item()
                    num_batches += 1
                    
            except Exception as e:
                print(f"Error in forward pass: {e}")
                continue
        
        scheduler.step()
        
        # Record history
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_task_loss = epoch_task_loss / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        
        history['loss'].append(avg_loss)
        history['task_loss'].append(avg_task_loss)
        history['lr'].append(current_lr)
        
        # Track best
        if avg_task_loss < best_loss:
            best_loss = avg_task_loss
            best_embeddings = soft_prompt.soft_embeddings.detach().clone()
        
        if step % config.log_every == 0:
            # Project current embeddings to tokens
            from soft_prompt import project_to_discrete_tokens
            curr_tokens, curr_sims = project_to_discrete_tokens(
                soft_prompt.soft_embeddings.detach().cpu(),
                vocab_embeddings,  # Need to pass this in or access it
            )
            
            # Track token changes from initial
            if 'init_tokens' not in history:
                history['init_tokens'] = curr_tokens
            
            tokens_changed = sum(1 for a, b in zip(history['init_tokens'], curr_tokens) if a != b)
            
            # Embedding movement
            if 'init_embeddings' not in history:
                history['init_embeddings'] = soft_prompt.soft_embeddings.detach().clone()
            
            movement = (soft_prompt.soft_embeddings.detach() - history['init_embeddings'].to(soft_prompt.soft_embeddings.device)).norm().item()
            
            print(f"Step {step}: Loss={avg_task_loss:.4f}, LR={current_lr:.6f}")
            print(f"  Tokens: {curr_tokens} ({tokens_changed}/5 changed)")
            print(f"  Sims: {[f'{s:.3f}' for s in curr_sims.tolist()]}")
            print(f"  Movement: {movement:.2f}")
    
    # Restore best embeddings
    if best_embeddings is not None:
        soft_prompt.soft_embeddings.data = best_embeddings
    
    return soft_prompt, history


def project_to_discrete_tokens(
    soft_embeddings: torch.Tensor,  # [seq_len, embed_dim]
    vocab_embeddings,  # VocabularyEmbeddings
    exclude_special: bool = True,
) -> Tuple[List[int], torch.Tensor]:
    """
    Project soft embeddings to nearest discrete tokens.
    
    Args:
        soft_embeddings: Optimized soft prompt embeddings
        vocab_embeddings: Pre-computed vocabulary embeddings
        exclude_special: Whether to exclude special tokens
        
    Returns:
        token_ids: List of projected token IDs
        similarities: Similarity scores for each projection
    """
    exclude_ids = [49406, 49407] if exclude_special else []  # SOT, EOT
    
    token_ids = []
    similarities = []
    
    for i in range(soft_embeddings.shape[0]):
        emb = soft_embeddings[i]
        nearest_ids, nearest_sims = vocab_embeddings.find_nearest(
            emb, k=1, exclude_ids=exclude_ids
        )
        token_ids.append(nearest_ids[0].item())
        similarities.append(nearest_sims[0].item())
    
    return token_ids, torch.tensor(similarities)


def evaluate_token_sequence(
    model: nn.Module,
    token_ids: List[int],
    dataloader,
    device: str = 'mps',
    confidence_threshold: float = 0.5,
) -> MetricResult:
    """
    Evaluate a discrete token sequence on a dataset.
    
    Args:
        model: SAM3 model
        token_ids: Token sequence to evaluate
        dataloader: DataLoader yielding images and masks
        device: Device to use
        confidence_threshold: Threshold for predictions
        
    Returns:
        MetricResult with evaluation metrics
    """
    model.eval()
    
    # Get tokenizer
    tokenizer = model.backbone.language_backbone.tokenizer
    
    # Decode tokens to string (for SAM3's expected input)
    # This is a bit roundabout but maintains compatibility
    text_string = tokenizer.decode(token_ids)
    
    all_pred_masks = []
    all_gt_masks = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            gt_masks = batch['masks']
            
            # Use SAM3's normal inference path
            backbone_out = model.backbone.forward_image(images)
            text_out = model.backbone.forward_text([text_string], device=device)
            backbone_out.update(text_out)
            
            batch_size = images.shape[0]
            
            from sam3.model.data_misc import FindStage
            find_input = FindStage(
                img_ids=torch.arange(batch_size, device=device),
                text_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            
            geometric_prompt = model._get_dummy_prompt(num_prompts=batch_size)
            
            out = model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=None,
                geometric_prompt=geometric_prompt,
            )
            
            pred_masks = out.get('pred_masks')
            pred_logits = out.get('pred_logits')
            
            if pred_masks is not None and pred_logits is not None:
                # Filter by confidence
                probs = pred_logits.sigmoid().squeeze(-1)  # [B, num_queries]
                
                for b in range(batch_size):
                    keep = probs[b] > confidence_threshold
                    pred_b = pred_masks[b][keep]  # [N_keep, H, W] - these are logits!
                    gt_b = gt_masks[b].to(device)  # [M, H, W]
                    
                    if pred_b.shape[0] > 0:
                        # Apply sigmoid to convert logits to probabilities
                        pred_b = pred_b.sigmoid()
                        
                        # Resize predictions if needed
                        if pred_b.shape[-2:] != gt_b.shape[-2:]:
                            pred_b = F.interpolate(
                                pred_b.unsqueeze(0).float(),
                                size=gt_b.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0)
                        
                        # Threshold to binary
                        pred_b = pred_b > 0.5
                    
                    all_pred_masks.append(pred_b)
                    all_gt_masks.append(gt_b)
    
    # Compute metrics
    metrics = compute_metrics_batch(all_pred_masks, all_gt_masks)
    
    return metrics


def evaluate_token_sequence_on_batches(
    model: nn.Module,
    token_ids: List[int],
    batches: List[dict],  # Pre-sampled batches for speed
    device: str = 'mps',
    confidence_threshold: float = 0.5,
) -> MetricResult:
    """
    Evaluate a discrete token sequence on pre-sampled batches.
    
    This is a faster version that avoids re-iterating the dataloader
    and only uses a small sample of images for evaluation.
    
    Args:
        model: SAM3 model
        token_ids: Token sequence to evaluate
        batches: Pre-sampled list of batch dicts
        device: Device to use
        confidence_threshold: Threshold for predictions
        
    Returns:
        MetricResult with evaluation metrics
    """
    model.eval()
    
    # Get tokenizer
    tokenizer = model.backbone.language_backbone.tokenizer
    
    # Decode tokens to string
    text_string = tokenizer.decode(token_ids)
    
    all_pred_masks = []
    all_gt_masks = []
    
    with torch.no_grad():
        for batch in batches:
            images = batch['images'].to(device)
            gt_masks = batch['masks']
            
            # Use SAM3's normal inference path
            backbone_out = model.backbone.forward_image(images)
            text_out = model.backbone.forward_text([text_string], device=device)
            backbone_out.update(text_out)
            
            batch_size = images.shape[0]
            
            from sam3.model.data_misc import FindStage
            find_input = FindStage(
                img_ids=torch.arange(batch_size, device=device),
                text_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            
            geometric_prompt = model._get_dummy_prompt(num_prompts=batch_size)
            
            out = model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=None,
                geometric_prompt=geometric_prompt,
            )
            
            pred_masks = out.get('pred_masks')
            pred_logits = out.get('pred_logits')
            
            if pred_masks is not None and pred_logits is not None:
                probs = pred_logits.sigmoid().squeeze(-1)
                
                for b in range(batch_size):
                    keep = probs[b] > confidence_threshold
                    pred_b = pred_masks[b][keep]  # logits
                    gt_b = gt_masks[b].to(device)
                    
                    if pred_b.shape[0] > 0:
                        # Apply sigmoid to convert logits to probabilities
                        pred_b = pred_b.sigmoid()
                        
                        # Resize if needed
                        if pred_b.shape[-2:] != gt_b.shape[-2:]:
                            pred_b = F.interpolate(
                                pred_b.unsqueeze(0).float(),
                                size=gt_b.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0)
                        
                        # Threshold to binary
                        pred_b = pred_b > 0.5
                    
                    all_pred_masks.append(pred_b)
                    all_gt_masks.append(gt_b)
    
    metrics = compute_metrics_batch(all_pred_masks, all_gt_masks)
    return metrics


###DEPRECATED 256 Exploration
# class SoftPromptModule256(nn.Module):
#     """
#     Learnable soft prompt embeddings in 256-dim post-resizer space.
    
#     These embeddings are injected directly into language_features,
#     bypassing the text transformer.
#     """
    
#     def __init__(
#         self,
#         seq_length: int,
#         embed_dim: int = 256,
#         init_embeddings: Optional[torch.Tensor] = None,
#     ):
#         super().__init__()
        
#         self.seq_length = seq_length
#         self.embed_dim = embed_dim
        
#         if init_embeddings is not None:
#             assert init_embeddings.shape == (seq_length, embed_dim), \
#                 f"Expected ({seq_length}, {embed_dim}), got {init_embeddings.shape}"
#             self.soft_embeddings = nn.Parameter(init_embeddings.clone())
#         else:
#             self.soft_embeddings = nn.Parameter(
#                 torch.randn(seq_length, embed_dim) * 0.02
#             )
    
#     def forward(self) -> torch.Tensor:
#         """Return current soft embeddings [seq_length, embed_dim]."""
#         return self.soft_embeddings


# class SAM3SoftPromptWrapper256(nn.Module):
#     """
#     Wrapper that injects 256-dim soft prompts directly into language_features.
    
#     Bypasses the text transformer and resizer - works in the post-resizer
#     space where vision-language alignment happens.
#     """
    
#     def __init__(
#         self,
#         sam3_model: nn.Module,
#         soft_prompt: SoftPromptModule256,
#     ):
#         super().__init__()
        
#         self.sam3 = sam3_model
#         self.soft_prompt = soft_prompt
        
#         # Freeze SAM3 parameters
#         for param in self.sam3.parameters():
#             param.requires_grad = False
            
#         # Keep soft prompt trainable
#         for param in self.soft_prompt.parameters():
#             param.requires_grad = True
    
#     def forward(
#         self,
#         images: torch.Tensor,
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Forward pass with soft prompts injected at 256-dim level.
#         """
#         device = images.device
#         batch_size = images.shape[0]
        
#         # Get image features
#         backbone_out = self.sam3.backbone.forward_image(images)
        
#         # Build language_features from soft prompt
#         # Shape expected: [seq_len, batch, 256]
#         soft_emb = self.soft_prompt()  # [seq_len, 256]
        
#         # Add batch dimension and transpose to [seq_len, batch, 256]
#         language_features = soft_emb.unsqueeze(1).expand(-1, batch_size, -1)
        
#         # Create attention mask (False = attend, True = ignore)
#         # All tokens are valid
#         language_mask = torch.zeros(
#             batch_size, self.soft_prompt.seq_length, 
#             dtype=torch.bool, device=device
#         )
        
#         # Inject into backbone_out
#         backbone_out['language_features'] = language_features
#         backbone_out['language_mask'] = language_mask
        
#         # Create find_input for grounding
#         from sam3.model.data_misc import FindStage
        
#         find_input = FindStage(
#             img_ids=torch.arange(batch_size, device=device),
#             text_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
#             input_boxes=None,
#             input_boxes_mask=None,
#             input_boxes_label=None,
#             input_points=None,
#             input_points_mask=None,
#         )
        
#         # Get dummy geometric prompt
#         geometric_prompt = self.sam3._get_dummy_prompt(num_prompts=batch_size)
        
#         # Run grounding
#         out = self.sam3.forward_grounding(
#             backbone_out=backbone_out,
#             find_input=find_input,
#             find_target=None,
#             geometric_prompt=geometric_prompt,
#         )
        
#         return out


# def optimize_soft_prompt_256(
#     model: nn.Module,
#     dataloader,
#     init_embeddings: torch.Tensor,  # [seq_len, 256] - required
#     config: Config = None,
#     device: str = 'mps',
# ) -> Tuple[SoftPromptModule256, Dict]:
#     """
#     Optimize soft prompt embeddings in 256-dim post-resizer space.
#     """
#     if config is None:
#         config = Config()
    
#     opt_config = config.optimization
#     seq_length = init_embeddings.shape[0]
#     embed_dim = init_embeddings.shape[1]
    
#     # Create soft prompt module
#     soft_prompt = SoftPromptModule256(
#         seq_length=seq_length,
#         embed_dim=embed_dim,
#         init_embeddings=init_embeddings,
#     ).to(device)
    
#     # Create wrapper
#     wrapper = SAM3SoftPromptWrapper256(model, soft_prompt)
#     wrapper.to(device)
    
#     # Optimizer
#     optimizer = AdamW(
#         soft_prompt.parameters(),
#         lr=opt_config.soft_lr,
#         weight_decay=0.01,
#     )
    
#     # Warmup + Cosine scheduler
#     warmup_steps = int(opt_config.soft_steps * 0.1)
    
#     def lr_lambda(step):
#         if step < warmup_steps:
#             return step / warmup_steps
#         else:
#             progress = (step - warmup_steps) / (opt_config.soft_steps - warmup_steps)
#             return 0.5 * (1 + np.cos(np.pi * progress))
    
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
#     # Training loop
#     history = {
#         'loss': [],
#         'task_loss': [],
#         'lr': [],
#     }
    
#     best_loss = float('inf')
#     best_embeddings = None
    
#     for step in tqdm(range(opt_config.soft_steps), desc="Soft prompt optimization (256d)"):
#         epoch_loss = 0.0
#         epoch_task_loss = 0.0
#         num_batches = 0
        
#         for batch in dataloader:
#             images = batch['images'].to(device)
#             gt_masks = [m.to(device) for m in batch['masks']]
            
#             optimizer.zero_grad()
            
#             try:
#                 out = wrapper(images)
#                 pred_masks = out.get('pred_masks')
                
#                 if pred_masks is None:
#                     continue
                
#                 # Compute task loss
#                 batch_loss = 0.0
#                 size_penalty = 0.0
                
#                 for b in range(len(gt_masks)):
#                     if gt_masks[b].shape[0] == 0:
#                         continue
                    
#                     pred_b = pred_masks[b]
#                     gt_b = gt_masks[b]
                    
#                     if pred_b.shape[-2:] != gt_b.shape[-2:]:
#                         pred_b = F.interpolate(
#                             pred_b.unsqueeze(0),
#                             size=gt_b.shape[-2:],
#                             mode='bilinear',
#                             align_corners=False
#                         ).squeeze(0)
                    
#                     loss_b = soft_iou_loss(pred_b, gt_b)
#                     batch_loss = batch_loss + loss_b
                    
#                     # Size penalty
#                     pred_sizes = pred_b.sigmoid().sum(dim=(-1, -2))
#                     gt_avg_size = gt_b.float().sum(dim=(-1, -2)).mean()
#                     max_allowed_size = 2.0 * gt_avg_size
#                     size_excess = F.relu(pred_sizes - max_allowed_size)
#                     size_penalty = size_penalty + size_excess.mean()
                
#                 if batch_loss > 0:
#                     task_loss = batch_loss / len(gt_masks)
#                     size_loss = 0.1 * size_penalty / len(gt_masks)
#                     total_loss = task_loss + size_loss
                    
#                     total_loss.backward()
#                     torch.nn.utils.clip_grad_norm_(soft_prompt.parameters(), 1.0)
#                     optimizer.step()
                    
#                     epoch_loss += total_loss.item()
#                     epoch_task_loss += task_loss.item()
#                     num_batches += 1
                    
#             except Exception as e:
#                 print(f"Error in forward pass: {e}")
#                 continue
        
#         scheduler.step()
        
#         avg_loss = epoch_loss / max(num_batches, 1)
#         avg_task_loss = epoch_task_loss / max(num_batches, 1)
#         current_lr = scheduler.get_last_lr()[0]
        
#         history['loss'].append(avg_loss)
#         history['task_loss'].append(avg_task_loss)
#         history['lr'].append(current_lr)
        
#         if avg_task_loss < best_loss and avg_task_loss > 0:
#             best_loss = avg_task_loss
#             best_embeddings = soft_prompt.soft_embeddings.detach().clone()
        
#         if step % config.log_every == 0:
#             print(f"Step {step}: Loss = {avg_task_loss:.4f}, LR = {current_lr:.6f}")
    
#     # Restore best embeddings
#     if best_embeddings is not None:
#         soft_prompt.soft_embeddings.data = best_embeddings
    
#     return soft_prompt, history