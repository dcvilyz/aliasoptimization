"""
Main Alias Optimizer - Ties all components together.
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from config import Config, get_config
from data_loader import COCODataset, CategoryData, get_category_dataloader
from embeddings import (
    build_vocabulary_embeddings,
    extract_concept_from_category,
    initialize_soft_prompt_from_visual,
    VocabularyEmbeddings,
)
from soft_prompt import (
    SoftPromptModule,
    optimize_soft_prompt,
    project_to_discrete_tokens,
    evaluate_token_sequence,
)
from discrete_search import DiscreteSearchOptimizer, create_evaluate_fn
from metrics import MetricResult, compute_combined_score


@dataclass
class OptimizationResult:
    """Result of alias optimization for a single category."""
    category_id: int
    category_name: str
    
    # Best token sequence found
    best_tokens: List[int]
    best_decoded: str  # Decoded to string (may be gibberish)
    
    # Metrics
    best_fitness: float
    best_metrics: Dict
    
    # Baseline metrics (if available)
    baseline_tokens: Optional[List[int]] = None
    baseline_decoded: Optional[str] = None
    baseline_fitness: Optional[float] = None
    baseline_metrics: Optional[Dict] = None
    
    # Optimization history
    soft_prompt_history: Optional[Dict] = None
    discrete_search_history: Optional[Dict] = None
    
    # Timing
    total_time_seconds: float = 0.0
    
    def improvement_over_baseline(self) -> Optional[float]:
        """Calculate improvement over baseline if available."""
        if self.baseline_fitness is not None and self.baseline_fitness > 0:
            return (self.best_fitness - self.baseline_fitness) / self.baseline_fitness
        return None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'OptimizationResult':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class AliasOptimizer:
    """
    Main alias optimizer class.
    
    Orchestrates the full optimization pipeline:
    1. Extract visual concept embedding from masks
    2. Initialize soft prompts near visual concept
    3. Optimize soft prompts via gradient descent
    4. Project to discrete tokens
    5. Refine via discrete search
    """

    # Updated __init__ for AliasOptimizer to handle dual embeddings
    def __init__(
        self,
        model,  # SAM3 model
        config,  # Config
        vocab_embeddings_256=None,  # VocabularyEmbeddings (256-dim, post-resizer)
        vocab_embeddings_1024=None,  # VocabularyEmbeddings (1024-dim, pre-transformer)
    ):
        """
        Initialize AliasOptimizer with dual embedding support.
        
        Args:
            model: SAM3 model
            config: Config object
            vocab_embeddings_256: Pre-computed 256-dim embeddings (for visual matching)
            vocab_embeddings_1024: Pre-computed 1024-dim embeddings (for soft prompt optimization)
        """
        self.model = model
        self.config = config
        self.device = config.device
        
        # Store both embedding spaces
        self.vocab_embeddings_256 = vocab_embeddings_256
        self.vocab_embeddings_1024 = vocab_embeddings_1024
        
        # For backward compatibility, also set vocab_embeddings to 1024
        self.vocab_embeddings = vocab_embeddings_1024
        
        # Get tokenizer for decoding
        self.tokenizer = model.backbone.language_backbone.tokenizer
        
        print(f"AliasOptimizer initialized with dual embeddings:")
        if vocab_embeddings_256:
            print(f"  256-dim: {vocab_embeddings_256.vocab_size} tokens")
        if vocab_embeddings_1024:
            print(f"  1024-dim: {vocab_embeddings_1024.vocab_size} tokens")
    # def __init__(
    #     self,
    #     model,  # SAM3 model
    #     config: Config,
    #     vocab_embeddings: Optional[VocabularyEmbeddings] = None,
    # ):
    #     self.model = model
    #     self.config = config
    #     self.device = config.device
        
    #     # Build or load vocabulary embeddings
    #     if vocab_embeddings is not None:
    #         self.vocab_embeddings = vocab_embeddings
    #     else:
    #         self.vocab_embeddings = build_vocabulary_embeddings(
    #             model=model,
    #             config=config.embedding,
    #             device=self.device,
    #         )
        
    #     # Get tokenizer for decoding
    #     self.tokenizer = model.backbone.language_backbone.tokenizer
        
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs to string."""
        return self.tokenizer.decode(token_ids)
    
    def encode_text(self, text: str) -> List[int]:
        """Encode text to token IDs (without SOT/EOT)."""
        return self.tokenizer.encode(text)
    
    def optimize_category(
        self,
        category_data: CategoryData,
        baseline_text: Optional[str] = None,
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        Optimize alias for a single category.
        
        Args:
            category_data: CategoryData with images and masks
            baseline_text: Optional baseline text to compare against (e.g., class name)
            verbose: Whether to print progress
            
        Returns:
            OptimizationResult with best found alias
        """
        import time
        start_time = time.time()
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Optimizing alias for: {category_data.category_name}")
            print(f"  Images: {category_data.num_images}")
            print(f"  Instances: {category_data.num_instances}")
            print(f"{'=' * 60}\n")
        
        # Create dataloader
        dataloader = get_category_dataloader(
            category_data=category_data,
            batch_size=self.config.optimization.soft_batch_size,
            image_size=self.config.data.image_size,
            shuffle=False,
        )
        
        # ==================== Phase 1: Visual Concept Extraction ====================
        if verbose:
            print("[Phase 1] Extracting visual concept embedding...")
        
        concept_embedding = extract_concept_from_category(
            model=self.model,
            dataloader=dataloader,
            pool_type=self.config.embedding.pool_type,
            device=self.device,
        )
        
        if verbose:
            print(f"  Concept embedding shape: {concept_embedding.shape}")
        
        # ==================== Phase 2: Initialize Soft Prompt ====================
        if verbose:
            print("\n[Phase 2] Initializing soft prompt from visual concept...")
        
        init_embeddings = initialize_soft_prompt_from_visual(
            visual_embedding=concept_embedding,
            vocab_embeddings=self.vocab_embeddings,
            seq_length=self.config.optimization.soft_prompt_length,
            noise_scale=0.1,
        ).to(self.device)
        
        # Project initial embeddings to see what tokens we start with
        init_tokens, init_sims = project_to_discrete_tokens(
            init_embeddings.cpu(),
            self.vocab_embeddings,
        )
        if verbose:
            print(f"  Initial tokens: {init_tokens}")
            print(f"  Initial decoded: '{self.decode_tokens(init_tokens)}'")
        
        # ==================== Phase 3: Soft Prompt Optimization ====================
        if verbose:
            print("\n[Phase 3] Optimizing soft prompt via gradient descent...")
        
        optimized_soft_prompt, soft_history = optimize_soft_prompt(
            model=self.model,
            dataloader=dataloader,
            init_embeddings=init_embeddings,
            config=self.config,
            device=self.device,
            embed_dim=self.vocab_embeddings.embed_dim,  # Add this
        )
        # Project optimized soft prompt to discrete tokens
        projected_tokens, projected_sims = project_to_discrete_tokens(
            optimized_soft_prompt.soft_embeddings.detach().cpu(),
            self.vocab_embeddings,
        )
        
        if verbose:
            print(f"  Projected tokens: {projected_tokens}")
            print(f"  Projected decoded: '{self.decode_tokens(projected_tokens)}'")
            print(f"  Projection similarities: {projected_sims.tolist()}")
        
        # ==================== Phase 4: Discrete Search Refinement ====================
        if verbose:
            print("\n[Phase 4] Refining via discrete search...")
        
        # Create evaluation function
        eval_fn = create_evaluate_fn(
            model=self.model,
            dataloader=dataloader,
            device=self.device,
            confidence_threshold=self.config.optimization.confidence_threshold,
        )
        
        # Create discrete search optimizer
        discrete_optimizer = DiscreteSearchOptimizer(
            vocab_embeddings=self.vocab_embeddings,
            evaluate_fn=eval_fn,
            config=self.config,
            special_token_ids=[self.tokenizer.sot_token_id, self.tokenizer.eot_token_id],
        )
        
        # Run combined optimization
        best_candidate = discrete_optimizer.optimize(
            seed_tokens=projected_tokens,
            verbose=verbose,
        )
        
        # ==================== Baseline Comparison ====================
        baseline_fitness = None
        baseline_metrics = None
        baseline_tokens = None
        
        if baseline_text:
            if verbose:
                print(f"\n[Baseline] Evaluating baseline text: '{baseline_text}'")
            
            baseline_tokens = self.encode_text(baseline_text)
            baseline_fitness, baseline_metrics_obj = eval_fn(baseline_tokens)
            baseline_metrics = baseline_metrics_obj.to_dict()
            
            if verbose:
                print(f"  Baseline fitness: {baseline_fitness:.4f}")
                print(f"  Baseline metrics: {baseline_metrics_obj}")
        
        # ==================== Results ====================
        total_time = time.time() - start_time
        
        result = OptimizationResult(
            category_id=category_data.category_id,
            category_name=category_data.category_name,
            best_tokens=best_candidate.tokens,
            best_decoded=self.decode_tokens(best_candidate.tokens),
            best_fitness=best_candidate.fitness,
            best_metrics=best_candidate.metrics.to_dict() if best_candidate.metrics else {},
            baseline_tokens=baseline_tokens,
            baseline_decoded=baseline_text,
            baseline_fitness=baseline_fitness,
            baseline_metrics=baseline_metrics,
            soft_prompt_history=soft_history,
            discrete_search_history=discrete_optimizer.tracker.get_summary(),
            total_time_seconds=total_time,
        )
        
        if verbose:
            print(f"\n{'=' * 60}")
            print("OPTIMIZATION COMPLETE")
            print(f"{'=' * 60}")
            print(f"Best alias: '{result.best_decoded}'")
            print(f"Best tokens: {result.best_tokens}")
            print(f"Best fitness: {result.best_fitness:.4f}")
            if result.baseline_fitness is not None:
                improvement = result.improvement_over_baseline()
                print(f"Baseline fitness: {result.baseline_fitness:.4f}")
                print(f"Improvement: {improvement * 100:.1f}%" if improvement else "N/A")
            print(f"Time: {total_time:.1f}s")
            print(f"{'=' * 60}\n")
        
        return result
    
    def optimize_dataset(
        self,
        dataset: COCODataset,
        use_class_names_as_baseline: bool = True,
        verbose: bool = True,
    ) -> Dict[int, OptimizationResult]:
        """
        Optimize aliases for all categories in a dataset.
        
        Args:
            dataset: COCODataset to optimize
            use_class_names_as_baseline: Whether to use original class names as baseline
            verbose: Whether to print progress
            
        Returns:
            Dictionary mapping category_id to OptimizationResult
        """
        results = {}
        
        for i, cat_id in enumerate(dataset.category_ids):
            if verbose:
                print(f"\n[{i+1}/{len(dataset.category_ids)}] Processing category {cat_id}")
            
            category_data = dataset.get_category_data(cat_id, max_images=10)
            
            baseline_text = category_data.category_name if use_class_names_as_baseline else None
            
            result = self.optimize_category(
                category_data=category_data,
                baseline_text=baseline_text,
                verbose=verbose,
            )
            
            results[cat_id] = result
            
            # Save intermediate results
            output_path = Path(self.config.output_dir) / f"result_cat_{cat_id}.json"
            result.save(str(output_path))
        
        return results

    #####256 DETOUR DEPRECATED, produced learnings about visual, token space misalignment
    # def optimize_category_256(   
    #     self,
    #     category_data: CategoryData,
    #     baseline_text: Optional[str] = None,
    #     verbose: bool = True,
    # ) -> OptimizationResult:
    #     """
    #     Optimize alias using 256-dim post-resizer soft prompts.
    #     """
    #     import time
    #     from soft_prompt import optimize_soft_prompt_256, SoftPromptModule256, project_to_discrete_tokens
        
    #     start_time = time.time()
        
    #     if verbose:
    #         print(f"\n{'=' * 60}")
    #         print(f"Optimizing alias for: {category_data.category_name}")
    #         print(f"  Images: {category_data.num_images}")
    #         print(f"  Instances: {category_data.num_instances}")
    #         print(f"{'=' * 60}\n")
        
    #     # Create dataloader
    #     dataloader = get_category_dataloader(
    #         category_data=category_data,
    #         batch_size=self.config.optimization.soft_batch_size,
    #         image_size=self.config.data.image_size,
    #         shuffle=False,
    #     )
        
    #     # Phase 1: Extract visual concept embedding
    #     if verbose:
    #         print("[Phase 1] Extracting visual concept embedding...")
        
    #     concept_embedding = extract_concept_from_category(
    #         model=self.model,
    #         dataloader=dataloader,
    #         pool_type=self.config.embedding.pool_type,
    #         device=self.device,
    #     )
        
    #     if verbose:
    #         print(f"  Concept embedding shape: {concept_embedding.shape}")
        
    #     # Phase 2: Initialize soft prompt from visual concept
    #     if verbose:
    #         print("\n[Phase 2] Initializing soft prompt from visual concept...")
        
    #     init_embeddings = initialize_soft_prompt_from_visual(
    #         visual_embedding=concept_embedding,
    #         vocab_embeddings=self.vocab_embeddings,
    #         seq_length=self.config.optimization.soft_prompt_length,
    #         noise_scale=0.1,
    #     ).to(self.device)
        
    #     init_tokens, init_sims = project_to_discrete_tokens(
    #         init_embeddings.cpu(),
    #         self.vocab_embeddings,
    #     )
    #     if verbose:
    #         print(f"  Initial tokens: {init_tokens}")
    #         print(f"  Initial decoded: '{self.decode_tokens(init_tokens)}'")
        
    #     # Phase 3: Optimize soft prompt (256-dim)
    #     if verbose:
    #         print("\n[Phase 3] Optimizing soft prompt via gradient descent (256d)...")
        
    #     optimized_soft_prompt, soft_history = optimize_soft_prompt_256(
    #         model=self.model,
    #         dataloader=dataloader,
    #         init_embeddings=init_embeddings,
    #         config=self.config,
    #         device=self.device,
    #     )
        
    #     # Project to discrete tokens
    #     projected_tokens, projected_sims = project_to_discrete_tokens(
    #         optimized_soft_prompt.soft_embeddings.detach().cpu(),
    #         self.vocab_embeddings,
    #     )
        
    #     if verbose:
    #         print(f"  Projected tokens: {projected_tokens}")
    #         print(f"  Projected decoded: '{self.decode_tokens(projected_tokens)}'")
        
    #     # Phase 4: Discrete search refinement
    #     if verbose:
    #         print("\n[Phase 4] Refining via discrete search...")
        
    #     eval_fn = create_evaluate_fn(
    #         model=self.model,
    #         dataloader=dataloader,
    #         device=self.device,
    #         confidence_threshold=self.config.optimization.confidence_threshold,
    #     )
        
    #     discrete_optimizer = DiscreteSearchOptimizer(
    #         vocab_embeddings=self.vocab_embeddings,
    #         evaluate_fn=eval_fn,
    #         config=self.config,
    #         special_token_ids=[self.tokenizer.sot_token_id, self.tokenizer.eot_token_id],
    #     )
        
    #     best_candidate = discrete_optimizer.optimize(
    #         seed_tokens=projected_tokens,
    #         verbose=verbose,
    #     )
        
    #     # Baseline comparison
    #     baseline_fitness = None
    #     baseline_metrics = None
    #     baseline_tokens = None
        
    #     if baseline_text:
    #         if verbose:
    #             print(f"\n[Baseline] Evaluating baseline text: '{baseline_text}'")
            
    #         baseline_tokens = self.encode_text(baseline_text)
    #         baseline_fitness, baseline_metrics_obj = eval_fn(baseline_tokens)
    #         baseline_metrics = baseline_metrics_obj.to_dict()
            
    #         if verbose:
    #             print(f"  Baseline fitness: {baseline_fitness:.4f}")
        
    #     total_time = time.time() - start_time
        
    #     result = OptimizationResult(
    #         category_id=category_data.category_id,
    #         category_name=category_data.category_name,
    #         best_tokens=best_candidate.tokens,
    #         best_decoded=self.decode_tokens(best_candidate.tokens),
    #         best_fitness=best_candidate.fitness,
    #         best_metrics=best_candidate.metrics.to_dict() if best_candidate.metrics else {},
    #         baseline_tokens=baseline_tokens,
    #         baseline_decoded=baseline_text,
    #         baseline_fitness=baseline_fitness,
    #         baseline_metrics=baseline_metrics,
    #         soft_prompt_history=soft_history,
    #         discrete_search_history=discrete_optimizer.tracker.get_summary(),
    #         total_time_seconds=total_time,
    #     )
        
    #     if verbose:
    #         print(f"\n{'=' * 60}")
    #         print("OPTIMIZATION COMPLETE")
    #         print(f"{'=' * 60}")
    #         print(f"Best alias: '{result.best_decoded}'")
    #         print(f"Best fitness: {result.best_fitness:.4f}")
    #         if result.baseline_fitness is not None:
    #             improvement = result.improvement_over_baseline()
    #             print(f"Baseline fitness: {result.baseline_fitness:.4f}")
    #             print(f"Improvement: {improvement * 100:.1f}%" if improvement else "N/A")
    #         print(f"Time: {total_time:.1f}s")
        
    #     return result


        """
    Updated AliasOptimizer method with corrected initialization.
    
    Uses:
    - 256-dim embeddings for finding nearest tokens to visual features (aligned space)
    - 1024-dim embeddings for soft prompt optimization (through transformer)
    
    Add this method to the AliasOptimizer class in optimizer.py
    """
    
    def optimize_category_v2(
        self,
        category_data,  # CategoryData
        baseline_text: str = None,
        verbose: bool = True,
    ):
        """
        Optimize alias with corrected dual-embedding initialization.
        
        Key difference from original:
        - Uses 256-dim space to find nearest tokens to visual embedding
        - Initializes 1024-dim soft prompt with those tokens' 1024-dim embeddings
        - Optimizes in 1024-dim space through the transformer
        """
        import time
        from soft_prompt import optimize_soft_prompt, project_to_discrete_tokens
        from discrete_search import DiscreteSearchOptimizer, create_evaluate_fn
        from data_loader import get_category_dataloader
        from embeddings import extract_concept_from_category
        
        start_time = time.time()
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Optimizing alias for: {category_data.category_name}")
            print(f"  Images: {category_data.num_images}")
            print(f"  Instances: {category_data.num_instances}")
            print(f"{'=' * 60}\n")
        
        # Create dataloader
        dataloader = get_category_dataloader(
            category_data=category_data,
            batch_size=self.config.optimization.soft_batch_size,
            image_size=self.config.data.image_size,
            shuffle=False,
        )
        
        # ==================== Phase 1: Visual Concept Extraction ====================
        if verbose:
            print("[Phase 1] Extracting visual concept embedding...")
        
        # This extracts 256-dim visual features from masked regions
        concept_embedding = extract_concept_from_category(
            model=self.model,
            dataloader=dataloader,
            pool_type=self.config.embedding.pool_type,
            device=self.device,
        )
        
        if verbose:
            print(f"  Concept embedding shape: {concept_embedding.shape}")  # [256]
        
        # ==================== Phase 2: Initialize Soft Prompt ====================
        if verbose:
            print("\n[Phase 2] Initializing soft prompt from visual concept...")
            print("  Using 256-dim space for token matching, 1024-dim for optimization")
        
        # Use the corrected initialization function
        from embeddings import initialize_soft_prompt_from_visual_v2
        
        init_embeddings = initialize_soft_prompt_from_visual_v2(
            visual_embedding=concept_embedding,
            vocab_embeddings_256=self.vocab_embeddings_256,
            vocab_embeddings_1024=self.vocab_embeddings_1024,
            seq_length=self.config.optimization.soft_prompt_length,
            noise_scale=0.1,
        ).to(self.device)
        
        if verbose:
            print(f"  Soft prompt shape: {init_embeddings.shape}")  # [seq_len, 1024]
        
        # Project to see what tokens we start with (using 1024-dim space)
        init_tokens, init_sims = project_to_discrete_tokens(
            init_embeddings.cpu(),
            self.vocab_embeddings_1024,
        )
        if verbose:
            print(f"  Initial tokens: {init_tokens}")
            print(f"  Initial decoded: '{self.decode_tokens(init_tokens)}'")
            print(f"  Initial similarities (1024d): {[f'{s:.4f}' for s in init_sims.tolist()]}")
        
        # ==================== Phase 3: Soft Prompt Optimization ====================
        if verbose:
            print("\n[Phase 3] Optimizing soft prompt via gradient descent (1024d)...")
        
        optimized_soft_prompt, soft_history = optimize_soft_prompt(
            model=self.model,
            dataloader=dataloader,
            init_embeddings=init_embeddings,
            config=self.config,
            device=self.device,
            embed_dim=1024,
            vocab_embeddings=self.vocab_embeddings_1024,  # ADD THIS
        )
        
        # Project optimized soft prompt to discrete tokens
        projected_tokens, projected_sims = project_to_discrete_tokens(
            optimized_soft_prompt.soft_embeddings.detach().cpu(),
            self.vocab_embeddings_1024,
        )
        
        if verbose:
            print(f"  Projected tokens: {projected_tokens}")
            print(f"  Projected decoded: '{self.decode_tokens(projected_tokens)}'")
            print(f"  Projection similarities: {[f'{s:.4f}' for s in projected_sims.tolist()]}")
            
            # Check how many tokens changed
            tokens_changed = sum(1 for a, b in zip(init_tokens, projected_tokens) if a != b)
            print(f"  Tokens changed: {tokens_changed}/{len(init_tokens)}")
        
        # ==================== Phase 4: Discrete Search Refinement ====================
        if verbose:
            print("\n[Phase 4] Refining via discrete search...")
        
        # Create evaluation function
        eval_fn = create_evaluate_fn(
            model=self.model,
            dataloader=dataloader,
            device=self.device,
            confidence_threshold=self.config.optimization.confidence_threshold,
        )
        
        # Create discrete search optimizer (uses 1024-dim embeddings for similarity)
        discrete_optimizer = DiscreteSearchOptimizer(
            vocab_embeddings=self.vocab_embeddings_1024,
            evaluate_fn=eval_fn,
            config=self.config,
            special_token_ids=[self.tokenizer.sot_token_id, self.tokenizer.eot_token_id],
        )
        
        best_candidate = discrete_optimizer.optimize(
            seed_tokens=projected_tokens,
            verbose=verbose,
        )
        
        # ==================== Baseline Comparison ====================
        baseline_fitness = None
        baseline_metrics = None
        baseline_tokens = None
        
        if baseline_text:
            if verbose:
                print(f"\n[Baseline] Evaluating baseline text: '{baseline_text}'")
            
            baseline_tokens = self.encode_text(baseline_text)
            baseline_fitness, baseline_metrics_obj = eval_fn(baseline_tokens)
            baseline_metrics = baseline_metrics_obj.to_dict()
            
            if verbose:
                print(f"  Baseline fitness: {baseline_fitness:.4f}")
        
        total_time = time.time() - start_time
        
        # Build result
        from optimizer import OptimizationResult
        
        result = OptimizationResult(
            category_id=category_data.category_id,
            category_name=category_data.category_name,
            best_tokens=best_candidate.tokens,
            best_decoded=self.decode_tokens(best_candidate.tokens),
            best_fitness=best_candidate.fitness,
            best_metrics=best_candidate.metrics.to_dict() if best_candidate.metrics else {},
            baseline_tokens=baseline_tokens,
            baseline_decoded=baseline_text,
            baseline_fitness=baseline_fitness,
            baseline_metrics=baseline_metrics,
            soft_prompt_history=soft_history,
            discrete_search_history=discrete_optimizer.tracker.get_summary(),
            total_time_seconds=total_time,
        )
        
        if verbose:
            print(f"\n{'=' * 60}")
            print("OPTIMIZATION COMPLETE")
            print(f"{'=' * 60}")
            print(f"Best alias: '{result.best_decoded}'")
            print(f"Best fitness: {result.best_fitness:.4f}")
            if result.baseline_fitness is not None:
                improvement = result.improvement_over_baseline()
                print(f"Baseline fitness: {result.baseline_fitness:.4f}")
                print(f"Improvement: {improvement * 100:.1f}%" if improvement else "N/A")
            print(f"Time: {total_time:.1f}s")
        
        return result





def save_all_results(
    results: Dict[int, OptimizationResult],
    output_path: str,
):
    """Save all optimization results to a single JSON file."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'num_categories': len(results),
        'results': {str(k): v.to_dict() for k, v in results.items()},
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved results to: {output_path}")


def print_results_summary(results: Dict[int, OptimizationResult]):
    """Print summary of optimization results."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Category':<30} {'Best Alias':<20} {'Fitness':>10} {'Baseline':>10} {'Δ%':>8}")
    print("-" * 70)
    
    improvements = []
    
    for cat_id, result in results.items():
        cat_name = result.category_name[:28]
        alias = result.best_decoded[:18] if result.best_decoded else "N/A"
        fitness = f"{result.best_fitness:.4f}"
        baseline = f"{result.baseline_fitness:.4f}" if result.baseline_fitness else "N/A"
        
        improvement = result.improvement_over_baseline()
        if improvement is not None:
            improvements.append(improvement)
            delta = f"{improvement * 100:+.1f}%"
        else:
            delta = "N/A"
        
        print(f"{cat_name:<30} {alias:<20} {fitness:>10} {baseline:>10} {delta:>8}")
    
    print("-" * 70)
    
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"\nAverage improvement over baseline: {avg_improvement * 100:+.1f}%")
        print(f"Categories improved: {sum(1 for x in improvements if x > 0)}/{len(improvements)}")


# In optimizer.py, add this method to AliasOptimizer class:

