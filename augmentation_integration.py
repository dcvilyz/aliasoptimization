"""
Integration code for augmented evaluation.

USAGE IN NOTEBOOK:

1. Import augmentation:
   ```python
   from augmentation import AugmentationConfig, AugmentedDataLoader, create_augmented_dataloader
   ```

2. Update validate_single_concept to use augmented evaluation:
   ```python
   def validate_single_concept(
       model,
       optimizer: AliasOptimizer,
       concept: SaCoConceptData,
       config: Config,
       max_images: int = 10,
       verbose: bool = True,
       use_augmentation: bool = True,  # NEW PARAMETER
       aug_config: AugmentationConfig = None,  # NEW PARAMETER
   ) -> dict:
       # ... existing code ...
       
       # When creating dataloader for optimizer, pass augmentation config
       # This requires updating optimize_category_v2 to accept aug_config
   ```

3. Update run_random_baseline to use augmented evaluation:
   (see below)

"""

# =============================================================================
# UPDATED run_random_baseline with augmentation support
# =============================================================================

def run_random_baseline_augmented(
    model,
    concept,  # SaCoConceptData
    config,  # Config
    vocab_embeddings_1024,  # VocabularyEmbeddings
    num_random_seeds: int = 4,
    max_images: int = 10,
    seq_length: int = 5,
    use_augmentation: bool = True,
    aug_config = None,  # AugmentationConfig
    verbose: bool = True,
):
    """
    Run discrete search from random token initializations with augmented evaluation.
    No soft prompt optimization - just discrete search.
    """
    import random
    from discrete_search import DiscreteSearchOptimizer, create_evaluate_fn
    from saco_loader import get_concept_dataloader
    from augmentation import AugmentationConfig, AugmentedDataLoader
    
    text_input = concept.text_input
    tokenizer = model.backbone.language_backbone.tokenizer
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RANDOM BASELINE (augmented={use_augmentation}): {text_input}")
        print(f"{'='*60}")
    
    # Create category data and base dataloader
    category_data = create_category_data_from_concept(concept, max_images)
    
    from data_loader import get_category_dataloader
    base_loader = get_category_dataloader(
        category_data=category_data,
        batch_size=config.optimization.soft_batch_size,
        image_size=config.data.image_size,
        shuffle=False,
    )
    
    # Wrap with augmentation if enabled
    if use_augmentation:
        if aug_config is None:
            aug_config = AugmentationConfig(
                enabled=True,
                include_original=True,
                augmentation_multiplier=2,  # 2 augmented versions per image
            )
        dataloader = AugmentedDataLoader(base_loader, aug_config)
        if verbose:
            print(f"  Using augmentation: {aug_config.augmentation_multiplier}x + original")
    else:
        dataloader = base_loader
    
    # Create evaluation function with augmented loader
    eval_fn = create_evaluate_fn(
        model=model,
        dataloader=dataloader,
        device=config.device,
        confidence_threshold=config.optimization.confidence_threshold,
    )
    
    # Evaluate baseline text
    baseline_tokens = tokenizer.encode(text_input)
    baseline_fitness, baseline_metrics = eval_fn(baseline_tokens)
    
    if verbose:
        print(f"\nBaseline: '{text_input}'")
        print(f"  Fitness: {baseline_fitness:.4f}")
    
    # Special tokens to exclude
    special_tokens = [49406, 49407]  # SOT, EOT
    vocab_size = vocab_embeddings_1024.vocab_size
    
    results = []
    
    for seed_idx in range(num_random_seeds):
        if verbose:
            print(f"\n--- Random seed {seed_idx + 1}/{num_random_seeds} ---")
        
        # Generate random starting tokens
        random.seed(seed_idx * 1000 + 42)  # Reproducible
        random_tokens = []
        while len(random_tokens) < seq_length:
            t = random.randint(0, vocab_size - 1)
            if t not in special_tokens:
                random_tokens.append(t)
        
        if verbose:
            print(f"  Start: {random_tokens}")
            print(f"  Decoded: '{tokenizer.decode(random_tokens)}'")
        
        # Create discrete search optimizer
        discrete_optimizer = DiscreteSearchOptimizer(
            vocab_embeddings=vocab_embeddings_1024,
            evaluate_fn=eval_fn,
            config=config,
            special_token_ids=special_tokens,
        )
        
        # Run optimization
        best_candidate = discrete_optimizer.optimize(
            seed_tokens=random_tokens,
            verbose=verbose,
        )
        
        results.append({
            'seed_idx': seed_idx,
            'start_tokens': random_tokens,
            'start_decoded': tokenizer.decode(random_tokens),
            'best_tokens': best_candidate.tokens,
            'best_decoded': tokenizer.decode(best_candidate.tokens),
            'best_fitness': best_candidate.fitness,
        })
        
        if verbose:
            print(f"  Best: '{tokenizer.decode(best_candidate.tokens)}'")
            print(f"  Fitness: {best_candidate.fitness:.4f}")
    
    # Compute summary stats
    fitnesses = [r['best_fitness'] for r in results]
    mean_fitness = sum(fitnesses) / len(fitnesses)
    max_fitness = max(fitnesses)
    min_fitness = min(fitnesses)
    
    summary = {
        'text_input': text_input,
        'baseline_fitness': baseline_fitness,
        'num_seeds': num_random_seeds,
        'mean_fitness': mean_fitness,
        'max_fitness': max_fitness,
        'min_fitness': min_fitness,
        'std_fitness': (sum((f - mean_fitness)**2 for f in fitnesses) / len(fitnesses))**0.5,
        'augmentation_enabled': use_augmentation,
        'results': results,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RANDOM BASELINE SUMMARY: {text_input}")
        print(f"{'='*60}")
        print(f"  Baseline fitness: {baseline_fitness:.4f}")
        print(f"  Random mean:      {mean_fitness:.4f}")
        print(f"  Random max:       {max_fitness:.4f}")
        print(f"  Random min:       {min_fitness:.4f}")
        print(f"  Random std:       {summary['std_fitness']:.4f}")
    
    return summary


# =============================================================================
# Helper function used by both validation and ablation
# =============================================================================

def create_category_data_from_concept(concept, max_images: int = 10):
    """
    Convert SaCoConceptData to CategoryData format for optimizer compatibility.
    
    This is needed because AliasOptimizer.optimize_category_v2 expects CategoryData.
    """
    from data_loader import CategoryData, ImageAnnotation, InstanceMask
    
    annotations = []
    
    num_images = min(concept.num_positive_images, max_images)
    
    for i in range(num_images):
        img_path = concept.positive_image_paths[i]
        masks_np = concept.positive_masks[i]
        
        # Get image dimensions from first mask or default
        if masks_np:
            h, w = masks_np[0].shape
        else:
            h, w = 1008, 1008
        
        instances = []
        for j, mask in enumerate(masks_np):
            instances.append(InstanceMask(
                mask=mask,
                bbox=[0, 0, w, h],  # Dummy bbox
                area=float(mask.sum()),
                category_id=0,
                instance_id=j,
            ))
        
        annotations.append(ImageAnnotation(
            image_id=i,
            image_path=img_path,
            width=w,
            height=h,
            instances=instances,
        ))
    
    return CategoryData(
        category_id=0,
        category_name=concept.text_input,
        annotations=annotations,
    )


# =============================================================================
# UPDATED validate_single_concept with augmentation
# =============================================================================

def validate_single_concept_augmented(
    model,
    optimizer,  # AliasOptimizer
    concept,  # SaCoConceptData
    config,  # Config
    max_images: int = 10,
    use_augmentation: bool = True,
    aug_config = None,
    verbose: bool = True,
) -> dict:
    """
    Run validation for a single concept with optional augmentation.
    
    This is a drop-in replacement for validate_single_concept that adds
    augmentation support for more robust evaluation.
    """
    import time
    from augmentation import AugmentationConfig, AugmentedDataLoader
    from data_loader import get_category_dataloader
    from metrics import compute_combined_score
    
    text_input = concept.text_input
    
    if verbose:
        print(f"\n  Concept: '{text_input}'")
        print(f"  Images: {min(concept.num_positive_images, max_images)}, "
              f"Instances: {concept.num_instances}")
        print(f"  Augmentation: {use_augmentation}")
    
    # Convert to CategoryData
    category_data = create_category_data_from_concept(concept, max_images)
    
    # Create base dataloader
    base_loader = get_category_dataloader(
        category_data=category_data,
        batch_size=config.optimization.soft_batch_size,
        image_size=config.data.image_size,
        shuffle=False,
    )
    
    # Wrap with augmentation if enabled
    if use_augmentation:
        if aug_config is None:
            aug_config = AugmentationConfig(
                enabled=True,
                include_original=True,
                augmentation_multiplier=2,
            )
        eval_dataloader = AugmentedDataLoader(base_loader, aug_config)
    else:
        eval_dataloader = base_loader
    
    # Run optimization
    if verbose:
        print(f"\n  Running optimization...")
    
    start = time.time()
    
    # Note: optimize_category_v2 creates its own dataloader internally
    # We need to modify it to accept a pre-built augmented dataloader
    # For now, we run optimization without augmentation, then evaluate with augmentation
    
    opt_result = optimizer.optimize_category_v2(
        category_data=category_data,
        baseline_text=text_input,
        verbose=verbose,
    )
    
    opt_time = time.time() - start
    
    # Re-evaluate with augmented dataloader for fair comparison
    if use_augmentation:
        from discrete_search import create_evaluate_fn
        
        aug_eval_fn = create_evaluate_fn(
            model=model,
            dataloader=eval_dataloader,
            device=config.device,
            confidence_threshold=config.optimization.confidence_threshold,
        )
        
        # Re-evaluate baseline and optimized with augmentation
        tokenizer = model.backbone.language_backbone.tokenizer
        baseline_tokens = tokenizer.encode(text_input)
        
        baseline_fitness, baseline_metrics = aug_eval_fn(baseline_tokens)
        optimized_fitness, optimized_metrics = aug_eval_fn(opt_result.best_tokens)
        
        if verbose:
            print(f"\n  [Augmented Evaluation]")
            print(f"    Baseline: {baseline_fitness:.4f}")
            print(f"    Optimized: {optimized_fitness:.4f}")
    else:
        baseline_fitness = opt_result.baseline_fitness
        baseline_metrics = opt_result.baseline_metrics
        optimized_fitness = opt_result.best_fitness
        optimized_metrics = opt_result.best_metrics
    
    # Compute improvement
    improvement = optimized_fitness - baseline_fitness
    relative_imp = improvement / baseline_fitness if baseline_fitness > 0 else 0
    wins = optimized_fitness >= baseline_fitness
    
    return {
        'text_input': text_input,
        'num_images': min(concept.num_positive_images, max_images),
        'num_instances': sum(len(m) for m in concept.positive_masks[:max_images]),
        'baseline_fitness': baseline_fitness,
        'baseline_iou': baseline_metrics.get('mean_iou', 0) if isinstance(baseline_metrics, dict) else baseline_metrics.mean_iou,
        'optimized_fitness': optimized_fitness,
        'optimized_iou': optimized_metrics.get('mean_iou', 0) if isinstance(optimized_metrics, dict) else 0,
        'optimized_tokens': opt_result.best_tokens,
        'optimized_decoded': opt_result.best_decoded,
        'improvement': improvement,
        'relative_improvement': relative_imp,
        'optimized_wins': wins,
        'augmentation_enabled': use_augmentation,
        'time_seconds': opt_time,
    }


# =============================================================================
# Full ablation comparison with augmentation
# =============================================================================

def run_ablation_comparison(
    model,
    dataset,  # SaCoDataset
    concepts: list,  # List of concept names
    config,
    vocab_embeddings_1024,
    optimizer,  # AliasOptimizer (for soft prompt method)
    num_random_seeds: int = 4,
    max_images: int = 10,
    use_augmentation: bool = True,
    verbose: bool = True,
):
    """
    Run full ablation comparing soft prompt init vs random init.
    
    Returns dict with results for each concept.
    """
    from datetime import datetime
    
    results = {}
    
    for concept_name in concepts:
        print(f"\n{'='*70}")
        print(f"CONCEPT: {concept_name}")
        print(f"{'='*70}")
        
        concept = dataset[concept_name]
        
        # 1. Run with soft prompt initialization
        print("\n[SOFT PROMPT INIT]")
        soft_result = validate_single_concept_augmented(
            model=model,
            optimizer=optimizer,
            concept=concept,
            config=config,
            max_images=max_images,
            use_augmentation=use_augmentation,
            verbose=verbose,
        )
        
        # 2. Run with random initialization
        print("\n[RANDOM INIT]")
        random_result = run_random_baseline_augmented(
            model=model,
            concept=concept,
            config=config,
            vocab_embeddings_1024=vocab_embeddings_1024,
            num_random_seeds=num_random_seeds,
            max_images=max_images,
            use_augmentation=use_augmentation,
            verbose=verbose,
        )
        
        results[concept_name] = {
            'soft_prompt': soft_result,
            'random': random_result,
            'baseline_fitness': soft_result['baseline_fitness'],
            'soft_prompt_fitness': soft_result['optimized_fitness'],
            'random_mean_fitness': random_result['mean_fitness'],
            'random_max_fitness': random_result['max_fitness'],
            'soft_beats_random_mean': soft_result['optimized_fitness'] > random_result['mean_fitness'],
            'soft_beats_random_max': soft_result['optimized_fitness'] > random_result['max_fitness'],
        }
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION SUMMARY")
    print("="*80)
    print(f"\n{'Concept':<25} {'Baseline':>10} {'SoftPrompt':>12} {'Rand Mean':>12} {'Rand Max':>12} {'SP>Mean':>8} {'SP>Max':>8}")
    print("-"*80)
    
    for name, r in results.items():
        print(f"{name:<25} {r['baseline_fitness']:>10.4f} {r['soft_prompt_fitness']:>12.4f} "
              f"{r['random_mean_fitness']:>12.4f} {r['random_max_fitness']:>12.4f} "
              f"{'✓' if r['soft_beats_random_mean'] else '✗':>8} "
              f"{'✓' if r['soft_beats_random_max'] else '✗':>8}")
    
    # Overall stats
    n_beats_mean = sum(1 for r in results.values() if r['soft_beats_random_mean'])
    n_beats_max = sum(1 for r in results.values() if r['soft_beats_random_max'])
    
    print("-"*80)
    print(f"Soft prompt beats random mean: {n_beats_mean}/{len(results)}")
    print(f"Soft prompt beats random max:  {n_beats_max}/{len(results)}")
    
    return results