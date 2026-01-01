"""
Fitness Landscape Analysis for Alias Optimization.

If random initialization achieves similar results to soft prompt initialization,
we need to understand the structure of the fitness landscape.

Key questions:
1. Is the landscape flat? (many equally-good solutions everywhere)
2. Are there many basins? (multiple local optima reachable from different starts)
3. Is there one global basin? (everything converges to the same region)
4. Is our fitness metric broken? (not discriminating between good/bad prompts)

Analysis approaches:
1. Sample fitness at random points
2. Track convergence trajectories from different starts
3. Measure distance between converged solutions
4. Visualize with dimensionality reduction
"""

import torch
import numpy as np
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class LandscapeProbe:
    """Result of probing fitness at a point."""
    tokens: List[int]
    decoded: str
    fitness: float
    metrics: dict


def probe_random_points(
    model,
    dataloader,
    tokenizer,
    vocab_size: int,
    num_probes: int = 100,
    seq_length: int = 5,
    device: str = 'mps',
    confidence_threshold: float = 0.5,
) -> List[LandscapeProbe]:
    """
    Sample fitness at random points in token space.
    
    This tells us the "background" fitness level - what fitness do
    random token sequences achieve?
    """
    from discrete_search import create_evaluate_fn
    
    eval_fn = create_evaluate_fn(
        model=model,
        dataloader=dataloader,
        device=device,
        confidence_threshold=confidence_threshold,
    )
    
    special_tokens = [49406, 49407]
    probes = []
    
    for i in range(num_probes):
        # Generate random tokens
        tokens = []
        while len(tokens) < seq_length:
            t = random.randint(0, vocab_size - 1)
            if t not in special_tokens:
                tokens.append(t)
        
        # Evaluate
        fitness, metrics = eval_fn(tokens)
        
        probes.append(LandscapeProbe(
            tokens=tokens,
            decoded=tokenizer.decode(tokens),
            fitness=fitness,
            metrics=metrics.to_dict() if hasattr(metrics, 'to_dict') else {},
        ))
        
        if (i + 1) % 20 == 0:
            print(f"  Probed {i + 1}/{num_probes} points, "
                  f"mean fitness so far: {np.mean([p.fitness for p in probes]):.4f}")
    
    return probes


def analyze_random_probes(probes: List[LandscapeProbe]) -> dict:
    """
    Analyze distribution of random probe fitnesses.
    """
    fitnesses = [p.fitness for p in probes]
    
    return {
        'num_probes': len(probes),
        'mean': np.mean(fitnesses),
        'std': np.std(fitnesses),
        'min': np.min(fitnesses),
        'max': np.max(fitnesses),
        'median': np.median(fitnesses),
        'percentile_25': np.percentile(fitnesses, 25),
        'percentile_75': np.percentile(fitnesses, 75),
        'percentile_90': np.percentile(fitnesses, 90),
        'percentile_99': np.percentile(fitnesses, 99),
        # How many random probes exceed various thresholds
        'pct_above_0.3': np.mean([f > 0.3 for f in fitnesses]) * 100,
        'pct_above_0.4': np.mean([f > 0.4 for f in fitnesses]) * 100,
        'pct_above_0.5': np.mean([f > 0.5 for f in fitnesses]) * 100,
        'pct_above_0.6': np.mean([f > 0.6 for f in fitnesses]) * 100,
    }


def measure_solution_diversity(
    solutions: List[List[int]],
    vocab_embeddings,
) -> dict:
    """
    Measure how diverse the converged solutions are.
    
    If all solutions end up in similar regions, there's one basin.
    If solutions are spread out, there are multiple basins.
    """
    if len(solutions) < 2:
        return {'num_solutions': len(solutions)}
    
    # Get embeddings for each solution
    embeddings = []
    for tokens in solutions:
        emb = vocab_embeddings.get_embeddings(tokens)
        # Average across tokens to get single embedding per solution
        embeddings.append(emb.mean(dim=0))
    
    embeddings = torch.stack(embeddings)  # [N, embed_dim]
    
    # Compute pairwise distances
    from torch.nn.functional import cosine_similarity, pdist
    
    # Euclidean distances
    euclidean_dists = pdist(embeddings).numpy()
    
    # Cosine similarities
    n = len(solutions)
    cosine_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0)
            ).item()
            cosine_sims.append(sim)
    
    # Token overlap
    token_overlaps = []
    for i in range(n):
        for j in range(i + 1, n):
            overlap = len(set(solutions[i]) & set(solutions[j]))
            token_overlaps.append(overlap / len(solutions[i]))
    
    return {
        'num_solutions': n,
        'euclidean_mean': np.mean(euclidean_dists),
        'euclidean_std': np.std(euclidean_dists),
        'cosine_sim_mean': np.mean(cosine_sims),
        'cosine_sim_std': np.std(cosine_sims),
        'token_overlap_mean': np.mean(token_overlaps),
        'token_overlap_std': np.std(token_overlaps),
    }


def run_convergence_analysis(
    model,
    dataloader,
    vocab_embeddings,
    config,
    tokenizer,
    num_starts: int = 10,
    seq_length: int = 5,
    device: str = 'mps',
) -> dict:
    """
    Run discrete search from multiple random starts and analyze convergence.
    
    Questions answered:
    1. Do all starts converge to similar fitness?
    2. Do they converge to similar tokens?
    3. How far do they travel in embedding space?
    """
    from discrete_search import DiscreteSearchOptimizer, create_evaluate_fn
    
    eval_fn = create_evaluate_fn(
        model=model,
        dataloader=dataloader,
        device=device,
        confidence_threshold=config.optimization.confidence_threshold,
    )
    
    vocab_size = vocab_embeddings.vocab_size
    special_tokens = [49406, 49407]
    
    trajectories = []
    final_solutions = []
    final_fitnesses = []
    
    for i in range(num_starts):
        print(f"\n--- Start {i + 1}/{num_starts} ---")
        
        # Generate random start
        random.seed(i * 1000 + 42)
        start_tokens = []
        while len(start_tokens) < seq_length:
            t = random.randint(0, vocab_size - 1)
            if t not in special_tokens:
                start_tokens.append(t)
        
        start_fitness, _ = eval_fn(start_tokens)
        print(f"  Start: '{tokenizer.decode(start_tokens)}' fitness={start_fitness:.4f}")
        
        # Run optimization
        discrete_optimizer = DiscreteSearchOptimizer(
            vocab_embeddings=vocab_embeddings,
            evaluate_fn=eval_fn,
            config=config,
            special_token_ids=special_tokens,
        )
        
        best = discrete_optimizer.optimize(seed_tokens=start_tokens, verbose=False)
        
        print(f"  End: '{tokenizer.decode(best.tokens)}' fitness={best.fitness:.4f}")
        
        trajectories.append({
            'start_tokens': start_tokens,
            'start_fitness': start_fitness,
            'end_tokens': best.tokens,
            'end_fitness': best.fitness,
            'improvement': best.fitness - start_fitness,
        })
        
        final_solutions.append(best.tokens)
        final_fitnesses.append(best.fitness)
    
    # Analyze
    diversity = measure_solution_diversity(final_solutions, vocab_embeddings)
    
    return {
        'num_starts': num_starts,
        'fitness_mean': np.mean(final_fitnesses),
        'fitness_std': np.std(final_fitnesses),
        'fitness_min': np.min(final_fitnesses),
        'fitness_max': np.max(final_fitnesses),
        'improvement_mean': np.mean([t['improvement'] for t in trajectories]),
        'diversity': diversity,
        'trajectories': trajectories,
    }


def run_full_landscape_analysis(
    model,
    concept,  # SaCoConceptData
    vocab_embeddings,
    config,
    tokenizer,
    max_images: int = 10,
    num_random_probes: int = 50,
    num_convergence_starts: int = 5,
    device: str = 'mps',
):
    """
    Run full landscape analysis for a concept.
    """
    from data_loader import get_category_dataloader
    from augmentation_integration import create_category_data_from_concept
    
    print(f"\n{'='*70}")
    print(f"LANDSCAPE ANALYSIS: {concept.text_input}")
    print(f"{'='*70}")
    
    # Create dataloader
    category_data = create_category_data_from_concept(concept, max_images)
    dataloader = get_category_dataloader(
        category_data=category_data,
        batch_size=config.optimization.soft_batch_size,
        image_size=config.data.image_size,
        shuffle=False,
    )
    
    # 1. Probe random points
    print(f"\n[1/3] Probing {num_random_probes} random points...")
    probes = probe_random_points(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        vocab_size=vocab_embeddings.vocab_size,
        num_probes=num_random_probes,
        seq_length=config.optimization.soft_prompt_length,
        device=device,
        confidence_threshold=config.optimization.confidence_threshold,
    )
    probe_stats = analyze_random_probes(probes)
    
    print(f"\n  Random probe statistics:")
    print(f"    Mean fitness: {probe_stats['mean']:.4f}")
    print(f"    Std fitness:  {probe_stats['std']:.4f}")
    print(f"    Max fitness:  {probe_stats['max']:.4f}")
    print(f"    % above 0.5:  {probe_stats['pct_above_0.5']:.1f}%")
    
    # 2. Evaluate baseline
    print(f"\n[2/3] Evaluating baseline...")
    from discrete_search import create_evaluate_fn
    eval_fn = create_evaluate_fn(
        model=model,
        dataloader=dataloader,
        device=device,
        confidence_threshold=config.optimization.confidence_threshold,
    )
    baseline_tokens = tokenizer.encode(concept.text_input)
    baseline_fitness, _ = eval_fn(baseline_tokens)
    print(f"  Baseline '{concept.text_input}': {baseline_fitness:.4f}")
    
    # Compare baseline to random distribution
    baseline_percentile = np.mean([p.fitness < baseline_fitness for p in probes]) * 100
    print(f"  Baseline percentile in random: {baseline_percentile:.1f}%")
    
    # 3. Convergence analysis
    print(f"\n[3/3] Running convergence analysis ({num_convergence_starts} starts)...")
    convergence = run_convergence_analysis(
        model=model,
        dataloader=dataloader,
        vocab_embeddings=vocab_embeddings,
        config=config,
        tokenizer=tokenizer,
        num_starts=num_convergence_starts,
        seq_length=config.optimization.soft_prompt_length,
        device=device,
    )
    
    print(f"\n  Convergence statistics:")
    print(f"    Final fitness mean: {convergence['fitness_mean']:.4f}")
    print(f"    Final fitness std:  {convergence['fitness_std']:.4f}")
    print(f"    Solution similarity (cosine): {convergence['diversity']['cosine_sim_mean']:.4f}")
    print(f"    Token overlap: {convergence['diversity']['token_overlap_mean']:.2%}")
    
    # Summary
    print(f"\n{'='*70}")
    print("LANDSCAPE SUMMARY")
    print(f"{'='*70}")
    
    # Interpretation
    if probe_stats['pct_above_0.5'] > 20:
        print("⚠️  Many random tokens achieve high fitness (>50% at 0.5+)")
        print("   → Fitness landscape may be too flat or metric too permissive")
    
    if convergence['diversity']['cosine_sim_mean'] > 0.8:
        print("✓  Solutions converge to similar region (cosine sim > 0.8)")
        print("   → Single basin or strong attractor")
    elif convergence['diversity']['cosine_sim_mean'] < 0.3:
        print("⚠️  Solutions are diverse (cosine sim < 0.3)")
        print("   → Multiple basins or flat landscape")
    
    if convergence['fitness_std'] < 0.05:
        print("✓  Consistent final fitness (std < 0.05)")
        print("   → Optimization is stable")
    else:
        print("⚠️  Variable final fitness (std > 0.05)")
        print("   → Optimization outcome depends heavily on start")
    
    return {
        'concept': concept.text_input,
        'baseline_fitness': baseline_fitness,
        'baseline_percentile': baseline_percentile,
        'random_probe_stats': probe_stats,
        'convergence': convergence,
    }


# =============================================================================
# Usage
# =============================================================================

"""
USAGE IN NOTEBOOK:

from landscape_analysis import run_full_landscape_analysis

# Analyze fitness landscape for a concept
analysis = run_full_landscape_analysis(
    model=model,
    concept=dataset['draft weapon'],
    vocab_embeddings=vocab_embeddings_1024,
    config=config,
    tokenizer=model.backbone.language_backbone.tokenizer,
    max_images=10,
    num_random_probes=50,
    num_convergence_starts=5,
    device=DEVICE,
)

# The analysis will tell us:
# 1. What's the distribution of fitness at random points?
# 2. How does baseline compare to random?
# 3. Do different starts converge to similar solutions?
# 4. Is the landscape flat, single-basin, or multi-basin?
"""