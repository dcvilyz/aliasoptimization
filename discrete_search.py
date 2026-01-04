"""
Discrete Search Optimization in Token Space.

After soft prompt optimization and projection, we refine
using discrete search algorithms in the token space.
"""

import random
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class TokenCandidate:
    """A candidate token sequence with its fitness score."""
    tokens: List[int]
    fitness: float
    metrics: Optional[object] = None
    
    def __lt__(self, other):
        """For heap operations - higher fitness is better."""
        return self.fitness > other.fitness
    
    def __eq__(self, other):
        return self.tokens == other.tokens
    
    def __hash__(self):
        return hash(tuple(self.tokens))


class DiscreteSearchOptimizer:
    """
    Discrete search optimizer for token sequences.
    
    Combines local search, beam search, and evolutionary algorithms
    to find optimal token sequences.
    """
    
    def __init__(
        self,
        vocab_embeddings,
        evaluate_fn: Callable[[List[int]], Tuple[float, object]],
        config,
        special_token_ids: List[int] = None,
    ):
        self.vocab_embeddings = vocab_embeddings
        self.evaluate_fn = evaluate_fn
        self.config = config
        self.opt_config = config.optimization
        
        self.special_token_ids = special_token_ids or [49406, 49407]
        self.eval_cache: Dict[tuple, Tuple[float, object]] = {}
        self.best_candidate = None
        self.patience_counter = 0
        
        # Tracker for results summary
        from metrics import MetricTracker
        self.tracker = MetricTracker()
        
    def evaluate(self, tokens: List[int]) -> Tuple[float, object]:
        """Evaluate a token sequence with caching."""
        key = tuple(tokens)
        if key not in self.eval_cache:
            fitness, metrics = self.evaluate_fn(tokens)
            self.eval_cache[key] = (fitness, metrics)
        return self.eval_cache[key]
    
    def get_token_neighbors(self, token_id: int, k: int = 50) -> List[int]:
        """Get k nearest neighbor tokens in embedding space."""
        neighbors, _ = self.vocab_embeddings.find_nearest(
            self.vocab_embeddings.get_embedding(token_id),
            k=k + len(self.special_token_ids) + 1,
            exclude_ids=self.special_token_ids + [token_id],
        )
        return neighbors[:k].tolist()
    
    def get_random_token(self) -> int:
        """Get a random non-special token."""
        while True:
            token_id = random.randint(0, self.vocab_embeddings.vocab_size - 1)
            if token_id not in self.special_token_ids:
                return token_id
    
    def local_search(
        self,
        initial_tokens: List[int],
        max_iterations: int = None,
        trajectory = None,
        parent_step_id: int = None,
        threshold: float = 0.0,
    ) -> TokenCandidate:
        """Greedy local search with early stopping and optional trajectory logging."""
        if max_iterations is None:
            max_iterations = self.opt_config.discrete_iterations
        
        current_fitness, current_metrics = self.evaluate(initial_tokens)
        current = TokenCandidate(
            tokens=initial_tokens.copy(),
            fitness=current_fitness,
            metrics=current_metrics,
        )
        
        # Log initial state if trajectory provided
        last_step_id = parent_step_id
        if trajectory is not None:
            last_step_id = trajectory.log(
                tokens=initial_tokens,
                fitness=current_fitness,
                threshold=threshold,
                parent_id=parent_step_id,
                method='local_search_init',
                metrics=current_metrics,
            )
        
        print(f"  Local search: initial fitness = {current_fitness:.4f}")
        eval_count = 1
        
        for iteration in range(max_iterations):
            improved = False
            
            for pos in range(len(current.tokens)):
                neighbors = self.get_token_neighbors(
                    current.tokens[pos],
                    k=self.opt_config.discrete_top_k,
                )
                
                for neighbor_token in neighbors:
                    candidate_tokens = current.tokens.copy()
                    candidate_tokens[pos] = neighbor_token
                    
                    fitness, metrics = self.evaluate(candidate_tokens)
                    eval_count += 1
                    
                    # Log every evaluation if trajectory provided
                    if trajectory is not None:
                        trajectory.log(
                            tokens=candidate_tokens,
                            fitness=fitness,
                            threshold=threshold,
                            parent_id=last_step_id,
                            method='local_search_eval',
                            metrics=metrics,
                        )
                    
                    if fitness > current.fitness:
                        print(f"  [iter {iteration}] pos={pos}: {current.fitness:.4f} -> {fitness:.4f} ({eval_count} evals)")
                        current = TokenCandidate(
                            tokens=candidate_tokens,
                            fitness=fitness,
                            metrics=metrics,
                        )
                        
                        # Update last_step_id to this improvement
                        if trajectory is not None:
                            last_step_id = trajectory.log(
                                tokens=candidate_tokens,
                                fitness=fitness,
                                threshold=threshold,
                                parent_id=last_step_id,
                                method='local_search_improvement',
                                metrics=metrics,
                            )
                        
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                print(f"  Local search converged after {iteration} iterations ({eval_count} evals)")
                break
        
        return current
    
    def beam_search(
        self,
        initial_tokens: List[int],
        beam_size: int = None,
        max_depth: int = None,
    ) -> TokenCandidate:
        """Beam search exploration."""
        if beam_size is None:
            beam_size = self.opt_config.discrete_beam_size
        if max_depth is None:
            max_depth = len(initial_tokens) * 2
        
        initial_fitness, initial_metrics = self.evaluate(initial_tokens)
        beam = [TokenCandidate(
            tokens=initial_tokens.copy(),
            fitness=initial_fitness,
            metrics=initial_metrics,
        )]
        
        best = beam[0]
        
        for depth in range(max_depth):
            candidates = []
            
            for current in beam:
                for pos in range(len(current.tokens)):
                    neighbors = self.get_token_neighbors(
                        current.tokens[pos],
                        k=max(1, self.opt_config.discrete_top_k // len(current.tokens)),
                    )
                    
                    for neighbor_token in neighbors:
                        candidate_tokens = current.tokens.copy()
                        candidate_tokens[pos] = neighbor_token
                        
                        fitness, metrics = self.evaluate(candidate_tokens)
                        candidates.append(TokenCandidate(
                            tokens=candidate_tokens,
                            fitness=fitness,
                            metrics=metrics,
                        ))
            
            candidates.sort(key=lambda x: x.fitness, reverse=True)
            beam = candidates[:beam_size]
            
            if beam and beam[0].fitness > best.fitness:
                best = beam[0]
            
            if not candidates:
                break
        
        return best
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Single-point crossover."""
        if len(parent1) != len(parent2):
            child = []
            for i in range(max(len(parent1), len(parent2))):
                if random.random() < 0.5 and i < len(parent1):
                    child.append(parent1[i])
                elif i < len(parent2):
                    child.append(parent2[i])
            return child if child else parent1.copy()
        
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:]
    
    def mutate(self, tokens: List[int], mutation_rate: float = None) -> List[int]:
        """Mutate a token sequence."""
        if mutation_rate is None:
            mutation_rate = self.opt_config.mutation_rate
        
        mutated = tokens.copy()
        
        for pos in range(len(mutated)):
            if random.random() < mutation_rate:
                if random.random() < 0.7:
                    neighbors = self.get_token_neighbors(mutated[pos], k=20)
                    if neighbors:
                        mutated[pos] = random.choice(neighbors)
                else:
                    mutated[pos] = self.get_random_token()
        
        if random.random() < 0.1 and len(mutated) > 1:
            if random.random() < 0.5 and len(mutated) < 10:
                pos = random.randint(0, len(mutated))
                mutated.insert(pos, self.get_random_token())
            elif len(mutated) > 2:
                pos = random.randint(0, len(mutated) - 1)
                mutated.pop(pos)
        
        return mutated
    
    def tournament_selection(
        self,
        population: List[TokenCandidate],
        tournament_size: int = None,
    ) -> TokenCandidate:
        """Tournament selection."""
        if tournament_size is None:
            tournament_size = self.opt_config.tournament_size
        
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def evolutionary_search(
        self,
        initial_population: List[List[int]] = None,
        seed_tokens: List[int] = None,
        population_size: int = None,
        generations: int = None,
        verbose: bool = True,
    ) -> TokenCandidate:
        """Evolutionary algorithm for token optimization."""
        if population_size is None:
            population_size = self.opt_config.population_size
        if generations is None:
            generations = self.opt_config.evolution_generations
        
        population = []
        
        if seed_tokens:
            fitness, metrics = self.evaluate(seed_tokens)
            population.append(TokenCandidate(
                tokens=seed_tokens.copy(),
                fitness=fitness,
                metrics=metrics,
            ))
        
        if initial_population:
            for tokens in initial_population:
                fitness, metrics = self.evaluate(tokens)
                population.append(TokenCandidate(
                    tokens=tokens.copy(),
                    fitness=fitness,
                    metrics=metrics,
                ))
        
        seq_length = len(seed_tokens) if seed_tokens else 5
        while len(population) < population_size:
            if seed_tokens and random.random() < 0.7:
                tokens = self.mutate(seed_tokens, mutation_rate=0.3)
            else:
                tokens = [self.get_random_token() for _ in range(seq_length)]
            
            fitness, metrics = self.evaluate(tokens)
            population.append(TokenCandidate(
                tokens=tokens, fitness=fitness, metrics=metrics
            ))
        
        population.sort(key=lambda x: x.fitness, reverse=True)
        best = population[0]
        self.patience_counter = 0
        
        iterator = tqdm(range(generations), desc="Evolution") if verbose else range(generations)
        
        for gen in iterator:
            new_population = []
            elite_size = self.opt_config.elite_size
            new_population.extend(population[:elite_size])
            
            while len(new_population) < population_size:
                if random.random() < self.opt_config.crossover_rate:
                    parent1 = self.tournament_selection(population)
                    parent2 = self.tournament_selection(population)
                    child_tokens = self.crossover(parent1.tokens, parent2.tokens)
                else:
                    parent = self.tournament_selection(population)
                    child_tokens = parent.tokens.copy()
                
                child_tokens = self.mutate(child_tokens)
                fitness, metrics = self.evaluate(child_tokens)
                new_population.append(TokenCandidate(
                    tokens=child_tokens, fitness=fitness, metrics=metrics
                ))
            
            population = new_population
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            if population[0].fitness > best.fitness:
                best = population[0]
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if verbose and gen % self.config.log_every == 0:
                avg_fitness = np.mean([p.fitness for p in population])
                print(f"Gen {gen}: Best={best.fitness:.4f}, Avg={avg_fitness:.4f}")
            
            if self.patience_counter >= self.opt_config.patience:
                if verbose:
                    print(f"Early stopping at generation {gen}")
                break
        
        return best
    
    def optimize(
        self,
        seed_tokens: List[int],
        verbose: bool = True,
    ) -> TokenCandidate:
        """Run full optimization pipeline."""
        if verbose:
            print("=" * 50)
            print("Starting discrete token optimization")
            print(f"Seed tokens: {seed_tokens}")
            print("=" * 50)
        
        if verbose:
            print("\n[Phase 1] Local Search")
        local_best = self.local_search(seed_tokens)
        if verbose:
            print(f"Local best: fitness={local_best.fitness:.4f}")
        
        # if verbose:
        #     print("\n[Phase 2] Beam Search")
        # beam_best = self.beam_search(local_best.tokens)
        # if verbose:
        #     print(f"Beam best: fitness={beam_best.fitness:.4f}")
        # Skip Beam Search, never improves performance
        beam_best = local_best
        if verbose:
            print("\n[Phase 3] Evolutionary Search")
        
        initial_pop = [seed_tokens, local_best.tokens, beam_best.tokens]
        for _ in range(5):
            initial_pop.append(self.mutate(beam_best.tokens, mutation_rate=0.5))
        
        evo_best = self.evolutionary_search(
            initial_population=initial_pop,
            seed_tokens=beam_best.tokens,
            verbose=verbose,
        )
        
        if verbose:
            print("\n" + "=" * 50)
            print("Optimization complete!")
            print(f"Best fitness: {evo_best.fitness:.4f}")
            print(f"Best tokens: {evo_best.tokens}")
            print("=" * 50)
        
        return evo_best


def create_evaluate_fn(
    model,
    dataloader,
    device: str = 'mps',
    confidence_threshold: float = 0.5,
    max_eval_images: int = 3,  # Only evaluate on a few images!
):
    """
    Create evaluation function for discrete search.
    
    IMPORTANT: For speed, we only evaluate on a small sample of images.
    The soft prompt optimization already used all images - discrete search
    is just refinement, so sampling is acceptable.
    """
    from metrics import compute_combined_score
    
    # Pre-sample images from dataloader for consistent fast evaluation
    all_batches = list(dataloader)
    if len(all_batches) > max_eval_images:
        import random
        sampled_batches = random.sample(all_batches, max_eval_images)
    else:
        sampled_batches = all_batches
    
    def evaluate_fn(tokens: List[int]) -> Tuple[float, object]:
        from soft_prompt import evaluate_token_sequence_on_batches
        
        metrics = evaluate_token_sequence_on_batches(
            model=model,
            token_ids=tokens,
            batches=sampled_batches,
            device=device,
            confidence_threshold=confidence_threshold,
        )
        fitness = compute_combined_score(metrics)
        return fitness, metrics
    
    return evaluate_fn


def threshold_curriculum(
    model,
    dataloader,
    vocab_embeddings,
    config,
    seed_tokens: List[int] = None,
    seed_tokens_list: List[List[int]] = None,
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    top_k: int = 5,
    max_eval_images: int = 3,
    num_random_seeds: int = 0,
    device: str = 'mps',
    verbose: bool = True,
):
    """
    Progressive threshold raising with diverse candidate tracking.
    
    At each threshold level:
    1. Run local search from all current candidates
    2. Keep top_k results for next threshold
    
    This mitigates local minima by maintaining population diversity.
    
    Args:
        model: SAM3 model
        dataloader: DataLoader for evaluation
        vocab_embeddings: Vocabulary embeddings for neighbor search
        config: Config object
        seed_tokens: Initial token sequence to start from (single seed)
        seed_tokens_list: List of initial token sequences (multiple seeds)
        thresholds: List of confidence thresholds to progress through
        top_k: Number of candidates to keep at each level
        max_eval_images: Images to use for fast evaluation
        num_random_seeds: Number of additional random token sequences to add as seeds
        device: Device to use
        verbose: Print progress
        
    Returns:
        Best TokenCandidate at final threshold, all_results dict, and SearchTrajectory
    """
    from soft_prompt import evaluate_token_sequence
    from metrics import SearchTrajectory
    import random
    
    # Get tokenizer for decoding
    tokenizer = model.backbone.language_backbone.tokenizer
    
    # Build initial candidate list from seeds
    initial_candidates = []
    
    if seed_tokens is not None:
        initial_candidates.append(seed_tokens)
    
    if seed_tokens_list is not None:
        initial_candidates.extend(seed_tokens_list)
    
    # Add random seeds if requested
    seq_length = len(initial_candidates[0]) if initial_candidates else config.optimization.soft_prompt_length
    for _ in range(num_random_seeds):
        random_tokens = [
            random.randint(0, vocab_embeddings.vocab_size - 1) 
            for _ in range(seq_length)
        ]
        initial_candidates.append(random_tokens)
    
    if not initial_candidates:
        raise ValueError("Must provide seed_tokens, seed_tokens_list, or num_random_seeds > 0")
    
    if verbose:
        print(f"Starting with {len(initial_candidates)} seed candidates:")
        for i, seeds in enumerate(initial_candidates):
            print(f"  {i+1}. {tokenizer.decode(seeds)[:50]}...")
    
    # Initialize trajectory
    trajectory = SearchTrajectory(
        category_name=config.data.dataset_path if hasattr(config.data, 'dataset_path') else 'unknown',
        config={
            'thresholds': thresholds,
            'top_k': top_k,
            'max_eval_images': max_eval_images,
            'num_initial_seeds': len(initial_candidates),
            'num_random_seeds': num_random_seeds,
        }
    )
    
    # Start with all initial candidates
    candidates = [tokens.copy() for tokens in initial_candidates]
    candidate_parent_ids = [None] * len(candidates)  # Track parent step for each candidate
    
    all_results = {}  # Track results at each threshold
    
    for thresh in thresholds:
        if verbose:
            print(f"\n{'='*60}")
            print(f"THRESHOLD: {thresh}")
            print(f"{'='*60}")
            print(f"Starting with {len(candidates)} candidates")
        
        # Create eval function for this threshold
        eval_fn = create_evaluate_fn(
            model=model,
            dataloader=dataloader,
            device=device,
            confidence_threshold=thresh,
            max_eval_images=max_eval_images,
        )
        
        # Create optimizer with trajectory logging
        discrete_opt = DiscreteSearchOptimizer(
            vocab_embeddings=vocab_embeddings,
            evaluate_fn=eval_fn,
            config=config,
        )
        
        # Search from all candidates
        results = []
        result_step_ids = []
        for i, (seed, parent_id) in enumerate(zip(candidates, candidate_parent_ids)):
            if verbose:
                print(f"\n  Candidate {i+1}/{len(candidates)}: {tokenizer.decode(seed)[:30]}...")
            
            # Log the initial candidate
            init_fitness, init_metrics = eval_fn(seed)
            init_step_id = trajectory.log(
                tokens=seed,
                fitness=init_fitness,
                threshold=thresh,
                parent_id=parent_id,
                method='init' if parent_id is None else 'candidate',
                metrics=init_metrics,
            )
            
            # Run local search with logging
            result = discrete_opt.local_search(
                seed, 
                trajectory=trajectory, 
                parent_step_id=init_step_id,
                threshold=thresh
            )
            results.append(result)
            
            # Log final result of local search
            final_step_id = trajectory.log(
                tokens=result.tokens,
                fitness=result.fitness,
                threshold=thresh,
                parent_id=init_step_id,
                method='local_search_result',
                metrics=result.metrics,
            )
            result_step_ids.append(final_step_id)
            
            if verbose:
                print(f"    -> fitness={result.fitness:.4f}: '{tokenizer.decode(result.tokens)}'")
        
        # Sort by fitness and keep top_k
        results_with_ids = list(zip(results, result_step_ids))
        results_with_ids.sort(key=lambda x: x[0].fitness, reverse=True)
        
        # Deduplicate (same tokens = same candidate)
        seen = set()
        unique_results = []
        unique_step_ids = []
        for r, step_id in results_with_ids:
            key = tuple(r.tokens)
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
                unique_step_ids.append(step_id)
        
        # Keep top_k unique candidates for next round
        candidates = [r.tokens for r in unique_results[:top_k]]
        candidate_parent_ids = unique_step_ids[:top_k]
        
        # Also add some mutations for diversity
        if len(unique_results) > 0:
            best = unique_results[0]
            best_step_id = unique_step_ids[0]
            for _ in range(min(3, top_k - len(candidates))):
                mutated = discrete_opt.mutate(best.tokens, mutation_rate=0.4)
                candidates.append(mutated)
                
                # Log mutation
                mut_fitness, mut_metrics = eval_fn(mutated)
                mut_step_id = trajectory.log(
                    tokens=mutated,
                    fitness=mut_fitness,
                    threshold=thresh,
                    parent_id=best_step_id,
                    method='mutation',
                    metrics=mut_metrics,
                )
                candidate_parent_ids.append(mut_step_id)
        
        all_results[thresh] = unique_results[:top_k]
        
        if verbose:
            print(f"\n  Best at {thresh}: fitness={unique_results[0].fitness:.4f}")
            print(f"    Tokens: {unique_results[0].tokens}")
            print(f"    Decoded: '{tokenizer.decode(unique_results[0].tokens)}'")
            print(f"  Keeping {len(candidates)} candidates for next threshold")
    
    # Return best from final threshold
    final_best = all_results[thresholds[-1]][0]
    
    if verbose:
        print(f"\n{'='*60}")
        print("CURRICULUM COMPLETE")
        print(f"{'='*60}")
        print(f"Final best at threshold {thresholds[-1]}:")
        print(f"  Fitness: {final_best.fitness:.4f}")
        print(f"  Tokens: {final_best.tokens}")
        print(f"  Decoded: '{tokenizer.decode(final_best.tokens)}'")
        
        # Show progression
        print(f"\nProgression across thresholds:")
        for thresh in thresholds:
            if thresh in all_results and all_results[thresh]:
                best = all_results[thresh][0]
                print(f"  {thresh}: {best.fitness:.4f} - '{tokenizer.decode(best.tokens)[:40]}'")
        
        print(f"\nTrajectory: {len(trajectory.steps)} total steps logged")
    
    return final_best, all_results, trajectory