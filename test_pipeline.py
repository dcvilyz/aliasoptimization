#!/usr/bin/env python3
"""
Test script for Alias Optimization pipeline.

This tests the core logic without requiring the full SAM3 model.
Useful for development and debugging.
"""

import torch
import numpy as np
import random
from typing import List, Tuple
from pathlib import Path

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def test_config():
    """Test configuration loading."""
    print("\n[TEST] Configuration...")
    
    from config import Config, get_config
    
    config = get_config()
    assert config.model.vocab_size == 49408
    assert config.optimization.soft_prompt_length == 5
    
    # Test overrides
    config = get_config(device="cpu", **{"optimization.soft_prompt_length": 10})
    assert config.device == "cpu"
    assert config.optimization.soft_prompt_length == 10
    
    print("  ✓ Config loading works")


def test_metrics():
    """Test metric computation."""
    print("\n[TEST] Metrics...")
    
    from metrics import (
        compute_mask_iou,
        compute_metrics_single_image,
        compute_metrics_batch,
        soft_iou_loss,
    )
    
    # Create test masks
    pred_mask = torch.zeros(64, 64)
    pred_mask[10:30, 10:30] = 1
    
    gt_mask = torch.zeros(64, 64)
    gt_mask[15:35, 15:35] = 1
    
    # Test IoU computation
    iou = compute_mask_iou(pred_mask, gt_mask)
    assert 0 < iou < 1, f"Expected IoU between 0 and 1, got {iou}"
    print(f"  ✓ Single mask IoU: {iou:.4f}")
    
    # Test batch IoU
    pred_batch = pred_mask.unsqueeze(0).repeat(3, 1, 1)  # 3 predictions
    gt_batch = gt_mask.unsqueeze(0).repeat(2, 1, 1)  # 2 ground truth
    
    iou_matrix = compute_mask_iou(pred_batch, gt_batch)
    assert iou_matrix.shape == (3, 2)
    print(f"  ✓ Batch IoU matrix shape: {iou_matrix.shape}")
    
    # Test metrics computation
    metrics = compute_metrics_single_image(pred_batch, gt_batch)
    assert 'mean_iou' in metrics
    assert 'recall' in metrics
    assert 'precision' in metrics
    print(f"  ✓ Single image metrics: IoU={metrics['mean_iou']:.4f}, Recall={metrics['recall']:.4f}")
    
    # Test batch metrics
    pred_list = [pred_batch, pred_batch[:2]]
    gt_list = [gt_batch, gt_batch[:1]]
    
    batch_metrics = compute_metrics_batch(pred_list, gt_list)
    assert batch_metrics.mean_iou >= 0
    print(f"  ✓ Batch metrics: {batch_metrics}")
    
    # Test differentiable loss
    pred_logits = torch.randn(5, 64, 64, requires_grad=True)
    loss = soft_iou_loss(pred_logits, gt_batch)
    loss.backward()
    assert pred_logits.grad is not None
    print(f"  ✓ Soft IoU loss: {loss.item():.4f} (gradients computed)")


def test_vocabulary_embeddings():
    """Test vocabulary embedding utilities."""
    print("\n[TEST] Vocabulary Embeddings...")
    
    from embeddings import VocabularyEmbeddings
    
    # Create mock vocabulary embeddings
    vocab_size = 1000
    embed_dim = 256
    
    embeddings = torch.randn(vocab_size, embed_dim)
    token_ids = list(range(vocab_size))
    
    vocab_emb = VocabularyEmbeddings(
        embeddings=embeddings,
        token_ids=token_ids,
        embed_dim=embed_dim,
    )
    
    assert vocab_emb.vocab_size == vocab_size
    print(f"  ✓ Created vocabulary with {vocab_size} tokens")
    
    # Test nearest neighbor search
    query = torch.randn(embed_dim)
    nearest_ids, similarities = vocab_emb.find_nearest(query, k=10)
    
    assert len(nearest_ids) == 10
    assert similarities[0] >= similarities[-1]  # Should be sorted
    print(f"  ✓ Nearest neighbor search: found {len(nearest_ids)} neighbors")
    
    # Test exclusion
    exclude = [0, 1, 2]
    nearest_ids, _ = vocab_emb.find_nearest(query, k=5, exclude_ids=exclude)
    for excl in exclude:
        assert excl not in nearest_ids.tolist()
    print(f"  ✓ Exclusion works")
    
    # Test single embedding retrieval
    emb = vocab_emb.get_embedding(50)
    assert emb.shape == (embed_dim,)
    print(f"  ✓ Single embedding retrieval works")


def test_discrete_search():
    """Test discrete search optimizer."""
    print("\n[TEST] Discrete Search...")
    
    from config import Config
    from discrete_search import DiscreteSearchOptimizer, TokenCandidate
    from embeddings import VocabularyEmbeddings
    
    # Create mock components
    vocab_size = 500
    embed_dim = 64
    
    embeddings = torch.randn(vocab_size, embed_dim)
    token_ids = list(range(vocab_size))
    
    vocab_emb = VocabularyEmbeddings(
        embeddings=embeddings,
        token_ids=token_ids,
        embed_dim=embed_dim,
    )
    
    # Mock evaluation function
    # Higher fitness for tokens with lower IDs (artificial objective)
    def mock_evaluate(tokens: List[int]) -> Tuple[float, dict]:
        fitness = 1.0 / (1.0 + np.mean(tokens) / 100)
        metrics = {'mock_metric': fitness}
        return fitness, metrics
    
    config = Config()
    config.optimization.discrete_iterations = 10
    config.optimization.discrete_beam_size = 5
    config.optimization.population_size = 20
    config.optimization.evolution_generations = 10
    config.log_every = 5
    
    optimizer = DiscreteSearchOptimizer(
        vocab_embeddings=vocab_emb,
        evaluate_fn=mock_evaluate,
        config=config,
        special_token_ids=[],  # No special tokens in mock
    )
    
    # Test local search
    seed = [100, 150, 200, 250, 300]
    local_best = optimizer.local_search(seed, max_iterations=5)
    print(f"  ✓ Local search: {seed} -> {local_best.tokens} (fitness: {local_best.fitness:.4f})")
    
    # Test beam search
    beam_best = optimizer.beam_search(seed, beam_size=3, max_depth=3)
    print(f"  ✓ Beam search: fitness={beam_best.fitness:.4f}")
    
    # Test evolutionary search (short version)
    evo_best = optimizer.evolutionary_search(
        seed_tokens=seed,
        population_size=10,
        generations=5,
        verbose=False,
    )
    print(f"  ✓ Evolution: {evo_best.tokens} (fitness: {evo_best.fitness:.4f})")


def test_soft_prompt_module():
    """Test soft prompt module."""
    print("\n[TEST] Soft Prompt Module...")
    
    from soft_prompt import SoftPromptModule
    
    seq_length = 5
    embed_dim = 256
    
    # Test random initialization
    soft_prompt = SoftPromptModule(seq_length, embed_dim)
    
    assert soft_prompt.soft_embeddings.shape == (seq_length, embed_dim)
    assert soft_prompt.soft_embeddings.requires_grad
    print(f"  ✓ Random init: shape={soft_prompt.soft_embeddings.shape}")
    
    # Test with custom initialization
    init_emb = torch.randn(seq_length, embed_dim)
    soft_prompt = SoftPromptModule(seq_length, embed_dim, init_embeddings=init_emb)
    
    assert torch.allclose(soft_prompt.soft_embeddings, init_emb)
    print(f"  ✓ Custom init works")
    
    # Test forward
    output = soft_prompt()
    assert output.shape == (seq_length, embed_dim)
    print(f"  ✓ Forward pass works")


def test_token_candidate():
    """Test TokenCandidate dataclass."""
    print("\n[TEST] Token Candidate...")
    
    from discrete_search import TokenCandidate
    
    c1 = TokenCandidate(tokens=[1, 2, 3], fitness=0.8)
    c2 = TokenCandidate(tokens=[4, 5, 6], fitness=0.6)
    c3 = TokenCandidate(tokens=[1, 2, 3], fitness=0.9)
    
    # Test comparison (higher fitness = "less than" for max-heap)
    assert c1 < c2  # 0.8 > 0.6, so c1 should be "less" for heap
    print(f"  ✓ Comparison works for heap")
    
    # Test equality (based on tokens)
    assert c1 == c3  # Same tokens
    assert c1 != c2  # Different tokens
    print(f"  ✓ Equality based on tokens")
    
    # Test hashing
    s = {c1, c2, c3}
    assert len(s) == 2  # c1 and c3 have same tokens
    print(f"  ✓ Hashing works")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("ALIAS OPTIMIZATION TESTS")
    print("=" * 60)
    
    try:
        test_config()
        test_metrics()
        test_vocabulary_embeddings()
        test_token_candidate()
        test_discrete_search()
        test_soft_prompt_module()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
