#!/usr/bin/env python
"""
Resume eval_full from where it left off.

Already completed (from sports and food datasets):
- sports equipment: tc 3,7,11
- bowling pin: tc 3,7,11  
- fruit: tc 3,7,11
- dumpling: tc 3,7,11

Need to run (had seed bug causing 0.0 fitness):
- the wall (crowded)
- cabinet (crowded)
- wall (metaclip)
- the glass (metaclip)
- adult shoe (attributes)
- leather bag (attributes)
- horse and buggy (wiki)
- mortar and pestle (wiki)

After completion, compiles all results into unified analysis.
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, '.')

def run_remaining_evals():
    from saco_loader import load_saco_dataset
    from eval_pipeline import run_full_evaluation
    from config import Config
    from embeddings import load_both_vocab_embeddings
    from sam3.model_builder import build_sam3_image_model
    
    # Load model once
    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    model.to('cuda')
    model.eval()
    
    config = Config()
    config.device = 'cuda'
    
    print("Loading embeddings...")
    _, vocab_1024 = load_both_vocab_embeddings(model=model, device='cuda')
    
    # REMAINING concepts (not yet run or had 0.0 bug)
    remaining_by_file = {
        'gold_crowded_merged_a_release_test.json': ['the wall', 'cabinet'],
        'gold_metaclip_merged_a_release_test.json': ['wall', 'the glass'],
        'gold_attributes_merged_a_release_test.json': ['adult shoe', 'leather bag'],
        'gold_wiki_common_merged_a_release_test.json': ['horse and buggy', 'mortar and pestle'],
    }
    
    token_counts = [3, 7, 11]
    output_dir = 'eval_full'
    
    for ann_file, concept_list in remaining_by_file.items():
        print(f"\n{'='*60}")
        print(f"Loading {ann_file}")
        print(f"{'='*60}")
        
        dataset = load_saco_dataset(
            '/home/ubuntu/sam3/SACo_Gold_bundle/gt-annotations',
            annotation_file=ann_file,
            images_dir='/home/ubuntu/sam3/SACo_Gold_bundle/metaclip-images',
        )
        
        summary = run_full_evaluation(
            model=model,
            dataset=dataset,
            concept_names=concept_list,
            config=config,
            vocab_embeddings=vocab_1024,
            token_counts=token_counts,
            output_dir=output_dir,
        )
    
    print("\n" + "="*60)
    print("REMAINING EVALUATIONS COMPLETE")
    print("="*60)


def compile_all_results():
    """Compile all results from eval_full into unified analysis."""
    import numpy as np
    
    output_dir = Path('eval_full')
    
    # Find all result files
    result_files = sorted(output_dir.glob('result_*.json'))
    
    all_results = []
    for rf in result_files:
        with open(rf) as f:
            result = json.load(f)
            all_results.append(result)
    
    print(f"\nLoaded {len(all_results)} result files")
    
    # Group by concept
    by_concept = {}
    for r in all_results:
        concept = r['concept_name']
        if concept not in by_concept:
            by_concept[concept] = []
        by_concept[concept].append(r)
    
    # Analysis
    print("\n" + "="*70)
    print("COMPREHENSIVE EVAL_FULL ANALYSIS")
    print("="*70)
    
    print(f"\nTotal runs: {len(all_results)}")
    print(f"Unique concepts: {len(by_concept)}")
    
    # Per-concept summary
    print("\n--- PER CONCEPT RESULTS ---")
    print(f"{'Concept':<25} {'TC':<4} {'Baseline':<10} {'Search':<10} {'Δ':<10} {'Win':<5}")
    print("-"*70)
    
    wins = 0
    total = 0
    improvements = []
    
    for concept in sorted(by_concept.keys()):
        for r in sorted(by_concept[concept], key=lambda x: x['token_count']):
            baseline = r['baseline_test_fitness']
            search = r['search_test_fitness']
            delta = search - baseline
            win = '✓' if search > baseline else ''
            if search > baseline:
                wins += 1
            total += 1
            improvements.append(delta)
            
            print(f"{concept:<25} {r['token_count']:<4} {baseline:<10.4f} {search:<10.4f} {delta:<+10.4f} {win:<5}")
    
    print("-"*70)
    print(f"\nOVERALL:")
    print(f"  Win rate: {wins}/{total} ({100*wins/total:.1f}%)")
    print(f"  Mean improvement: {np.mean(improvements):+.4f}")
    print(f"  Best improvement: {max(improvements):+.4f}")
    print(f"  Worst: {min(improvements):+.4f}")
    
    # By token count
    print("\n--- BY TOKEN COUNT ---")
    for tc in [3, 7, 11]:
        tc_results = [r for r in all_results if r['token_count'] == tc]
        tc_improvements = [r['search_test_fitness'] - r['baseline_test_fitness'] for r in tc_results]
        tc_wins = sum(1 for d in tc_improvements if d > 0)
        print(f"  TC={tc}: mean Δ={np.mean(tc_improvements):+.4f}, wins={tc_wins}/{len(tc_results)}")
    
    # Save compiled analysis
    compiled = {
        'total_runs': len(all_results),
        'unique_concepts': len(by_concept),
        'win_rate': wins / total if total > 0 else 0,
        'mean_improvement': float(np.mean(improvements)),
        'results_by_concept': {
            concept: [
                {
                    'token_count': r['token_count'],
                    'baseline': r['baseline_test_fitness'],
                    'search': r['search_test_fitness'],
                    'delta': r['search_test_fitness'] - r['baseline_test_fitness'],
                    'search_wins': r['search_test_fitness'] > r['baseline_test_fitness'],
                }
                for r in sorted(by_concept[concept], key=lambda x: x['token_count'])
            ]
            for concept in sorted(by_concept.keys())
        },
        'by_token_count': {
            tc: {
                'mean_delta': float(np.mean([r['search_test_fitness'] - r['baseline_test_fitness'] 
                                            for r in all_results if r['token_count'] == tc])),
                'win_rate': sum(1 for r in all_results if r['token_count'] == tc 
                               and r['search_test_fitness'] > r['baseline_test_fitness']) / 
                           len([r for r in all_results if r['token_count'] == tc])
            }
            for tc in [3, 7, 11]
        }
    }
    
    with open(output_dir / 'compiled_analysis.json', 'w') as f:
        json.dump(compiled, f, indent=2)
    
    print(f"\nCompiled analysis saved to: {output_dir}/compiled_analysis.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Run remaining evals')
    parser.add_argument('--compile', action='store_true', help='Compile all results')
    parser.add_argument('--all', action='store_true', help='Run remaining then compile')
    args = parser.parse_args()
    
    if args.all or args.run:
        run_remaining_evals()
    
    if args.all or args.compile:
        compile_all_results()
    
    if not (args.run or args.compile or args.all):
        print("Usage:")
        print("  python resume_eval.py --run      # Run remaining evals")
        print("  python resume_eval.py --compile  # Compile all results")
        print("  python resume_eval.py --all      # Run then compile")
