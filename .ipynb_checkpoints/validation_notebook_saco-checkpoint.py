# %% [markdown]
# # Alias Optimization - Method Validation (SA-Co)
# 
# Validates our optimization pipeline on SA-Co Gold (in-distribution) data.
# 
# **Protocol:**
# - For each concept (noun phrase): optimize tokens, compare to GT text prompt
# - Success = `IoU(optimized) >= IoU(baseline)`

# %% [markdown]
# ## 1. Setup & Imports

# %%
import sys
import torch
import json
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

# Add project path - UPDATE THIS
PROJECT_PATH = '/path/to/your/project'
sys.path.insert(0, PROJECT_PATH)

# Import project modules
from config import Config, get_config
from saco_loader import (
    SaCoDataset, 
    SaCoConceptData,
    load_saco_dataset, 
    get_concept_dataloader,
    print_concept_table,
)
from embeddings import VocabularyEmbeddings, build_vocabulary_embeddings
from optimizer import AliasOptimizer
from metrics import MetricResult, compute_combined_score
from soft_prompt import evaluate_token_sequence

# %% [markdown]
# ## 2. Configuration

# %%
# Paths - UPDATE THESE
DEVICE = 'mps'  # or 'cuda'
SACO_PATH = '/path/to/SA-Co-Gold'  # UPDATE THIS
OUTPUT_DIR = 'validation_results_saco'
TOKEN_MAP_PATH = 'cache/token_activation_map_256d.pt'

Path(OUTPUT_DIR).mkdir(exist_ok=True)

print(f"Device: {DEVICE}")
print(f"SA-Co path: {SACO_PATH}")
print(f"Output dir: {OUTPUT_DIR}")

# %% [markdown]
# ## 3. Load Model & Vocabulary Embeddings

# %%
# Load SAM3 model
from sam3.model_builder import build_sam3_image_model

print("Loading SAM3 model...")
model = build_sam3_image_model()
model.to(DEVICE)
model.eval()
print(f"✓ Model loaded on {DEVICE}")

# %%
# Load vocabulary embeddings (256-dim)
print(f"\nLoading vocabulary embeddings from: {TOKEN_MAP_PATH}")

if Path(TOKEN_MAP_PATH).exists():
    vocab_data = torch.load(TOKEN_MAP_PATH, map_location='cpu')
    
    # Handle both formats
    if isinstance(vocab_data, torch.Tensor):
        vocab_embeddings = VocabularyEmbeddings(
            embeddings=vocab_data,
            token_ids=list(range(vocab_data.shape[0])),
            embed_dim=vocab_data.shape[1],
        )
    else:
        vocab_embeddings = VocabularyEmbeddings.load(TOKEN_MAP_PATH, device='cpu')
    
    print(f"✓ Loaded: {vocab_embeddings.vocab_size} tokens, {vocab_embeddings.embed_dim}-dim")
else:
    raise FileNotFoundError(f"Token map not found at {TOKEN_MAP_PATH}")

assert vocab_embeddings.embed_dim == 256, f"Expected 256-dim, got {vocab_embeddings.embed_dim}"

# %% [markdown]
# ## 4. Load SA-Co Dataset & Explore

# %%
# Load SA-Co dataset
print(f"\nLoading SA-Co dataset from: {SACO_PATH}")
dataset = load_saco_dataset(SACO_PATH)
print(dataset.summary())

# %%
# Explore concepts - this shows all noun phrases with their stats
print("\n" + "="*70)
print("SA-Co CONCEPTS OVERVIEW")
print("="*70)

print_concept_table(dataset, min_images=1, max_rows=30, sort_by='instances')

# %%
# Get detailed concept list for selection
concept_info = dataset.list_concepts(min_positive_images=1, sort_by='instances')

print(f"\nTotal concepts: {len(concept_info)}")
print(f"Concepts with >= 5 images: {sum(1 for c in concept_info if c['num_positive_images'] >= 5)}")
print(f"Concepts with >= 10 images: {sum(1 for c in concept_info if c['num_positive_images'] >= 10)}")

# %% [markdown]
# ## 5. Select Concepts for Validation
# 
# **Modify this cell to select which concepts to validate**

# %%
# ============================================================================
# SELECT CONCEPTS HERE
# ============================================================================

# Option 1: Just test with 1-2 concepts first
VALIDATION_CONCEPTS = [
    concept_info[0]['text_input'],  # Concept with most instances
    concept_info[1]['text_input'],  # Second most
]

# Option 2: All concepts with >= N images
# MIN_IMAGES = 5
# VALIDATION_CONCEPTS = [c['text_input'] for c in concept_info if c['num_positive_images'] >= MIN_IMAGES]

# Option 3: Specific concepts by name
# VALIDATION_CONCEPTS = ['dog', 'cat', 'fire hydrant']

# Option 4: First N concepts
# VALIDATION_CONCEPTS = [c['text_input'] for c in concept_info[:10]]

print(f"\nSelected {len(VALIDATION_CONCEPTS)} concepts for validation:")
for i, name in enumerate(VALIDATION_CONCEPTS[:10]):
    concept = dataset[name]
    print(f"  {i+1}. '{name}' ({concept.num_positive_images} images, {concept.num_instances} instances)")
if len(VALIDATION_CONCEPTS) > 10:
    print(f"  ... and {len(VALIDATION_CONCEPTS) - 10} more")

# %% [markdown]
# ## 6. Preview Sample Images (Optional)

# %%
# Visualize samples from selected concepts
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def show_concept_samples(concept: SaCoConceptData, num_samples: int = 2):
    """Show sample images and masks for a concept."""
    num_samples = min(num_samples, concept.num_positive_images)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"Concept: '{concept.text_input}'", fontsize=14, fontweight='bold')
    
    for i in range(num_samples):
        img_path = concept.positive_image_paths[i]
        masks = concept.positive_masks[i]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i+1}")
        axes[i, 0].axis('off')
        
        # Composite mask
        if masks:
            h, w = masks[0].shape
            composite = np.zeros((h, w), dtype=np.float32)
            for m in masks:
                composite = np.maximum(composite, m.astype(np.float32))
            axes[i, 1].imshow(composite, cmap='viridis')
            axes[i, 1].set_title(f"Masks ({len(masks)} instances)")
        else:
            axes[i, 1].text(0.5, 0.5, "No masks", ha='center', va='center')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Show samples for first 2 selected concepts
for concept_name in VALIDATION_CONCEPTS[:2]:
    concept = dataset[concept_name]
    show_concept_samples(concept, num_samples=2)

# %% [markdown]
# ## 7. Validation Configuration

# %%
# Validation parameters
MAX_IMAGES_PER_CONCEPT = 10   # Images to use per concept
SOFT_PROMPT_STEPS = 200       # Gradient descent steps
SOFT_PROMPT_LENGTH = 5        # Number of tokens in soft prompt  
CONFIDENCE_THRESHOLD = 0.5    # Mask confidence threshold

# Create config
config = get_config(
    dataset_path=SACO_PATH,
    device=DEVICE,
    output_dir=OUTPUT_DIR,
)
config.optimization.soft_steps = SOFT_PROMPT_STEPS
config.optimization.soft_prompt_length = SOFT_PROMPT_LENGTH
config.optimization.confidence_threshold = CONFIDENCE_THRESHOLD

print("Validation Configuration:")
print(f"  Concepts to validate: {len(VALIDATION_CONCEPTS)}")
print(f"  Max images per concept: {MAX_IMAGES_PER_CONCEPT}")
print(f"  Soft prompt length: {SOFT_PROMPT_LENGTH} tokens")
print(f"  Soft prompt steps: {SOFT_PROMPT_STEPS}")
print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")

# %% [markdown]
# ## 8. Validation Functions

# %%
def evaluate_text_on_concept(
    model, 
    text: str, 
    concept: SaCoConceptData,
    config: Config,
    max_images: int = 10,
) -> tuple:
    """Evaluate a text prompt on a concept's images."""
    
    # Create dataloader for this concept
    dataloader = get_concept_dataloader(
        concept_data=concept,
        batch_size=config.optimization.soft_batch_size,
        image_size=config.data.image_size,
        max_images=max_images,
        shuffle=False,
    )
    
    # Encode text to tokens
    tokenizer = model.backbone.language_backbone.tokenizer
    token_ids = tokenizer.encode(text)
    
    # Evaluate
    metrics = evaluate_token_sequence(
        model=model,
        token_ids=token_ids,
        dataloader=dataloader,
        device=config.device,
        confidence_threshold=config.optimization.confidence_threshold,
    )
    
    fitness = compute_combined_score(metrics)
    return fitness, metrics, token_ids


def create_category_data_from_concept(concept: SaCoConceptData, max_images: int = 10):
    """
    Create a CategoryData-like object from SaCoConceptData.
    
    This allows us to reuse the existing AliasOptimizer which expects CategoryData.
    """
    from data_loader import CategoryData
    
    num_images = min(concept.num_positive_images, max_images)
    
    # Convert masks to torch tensors
    masks_tensors = []
    for masks_list in concept.positive_masks[:num_images]:
        masks_t = [torch.from_numpy(m).bool() for m in masks_list]
        if masks_t:
            masks_tensors.append(torch.stack(masks_t))
        else:
            masks_tensors.append(torch.zeros((0, 256, 256), dtype=torch.bool))
    
    return CategoryData(
        category_id=hash(concept.text_input) % 100000,  # Generate pseudo-ID
        category_name=concept.text_input,
        image_paths=concept.positive_image_paths[:num_images],
        masks=masks_tensors,
        num_images=num_images,
        num_instances=sum(len(m) for m in concept.positive_masks[:num_images]),
    )


def validate_single_concept(
    model,
    optimizer: AliasOptimizer,
    concept: SaCoConceptData,
    config: Config,
    max_images: int = 10,
    verbose: bool = True,
) -> dict:
    """Run validation for a single concept."""
    
    text_input = concept.text_input
    
    if verbose:
        print(f"\n  Concept: '{text_input}'")
        print(f"  Images: {min(concept.num_positive_images, max_images)}, "
              f"Instances: {concept.num_instances}")
    
    # 1. Evaluate baseline (the GT text prompt)
    if verbose:
        print(f"\n  [1/2] Evaluating baseline: '{text_input}'")
    
    baseline_fitness, baseline_metrics, baseline_tokens = evaluate_text_on_concept(
        model, text_input, concept, config, max_images
    )
    
    if verbose:
        print(f"    Fitness: {baseline_fitness:.4f}")
        print(f"    IoU: {baseline_metrics.mean_iou:.4f}")
        print(f"    Matched: {baseline_metrics.num_matched}/{baseline_metrics.num_ground_truth}")
    
    # 2. Run optimization
    if verbose:
        print(f"\n  [2/2] Running optimization...")
    
    import time
    start = time.time()
    
    # Convert to CategoryData for optimizer compatibility
    category_data = create_category_data_from_concept(concept, max_images)
    
    opt_result = optimizer.optimize_category(
        category_data=category_data,
        baseline_text=text_input,
        verbose=verbose,
    )
    
    opt_time = time.time() - start
    
    # 3. Compare
    improvement = opt_result.best_fitness - baseline_fitness
    relative_imp = improvement / baseline_fitness if baseline_fitness > 0 else 0
    wins = opt_result.best_fitness >= baseline_fitness
    
    return {
        'text_input': text_input,
        'num_images': min(concept.num_positive_images, max_images),
        'num_instances': sum(len(m) for m in concept.positive_masks[:max_images]),
        'baseline_fitness': baseline_fitness,
        'baseline_iou': baseline_metrics.mean_iou,
        'baseline_matched': f"{baseline_metrics.num_matched}/{baseline_metrics.num_ground_truth}",
        'baseline_tokens': baseline_tokens,
        'optimized_fitness': opt_result.best_fitness,
        'optimized_iou': opt_result.best_metrics.get('mean_iou', 0),
        'optimized_tokens': opt_result.best_tokens,
        'optimized_decoded': opt_result.best_decoded,
        'improvement': improvement,
        'relative_improvement': relative_imp,
        'optimized_wins': wins,
        'time_seconds': opt_time,
    }

# %% [markdown]
# ## 9. Run Validation Loop

# %%
# Create optimizer with pre-loaded vocabulary embeddings
optimizer = AliasOptimizer(
    model=model,
    config=config,
    vocab_embeddings=vocab_embeddings,
)

# %%
# Run validation
print("\n" + "="*70)
print("STARTING VALIDATION")
print("="*70)
print(f"Validating {len(VALIDATION_CONCEPTS)} concepts")

results = []
failed = []

for i, concept_name in enumerate(VALIDATION_CONCEPTS):
    print(f"\n{'='*70}")
    print(f"[{i+1}/{len(VALIDATION_CONCEPTS)}] {concept_name}")
    print("="*70)
    
    try:
        concept = dataset[concept_name]
        
        result = validate_single_concept(
            model=model,
            optimizer=optimizer,
            concept=concept,
            config=config,
            max_images=MAX_IMAGES_PER_CONCEPT,
            verbose=True,
        )
        
        results.append(result)
        
        # Print result
        winner = "✓ OPTIMIZED" if result['optimized_wins'] else "✗ BASELINE"
        print(f"\n  RESULT: {winner}")
        print(f"    Baseline: {result['baseline_fitness']:.4f} ('{result['text_input']}')")
        print(f"    Optimized: {result['optimized_fitness']:.4f} ('{result['optimized_decoded']}')")
        print(f"    Δ = {result['improvement']:+.4f} ({result['relative_improvement']*100:+.1f}%)")
        
        # Save intermediate
        with open(f"{OUTPUT_DIR}/result_{i:03d}.json", 'w') as f:
            json.dump(result, f, indent=2)
            
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        failed.append({'concept': concept_name, 'error': str(e)})

# %% [markdown]
# ## 10. Results Summary

# %%
if results:
    num_wins = sum(1 for r in results if r['optimized_wins'])
    win_rate = num_wins / len(results)
    mean_baseline = sum(r['baseline_fitness'] for r in results) / len(results)
    mean_optimized = sum(r['optimized_fitness'] for r in results) / len(results)
    mean_improvement = sum(r['improvement'] for r in results) / len(results)
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\n{'Concept':<35} {'Base':>8} {'Opt':>8} {'Δ':>8} {'Win':>6}")
    print("-"*70)
    
    for r in sorted(results, key=lambda x: x['improvement'], reverse=True):
        name = r['text_input'][:33]
        base = f"{r['baseline_fitness']:.4f}"
        opt = f"{r['optimized_fitness']:.4f}"
        delta = f"{r['improvement']:+.4f}"
        win = "✓" if r['optimized_wins'] else "✗"
        print(f"{name:<35} {base:>8} {opt:>8} {delta:>8} {win:>6}")
    
    print("-"*70)
    
    print(f"\nAggregate Results:")
    print(f"  Win rate: {win_rate*100:.1f}% ({num_wins}/{len(results)})")
    print(f"  Mean baseline fitness: {mean_baseline:.4f}")
    print(f"  Mean optimized fitness: {mean_optimized:.4f}")
    print(f"  Mean improvement: {mean_improvement:+.4f}")
    
    if failed:
        print(f"\n  Failed: {len(failed)}")
        for f in failed:
            print(f"    - '{f['concept']}': {f['error']}")
    
    print("\n" + "="*70)
    if win_rate >= 0.5:
        print("✓ VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED")
    print("="*70)

# %%
# Save full results
summary = {
    'timestamp': datetime.now().isoformat(),
    'config': {
        'dataset': SACO_PATH,
        'max_images': MAX_IMAGES_PER_CONCEPT,
        'soft_steps': SOFT_PROMPT_STEPS,
        'soft_prompt_length': SOFT_PROMPT_LENGTH,
    },
    'num_concepts': len(results),
    'num_wins': num_wins if results else 0,
    'win_rate': win_rate if results else 0,
    'mean_baseline': mean_baseline if results else 0,
    'mean_optimized': mean_optimized if results else 0,
    'mean_improvement': mean_improvement if results else 0,
    'results': results,
    'failed': failed,
}

with open(f"{OUTPUT_DIR}/validation_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Results saved to: {OUTPUT_DIR}/validation_summary.json")

# %% [markdown]
# ## 11. Analysis (Optional)

# %%
# Scatter plot: baseline vs optimized
if results and len(results) > 1:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    baselines = [r['baseline_fitness'] for r in results]
    optimized = [r['optimized_fitness'] for r in results]
    wins = [r['optimized_wins'] for r in results]
    
    # Plot 1: Baseline vs Optimized
    colors = ['green' if w else 'red' for w in wins]
    axes[0].scatter(baselines, optimized, c=colors, alpha=0.7, s=80)
    
    max_val = max(max(baselines), max(optimized)) * 1.1
    axes[0].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
    axes[0].set_xlabel('Baseline Fitness')
    axes[0].set_ylabel('Optimized Fitness')
    axes[0].set_title('Baseline vs Optimized\n(green=opt wins, red=baseline wins)')
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    
    # Plot 2: Improvement distribution
    improvements = [r['improvement'] for r in results]
    axes[1].hist(improvements, bins=15, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
    axes[1].axvline(np.mean(improvements), color='green', linestyle='-', linewidth=2,
                    label=f'Mean: {np.mean(improvements):.4f}')
    axes[1].set_xlabel('Improvement (Optimized - Baseline)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Improvements')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/validation_plots.png", dpi=150)
    plt.show()
