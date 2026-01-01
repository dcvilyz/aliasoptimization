# Alias Optimization for SAM3

## Problem Statement

Given a dataset with images and masks (no class names), find the optimal token sequence T* such that:

```
T* = argmin_T Σ_images [ Loss(SAM3(image, T), target_masks) ]
```

Where T is a sequence of BPE tokens from SAM3's vocabulary (~49K tokens).

## Key Insight

SAM3's text encoder works on **BPE tokens**, not words. We can optimize directly in token space
without caring about human interpretability. The goal is pure mask reconstruction fidelity.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ALIAS OPTIMIZATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────────────────────┐  │
│  │   Images    │     │    Masks     │     │   SAM3 Model (frozen)       │  │
│  │  + Masks    │────▶│   Features   │────▶│   - Vision Encoder          │  │
│  │  (COCO)     │     │  Extraction  │     │   - Text Encoder            │  │
│  └─────────────┘     └──────────────┘     │   - Detector                │  │
│                             │              └─────────────────────────────┘  │
│                             ▼                           │                    │
│                      ┌──────────────┐                   │                    │
│                      │   Visual     │                   │                    │
│                      │  Concept     │                   │                    │
│                      │  Embedding   │                   │                    │
│                      └──────────────┘                   │                    │
│                             │                           │                    │
│                             ▼                           │                    │
│  ┌─────────────────────────────────────────────────────┼──────────────────┐ │
│  │                    OPTIMIZATION LOOP                 │                  │ │
│  │                                                      │                  │ │
│  │   ┌────────────┐    ┌────────────┐    ┌────────────┐│                  │ │
│  │   │  Soft      │    │  Gradient  │    │  Token     ││   ┌──────────┐  │ │
│  │   │  Prompt    │───▶│  Descent   │───▶│  Projection│├──▶│  SAM3    │  │ │
│  │   │  (cont.)   │    │            │    │  (discrete)││   │  Forward │  │ │
│  │   └────────────┘    └────────────┘    └────────────┘│   └──────────┘  │ │
│  │         ▲                                     │      │        │        │ │
│  │         │                                     │      │        ▼        │ │
│  │         │           ┌────────────┐            │      │   ┌──────────┐  │ │
│  │         └───────────│   Loss     │◀───────────┼──────┼───│  Pred    │  │ │
│  │                     │ (IoU etc)  │            │      │   │  Masks   │  │ │
│  │                     └────────────┘            │      │   └──────────┘  │ │
│  │                           ▲                   │      │                  │ │
│  │                           │                   ▼      │                  │ │
│  │                     ┌────────────┐    ┌────────────┐ │                  │ │
│  │                     │  Target    │    │ Discrete   │ │                  │ │
│  │                     │  Masks     │    │ Search     │ │                  │ │
│  │                     └────────────┘    │(Evolution) │ │                  │ │
│  │                                       └────────────┘ │                  │ │
│  └──────────────────────────────────────────────────────┘──────────────────┘ │
│                                                                             │
│  Output: Optimal token sequence T* = [t1, t2, ..., tn]                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Approach

### Phase 1: Visual Concept Embedding
Extract features from masked regions using SAM3's vision encoder.
Aggregate across all instances of a class to get a "concept center" in embedding space.

### Phase 2: Vocabulary Embedding Index  
Pre-compute embeddings for all ~49K tokens in SAM3's vocabulary.
Build a searchable index for nearest-neighbor queries.

### Phase 3: Soft Prompt Optimization
Initialize soft (continuous) prompt embeddings near the visual concept.
Optimize via gradient descent to minimize mask reconstruction loss.
This requires injecting soft embeddings into SAM3's text encoder pathway.

### Phase 4: Discrete Projection
Project optimized soft embeddings to nearest real tokens.
This is lossy but gives us a valid token sequence.

### Phase 5: Discrete Refinement
Local search around projected solution (swap tokens, try neighbors).
Evolutionary refinement with population-based search.

## Key Technical Details

### Injection Point
From `text_encoder_ve.py`, the text encoder flow is:
```python
tokenized = tokenizer(text)           # [batch, seq_len] token IDs
inputs_embeds = encoder.token_embedding(tokenized)  # [batch, seq_len, 1024]
inputs_embeds = inputs_embeds + positional_embedding
output = transformer(inputs_embeds)   # [batch, seq_len, 1024]  
output = resizer(output)              # [batch, seq_len, d_model]
```

We can inject at `inputs_embeds` level - replace token embeddings with our learned soft embeddings.

### Vocabulary Structure
- BPE tokenizer with ~49,408 tokens
- Includes subword units, not full words
- Special tokens: SOT (start), EOT (end)
- Context length: 32 tokens max

### Loss Functions
- **Mask IoU**: Intersection over Union for mask quality
- **Instance Recall**: Did we find all instances?
- **Presence Score**: SAM3's own confidence measure

## Files

- `config.py` - Configuration and hyperparameters
- `data_loader.py` - COCO format dataset loading
- `embeddings.py` - Visual/text embedding extraction
- `soft_prompt.py` - Soft prompt optimization module
- `discrete_search.py` - Token space search algorithms
- `metrics.py` - Evaluation metrics (IoU, recall, etc.)
- `optimizer.py` - Main optimization loop
- `main.py` - Entry point

## Usage

```bash
# Run optimization for a dataset
python main.py --dataset /path/to/coco/dataset --output results/

# Evaluate discovered aliases
python evaluate.py --aliases results/aliases.json --dataset /path/to/dataset
```

## Requirements

- PyTorch with MPS support (Apple Silicon)
- SAM3 model weights
- COCO format dataset with masks
