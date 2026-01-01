"""
Configuration for Alias Optimization.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_path: str = ""
    train_annotations: str = "_annotations.coco.json"
    valid_annotations: str = "_annotations.coco.json"
    test_annotations: str = "_annotations.coco.json"
    image_size: int = 1008  # SAM3 default
    
    
@dataclass
class ModelConfig:
    """SAM3 model configuration."""
    # These will be populated from actual SAM3 model
    vocab_size: int = 49408  # BPE vocabulary size
    context_length: int = 32  # Max sequence length
    embed_dim: int = 1024  # Text encoder hidden dimension
    model_dim: int = 256  # SAM3 internal dimension (after resizer)
    
    # Special tokens
    sot_token_id: int = 49406  # <start_of_text>
    eot_token_id: int = 49407  # <end_of_text>


@dataclass  
class OptimizationConfig:
    """Optimization hyperparameters."""
    
    # Soft prompt optimization
    soft_prompt_length: int = 7  # Number of tokens to optimize
    soft_lr: float = 0.15  # Learning rate for soft prompts (increased)
    soft_steps: int = 700  # Gradient descent steps (increased from 200)
    soft_batch_size: int = 4  # Images per batch during soft optimization
    
    # Discrete search - REDUCED FOR SPEED
    discrete_top_k: int = 10  # Top-k tokens to consider per position (was 50)
    discrete_beam_size: int = 5  # Beam search width (was 10)
    discrete_iterations: int = 25  # Local search iterations (was 100)
    
    # Evolutionary refinement - REDUCED FOR SPEED
    population_size: int = 20  # (was 50)
    evolution_generations: int = 20  # (was 100)
    mutation_rate: float = 0.2
    crossover_rate: float = 0.5
    tournament_size: int = 3  # (was 5)
    elite_size: int = 3  # (was 5)
    
    # Early stopping
    patience: int = 20  # Generations without improvement
    min_improvement: float = 0.001  # Minimum IoU improvement to count
    
    # Evaluation
    eval_batch_size: int = 8
    iou_threshold: float = 0.5  # For instance matching
    confidence_threshold: float = 0.5  # SAM3 prediction threshold


@dataclass
class EmbeddingConfig:
    """Embedding extraction configuration."""
    
    # Visual concept extraction
    pool_type: str = "masked_mean"  # How to pool features over mask
    aggregation: str = "mean"  # How to aggregate across instances
    
    # Vocabulary embedding
    batch_size: int = 256  # Tokens per batch when building vocab embeddings
    use_cache: bool = True  # Cache vocabulary embeddings to disk
    cache_path: str = "cache/vocab_embeddings.pt"


@dataclass
class MetricConfig:
    """Metrics configuration."""
    
    # Primary metrics
    primary_metric: str = "mean_iou"  # Main metric for optimization
    
    # IoU settings
    iou_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])
    
    # Instance matching
    match_iou_threshold: float = 0.5
    
    # Weights for combined metric
    iou_weight: float = 0.5
    recall_weight: float = 0.3
    precision_weight: float = 0.2


@dataclass
class Config:
    """Master configuration."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    
    # General
    device: str = "mps"  # Apple Silicon
    seed: int = 42
    output_dir: str = "results"
    verbose: bool = True
    
    # Logging
    log_every: int = 10  # Log every N generations/steps
    save_checkpoints: bool = True
    checkpoint_every: int = 50
    
    def __post_init__(self):
        """Create output directory if needed."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.embedding.use_cache:
            Path(self.embedding.cache_path).parent.mkdir(parents=True, exist_ok=True)


def get_config(dataset_path: str = None, **kwargs) -> Config:
    """Get configuration with optional overrides."""
    config = Config()
    
    if dataset_path:
        config.data.dataset_path = dataset_path
        
    # Apply any additional overrides
    for key, value in kwargs.items():
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        
    return config