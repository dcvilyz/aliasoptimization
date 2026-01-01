"""
Alias Optimization for SAM3

A system for finding optimal token sequences (aliases) that maximize
SAM3's mask reconstruction fidelity for pre-annotated datasets.

The key insight is that SAM3's concept mode (one prompt → all instances)
can be used to compress annotations: find one token sequence that produces
all masks in a category.

Main Components:
- config: Configuration management
- data_loader: COCO format dataset loading
- embeddings: Visual/text embedding extraction
- metrics: Evaluation metrics (IoU, recall, etc.)
- soft_prompt: Gradient-based soft prompt optimization
- discrete_search: Evolutionary/local search in token space
- optimizer: Main optimization orchestrator
"""

__version__ = "0.1.0"

from .config import Config, get_config
from .data_loader import COCODataset, CategoryData
from .optimizer import AliasOptimizer, OptimizationResult
from .metrics import MetricResult, compute_combined_score

__all__ = [
    "Config",
    "get_config",
    "COCODataset",
    "CategoryData",
    "AliasOptimizer",
    "OptimizationResult",
    "MetricResult",
    "compute_combined_score",
]
