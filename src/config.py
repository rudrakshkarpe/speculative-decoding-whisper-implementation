"""Configuration for Speculative Decoding"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.
    
    Attributes:
        gamma: Number of tokens to speculatively decode per iteration (default: 4)
        acceptance_threshold: Probability threshold for accepting draft tokens (default: 0.8)
        max_iterations: Maximum number of speculative iterations (default: 100)
        temperature: Sampling temperature (default: 0.0 for greedy)
        use_adaptive_gamma: Whether to adaptively adjust gamma based on acceptance rate (default: True)
        min_gamma: Minimum gamma value when using adaptive gamma (default: 2)
        max_gamma: Maximum gamma value when using adaptive gamma (default: 8)
        acceptance_window: Window size for tracking acceptance rate (default: 10)
    """
    
    gamma: int = 4
    acceptance_threshold: float = 0.8
    max_iterations: int = 100
    temperature: float = 0.0
    use_adaptive_gamma: bool = True
    min_gamma: int = 2
    max_gamma: int = 8
    acceptance_window: int = 10
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.gamma < 1:
            raise ValueError(f"gamma must be >= 1, got {self.gamma}")
        if not 0.0 <= self.acceptance_threshold <= 1.0:
            raise ValueError(
                f"acceptance_threshold must be in [0, 1], got {self.acceptance_threshold}"
            )
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.use_adaptive_gamma:
            if self.min_gamma < 1 or self.min_gamma > self.max_gamma:
                raise ValueError(
                    f"min_gamma must be in [1, max_gamma], got {self.min_gamma}"
                )
            if self.max_gamma < self.min_gamma:
                raise ValueError(
                    f"max_gamma must be >= min_gamma, got {self.max_gamma}"
                )
