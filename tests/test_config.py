"""Tests for configuration"""

import pytest
import sys
sys.path.insert(0, "../")

from src.config import SpeculativeConfig


def test_default_config():
    """Test default configuration values."""
    config = SpeculativeConfig()
    
    assert config.gamma == 4
    assert config.acceptance_threshold == 0.8
    assert config.max_iterations == 100
    assert config.temperature == 0.0
    assert config.use_adaptive_gamma is True
    assert config.min_gamma == 2
    assert config.max_gamma == 8
    assert config.acceptance_window == 10


def test_custom_config():
    """Test custom configuration values."""
    config = SpeculativeConfig(
        gamma=6,
        acceptance_threshold=0.9,
        max_iterations=50,
        temperature=0.5,
    )
    
    assert config.gamma == 6
    assert config.acceptance_threshold == 0.9
    assert config.max_iterations == 50
    assert config.temperature == 0.5


def test_invalid_gamma():
    """Test that invalid gamma raises error."""
    with pytest.raises(ValueError, match="gamma must be >= 1"):
        SpeculativeConfig(gamma=0)


def test_invalid_acceptance_threshold():
    """Test that invalid acceptance threshold raises error."""
    with pytest.raises(ValueError, match="acceptance_threshold must be in"):
        SpeculativeConfig(acceptance_threshold=1.5)
    
    with pytest.raises(ValueError, match="acceptance_threshold must be in"):
        SpeculativeConfig(acceptance_threshold=-0.1)


def test_invalid_max_iterations():
    """Test that invalid max_iterations raises error."""
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        SpeculativeConfig(max_iterations=0)


def test_invalid_temperature():
    """Test that invalid temperature raises error."""
    with pytest.raises(ValueError, match="temperature must be >= 0"):
        SpeculativeConfig(temperature=-1.0)


def test_invalid_adaptive_gamma_range():
    """Test that invalid adaptive gamma range raises error."""
    with pytest.raises(ValueError, match="min_gamma must be in"):
        SpeculativeConfig(min_gamma=0, use_adaptive_gamma=True)
    
    with pytest.raises(ValueError, match="min_gamma must be in"):
        SpeculativeConfig(min_gamma=10, max_gamma=8, use_adaptive_gamma=True)
    
    with pytest.raises(ValueError, match="max_gamma must be >= min_gamma"):
        SpeculativeConfig(min_gamma=5, max_gamma=3, use_adaptive_gamma=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
