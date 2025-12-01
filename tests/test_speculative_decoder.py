"""Tests for speculative decoder"""

import pytest
import sys
sys.path.insert(0, "../")

import torch
from unittest.mock import Mock, MagicMock, patch

from src.speculative_decoder import SpeculativeWhisperDecoder
from src.draft_model import DraftModel
from src.config import SpeculativeConfig


class MockDraftModel(DraftModel):
    """Mock draft model for testing."""
    
    def __init__(self):
        self.reset_cache_called = False
        
    def generate_draft(self, tokens, audio_features, n_tokens, temperature=0.0):
        """Generate mock draft tokens."""
        batch_size = tokens.shape[0]
        # Return sequential tokens for testing
        return torch.arange(1, n_tokens + 1).unsqueeze(0).repeat(batch_size, 1)
    
    def get_logits(self, tokens, audio_features):
        """Return mock logits."""
        batch_size = tokens.shape[0]
        seq_len = tokens.shape[1]
        vocab_size = 1000
        return torch.randn(batch_size, seq_len, vocab_size)
    
    def reset_cache(self):
        """Reset cache."""
        self.reset_cache_called = True


def test_decoder_initialization():
    """Test decoder initialization."""
    mock_target = MagicMock()
    mock_draft = MockDraftModel()
    mock_tokenizer = MagicMock()
    config = SpeculativeConfig()
    
    decoder = SpeculativeWhisperDecoder(
        target_model=mock_target,
        draft_model=mock_draft,
        tokenizer=mock_tokenizer,
        config=config,
    )
    
    assert decoder.target_model == mock_target
    assert decoder.draft_model == mock_draft
    assert decoder.tokenizer == mock_tokenizer
    assert decoder.config == config
    assert decoder.current_gamma == config.gamma


def test_stats_initialization():
    """Test that statistics are properly initialized."""
    mock_target = MagicMock()
    mock_draft = MockDraftModel()
    mock_tokenizer = MagicMock()
    
    decoder = SpeculativeWhisperDecoder(
        target_model=mock_target,
        draft_model=mock_draft,
        tokenizer=mock_tokenizer,
    )
    
    assert decoder.stats["total_iterations"] == 0
    assert decoder.stats["total_draft_tokens"] == 0
    assert decoder.stats["total_accepted_tokens"] == 0
    assert len(decoder.stats["acceptance_rates"]) == 0


def test_get_current_gamma():
    """Test getting current gamma value."""
    mock_target = MagicMock()
    mock_draft = MockDraftModel()
    mock_tokenizer = MagicMock()
    
    # Test with adaptive gamma disabled
    config = SpeculativeConfig(gamma=5, use_adaptive_gamma=False)
    decoder = SpeculativeWhisperDecoder(
        target_model=mock_target,
        draft_model=mock_draft,
        tokenizer=mock_tokenizer,
        config=config,
    )
    assert decoder._get_current_gamma() == 5
    
    # Test with adaptive gamma enabled
    config = SpeculativeConfig(gamma=5, use_adaptive_gamma=True)
    decoder = SpeculativeWhisperDecoder(
        target_model=mock_target,
        draft_model=mock_draft,
        tokenizer=mock_tokenizer,
        config=config,
    )
    decoder.current_gamma = 7
    assert decoder._get_current_gamma() == 7


def test_update_stats():
    """Test statistics update."""
    mock_target = MagicMock()
    mock_draft = MockDraftModel()
    mock_tokenizer = MagicMock()
    
    decoder = SpeculativeWhisperDecoder(
        target_model=mock_target,
        draft_model=mock_draft,
        tokenizer=mock_tokenizer,
    )
    
    # Update stats with first iteration
    decoder._update_stats(gamma=4, n_accepted=3)
    
    assert decoder.stats["total_iterations"] == 1
    assert decoder.stats["total_draft_tokens"] == 4
    assert decoder.stats["total_accepted_tokens"] == 3
    assert len(decoder.stats["acceptance_rates"]) == 1
    assert decoder.stats["acceptance_rates"][0] == 0.75
    
    # Update stats with second iteration
    decoder._update_stats(gamma=4, n_accepted=2)
    
    assert decoder.stats["total_iterations"] == 2
    assert decoder.stats["total_draft_tokens"] == 8
    assert decoder.stats["total_accepted_tokens"] == 5
    assert len(decoder.stats["acceptance_rates"]) == 2


def test_adjust_gamma():
    """Test adaptive gamma adjustment."""
    mock_target = MagicMock()
    mock_draft = MockDraftModel()
    mock_tokenizer = MagicMock()
    
    config = SpeculativeConfig(
        gamma=4,
        use_adaptive_gamma=True,
        min_gamma=2,
        max_gamma=8,
    )
    decoder = SpeculativeWhisperDecoder(
        target_model=mock_target,
        draft_model=mock_draft,
        tokenizer=mock_tokenizer,
        config=config,
    )
    
    # Test increasing gamma with high acceptance rate
    for _ in range(10):
        decoder.stats["acceptance_rates"].append(0.9)
    
    initial_gamma = decoder.current_gamma
    decoder._adjust_gamma()
    assert decoder.current_gamma == initial_gamma + 1
    
    # Test decreasing gamma with low acceptance rate
    decoder.stats["acceptance_rates"].clear()
    for _ in range(10):
        decoder.stats["acceptance_rates"].append(0.3)
    
    initial_gamma = decoder.current_gamma
    decoder._adjust_gamma()
    assert decoder.current_gamma == initial_gamma - 1


def test_get_stats():
    """Test getting statistics."""
    mock_target = MagicMock()
    mock_draft = MockDraftModel()
    mock_tokenizer = MagicMock()
    
    decoder = SpeculativeWhisperDecoder(
        target_model=mock_target,
        draft_model=mock_draft,
        tokenizer=mock_tokenizer,
    )
    
    # Add some stats
    decoder._update_stats(gamma=4, n_accepted=3)
    decoder._update_stats(gamma=4, n_accepted=2)
    
    stats = decoder._get_stats()
    
    assert stats["total_iterations"] == 2
    assert stats["total_draft_tokens"] == 8
    assert stats["total_accepted_tokens"] == 5
    assert stats["overall_acceptance_rate"] == 5 / 8
    assert stats["avg_tokens_per_iteration"] == 5 / 2


def test_reset_stats():
    """Test resetting statistics."""
    mock_target = MagicMock()
    mock_draft = MockDraftModel()
    mock_tokenizer = MagicMock()
    
    decoder = SpeculativeWhisperDecoder(
        target_model=mock_target,
        draft_model=mock_draft,
        tokenizer=mock_tokenizer,
    )
    
    # Add some stats
    decoder._update_stats(gamma=4, n_accepted=3)
    decoder.current_gamma = 7
    
    # Reset
    decoder.reset_stats()
    
    assert decoder.stats["total_iterations"] == 0
    assert decoder.stats["total_draft_tokens"] == 0
    assert decoder.stats["total_accepted_tokens"] == 0
    assert len(decoder.stats["acceptance_rates"]) == 0
    assert decoder.current_gamma == decoder.config.gamma


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
