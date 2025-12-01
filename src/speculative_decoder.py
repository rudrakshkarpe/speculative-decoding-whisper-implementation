"""Speculative decoding implementation for Whisper"""

from typing import TYPE_CHECKING, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .config import SpeculativeConfig
from .draft_model import DraftModel

if TYPE_CHECKING:
    from whisper.model import Whisper
    from whisper.tokenizer import Tokenizer


class SpeculativeWhisperDecoder:
    """Speculative decoder for Whisper model.
    
    This implements speculative decoding where a smaller draft model generates
    candidate tokens that are verified in parallel by the target model,
    providing speedup while maintaining the same output distribution.
    """
    
    def __init__(
        self,
        target_model: "Whisper",
        draft_model: DraftModel,
        tokenizer: "Tokenizer",
        config: Optional[SpeculativeConfig] = None,
    ):
        """Initialize the speculative decoder.
        
        Args:
            target_model: The target Whisper model
            draft_model: The draft model for generating candidates
            tokenizer: Whisper tokenizer
            config: Configuration for speculative decoding
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.config = config or SpeculativeConfig()
        
        # Statistics tracking
        self.stats = {
            "total_iterations": 0,
            "total_draft_tokens": 0,
            "total_accepted_tokens": 0,
            "acceptance_rates": deque(maxlen=self.config.acceptance_window),
        }
        
        # Adaptive gamma
        self.current_gamma = self.config.gamma
        
        # KV cache for target model
        self.target_kv_cache = {}
        self.target_hooks = []
    
    def decode(
        self,
        mel: Tensor,
        initial_tokens: Tensor,
        max_length: int,
        eot_token: int,
        draft_mel: Tensor = None,
    ) -> Tuple[Tensor, dict]:
        """Perform speculative decoding.
        
        Args:
            mel: Mel spectrogram for target model [batch_size, n_mels, n_frames]
            initial_tokens: Initial token sequence [batch_size, seq_len]
            max_length: Maximum sequence length
            eot_token: End-of-text token ID
            draft_mel: Mel spectrogram for draft model (if different from target)
            
        Returns:
            Tuple of:
                - Generated tokens [batch_size, seq_len]
                - Statistics dictionary
        """
        batch_size = initial_tokens.shape[0]
        current_tokens = initial_tokens.clone()
        
        # Encode audio with target model
        with torch.no_grad():
            target_audio_features = self.target_model.encoder(mel)
        
        # Encode audio with draft model (for DistilWhisper)
        with torch.no_grad():
            if hasattr(self.draft_model, 'model'):
                # Use separate mel for draft model if provided
                draft_input = draft_mel if draft_mel is not None else mel
                draft_audio_features = self.draft_model.model.encoder(draft_input)
            else:
                # LayerDropout uses same model, so same features
                draft_audio_features = target_audio_features
        
        # Store audio features for use in methods
        self.target_audio_features = target_audio_features
        self.draft_audio_features = draft_audio_features
        
        # Install KV cache for target model
        if not self.target_kv_cache:
            self.target_kv_cache, self.target_hooks = (
                self.target_model.install_kv_cache_hooks()
            )
        
        # Reset draft model cache
        self.draft_model.reset_cache()
        
        # Main decoding loop with repetition detection
        iteration = 0
        repetition_count = 0
        last_text = ""
        
        while (
            current_tokens.shape[1] < max_length
            and iteration < self.config.max_iterations
        ):
            # Check if all sequences have reached EOT
            if (current_tokens[:, -1] == eot_token).all():
                break
            
            # Check for repetitions (early stopping)
            if iteration > 5:  # Only check after a few iterations
                current_text = self.tokenizer.decode(current_tokens[0].tolist()[-10:])
                if current_text == last_text:
                    repetition_count += 1
                    if repetition_count >= 3:
                        print("  Warning: Detected repetition, stopping early")
                        break
                else:
                    repetition_count = 0
                last_text = current_text
            
            # Step 1: Generate draft tokens
            gamma = self._get_current_gamma()
            draft_tokens = self.draft_model.generate_draft(
                current_tokens,
                draft_audio_features,
                n_tokens=gamma,
                temperature=self.config.temperature,
            )
            
            # Step 2: Verify draft tokens with target model
            # Clear KV cache before verification to avoid offset issues
            self.target_kv_cache.clear()
            
            accepted_tokens, n_accepted = self._verify_draft(
                current_tokens,
                draft_tokens,
                target_audio_features,
                draft_audio_features,
                eot_token,
            )
            
            # Step 3: Update current tokens
            current_tokens = torch.cat([current_tokens, accepted_tokens], dim=1)
            
            # Step 4: Update statistics
            self._update_stats(gamma, n_accepted)
            
            # Step 5: Adjust gamma if using adaptive mode
            if self.config.use_adaptive_gamma:
                self._adjust_gamma()
            
            iteration += 1
        
        # Cleanup
        self._cleanup()
        
        # Prepare statistics
        stats = self._get_stats()
        
        return current_tokens, stats
    
    def _verify_draft(
        self,
        current_tokens: Tensor,
        draft_tokens: Tensor,
        target_audio_features: Tensor,
        draft_audio_features: Tensor,
        eot_token: int,
    ) -> Tuple[Tensor, int]:
        """Verify draft tokens using the target model.
        
        Args:
            current_tokens: Current token sequence [batch_size, seq_len]
            draft_tokens: Draft tokens to verify [batch_size, gamma]
            target_audio_features: Encoded audio features from target model
            draft_audio_features: Encoded audio features from draft model
            eot_token: End-of-text token ID
            
        Returns:
            Tuple of:
                - Accepted tokens [batch_size, n_accepted]
                - Number of accepted tokens
        """
        batch_size = current_tokens.shape[0]
        gamma = draft_tokens.shape[1]
        
        # Concatenate current tokens with draft tokens
        candidate_tokens = torch.cat([current_tokens, draft_tokens], dim=1)
        
        # Get target model logits for all candidate tokens in parallel
        with torch.no_grad():
            target_logits = self._get_target_logits(
                candidate_tokens, target_audio_features
            )
        
        # Extract logits for the positions where we need to verify
        # target_logits shape: [batch_size, seq_len, vocab_size]
        verify_logits = target_logits[:, -gamma - 1 : -1, :]
        
        # Get probabilities
        target_probs = F.softmax(verify_logits, dim=-1)
        
        # Get draft model probabilities (for rejection sampling)
        draft_candidate_tokens = torch.cat(
            [current_tokens[:, -1:], draft_tokens[:, :-1]], dim=1
        )
        with torch.no_grad():
            draft_logits = self.draft_model.get_logits(
                torch.cat([current_tokens, draft_candidate_tokens], dim=1),
                draft_audio_features,
            )
        draft_verify_logits = draft_logits[:, -gamma:, :]
        draft_probs = F.softmax(draft_verify_logits, dim=-1)
        
        # Handle vocabulary size mismatch between models
        if draft_probs.shape[-1] != target_probs.shape[-1]:
            # Pad smaller vocabulary to match larger one
            vocab_diff = target_probs.shape[-1] - draft_probs.shape[-1]
            if vocab_diff > 0:
                # Draft model has smaller vocab, pad with zeros
                padding = torch.zeros(
                    draft_probs.shape[0], draft_probs.shape[1], vocab_diff,
                    device=draft_probs.device, dtype=draft_probs.dtype
                )
                draft_probs = torch.cat([draft_probs, padding], dim=-1)
            else:
                # Target model has smaller vocab (shouldn't happen but handle it)
                padding = torch.zeros(
                    target_probs.shape[0], target_probs.shape[1], -vocab_diff,
                    device=target_probs.device, dtype=target_probs.dtype
                )
                target_probs = torch.cat([target_probs, padding], dim=-1)
        
        # Verify each draft token using rejection sampling
        accepted_tokens_list = []
        n_accepted = 0
        
        for i in range(gamma):
            # Get probabilities for the draft token at position i
            draft_token = draft_tokens[:, i]
            target_prob = target_probs[
                torch.arange(batch_size), i, draft_token
            ]
            draft_prob = draft_probs[
                torch.arange(batch_size), i, draft_token
            ]
            
            # Rejection sampling: accept with probability min(1, p_target / p_draft)
            acceptance_prob = torch.minimum(
                torch.ones_like(target_prob),
                target_prob / (draft_prob + 1e-10),
            )
            
            # Check if we should accept (for simplicity, using threshold)
            should_accept = acceptance_prob >= self.config.acceptance_threshold
            
            # Also check if the token is EOT
            is_eot = draft_token == eot_token
            
            # Accept if probability is high enough or if it's EOT
            if should_accept.all() or is_eot.any():
                accepted_tokens_list.append(draft_token.unsqueeze(1))
                n_accepted += 1
                
                # If EOT is encountered, stop accepting
                if is_eot.any():
                    break
            else:
                # Rejection: sample from adjusted distribution
                # p'(x) = max(0, p_target(x) - p_draft(x)) / Z
                adjusted_probs = torch.maximum(
                    torch.zeros_like(target_probs[:, i, :]),
                    target_probs[:, i, :] - draft_probs[:, i, :],
                )
                adjusted_probs = adjusted_probs / (
                    adjusted_probs.sum(dim=-1, keepdim=True) + 1e-10
                )
                
                # Sample from adjusted distribution
                if self.config.temperature == 0:
                    sampled_token = adjusted_probs.argmax(dim=-1, keepdim=True)
                else:
                    sampled_token = torch.multinomial(
                        adjusted_probs, num_samples=1
                    )
                
                accepted_tokens_list.append(sampled_token)
                n_accepted += 1
                break  # Stop after first rejection and resampling
        
        # If no tokens were accepted, sample one token from target distribution
        if n_accepted == 0:
            last_target_probs = target_probs[:, -1, :]
            if self.config.temperature == 0:
                sampled_token = last_target_probs.argmax(dim=-1, keepdim=True)
            else:
                sampled_token = torch.multinomial(
                    last_target_probs, num_samples=1
                )
            accepted_tokens_list.append(sampled_token)
            n_accepted = 1
        
        # Concatenate accepted tokens
        accepted_tokens = torch.cat(accepted_tokens_list, dim=1)
        
        return accepted_tokens, n_accepted
    
    def _get_target_logits(
        self, tokens: Tensor, audio_features: Tensor
    ) -> Tensor:
        """Get logits from target model with KV caching.
        
        Args:
            tokens: Token sequence [batch_size, seq_len]
            audio_features: Encoded audio features
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # For speculative decoding, we need to verify multiple tokens at once
        # This doesn't work well with KV cache, so we disable it for verification
        # TODO: Implement proper KV cache management for speculative decoding
        logits = self.target_model.decoder(
            tokens, audio_features, kv_cache=None
        )
        
        return logits
    
    def _get_current_gamma(self) -> int:
        """Get the current gamma value (may be adaptive)."""
        if self.config.use_adaptive_gamma:
            return self.current_gamma
        else:
            return self.config.gamma
    
    def _adjust_gamma(self):
        """Adjust gamma based on recent acceptance rate."""
        if len(self.stats["acceptance_rates"]) < self.config.acceptance_window // 2:
            return  # Not enough data yet
        
        # Calculate average acceptance rate
        avg_acceptance = np.mean(self.stats["acceptance_rates"])
        
        # Adjust gamma based on acceptance rate
        if avg_acceptance > 0.8:
            # High acceptance: increase gamma
            self.current_gamma = min(
                self.current_gamma + 1, self.config.max_gamma
            )
        elif avg_acceptance < 0.5:
            # Low acceptance: decrease gamma
            self.current_gamma = max(
                self.current_gamma - 1, self.config.min_gamma
            )
    
    def _update_stats(self, gamma: int, n_accepted: int):
        """Update decoding statistics.
        
        Args:
            gamma: Number of draft tokens generated
            n_accepted: Number of tokens accepted
        """
        self.stats["total_iterations"] += 1
        self.stats["total_draft_tokens"] += gamma
        self.stats["total_accepted_tokens"] += n_accepted
        
        # Track acceptance rate
        acceptance_rate = n_accepted / gamma if gamma > 0 else 0
        self.stats["acceptance_rates"].append(acceptance_rate)
    
    def _get_stats(self) -> dict:
        """Get current statistics."""
        total_draft = self.stats["total_draft_tokens"]
        total_accepted = self.stats["total_accepted_tokens"]
        
        return {
            "total_iterations": self.stats["total_iterations"],
            "total_draft_tokens": total_draft,
            "total_accepted_tokens": total_accepted,
            "overall_acceptance_rate": (
                total_accepted / total_draft if total_draft > 0 else 0
            ),
            "avg_tokens_per_iteration": (
                total_accepted / self.stats["total_iterations"]
                if self.stats["total_iterations"] > 0
                else 0
            ),
            "final_gamma": self.current_gamma,
        }
    
    def _cleanup(self):
        """Clean up resources after decoding."""
        # Remove target model hooks
        for hook in self.target_hooks:
            hook.remove()
        
        # Clear caches
        self.target_kv_cache = {}
        self.target_hooks = []
        
        # Reset draft model
        self.draft_model.reset_cache()
    
    def reset_stats(self):
        """Reset decoding statistics."""
        self.stats = {
            "total_iterations": 0,
            "total_draft_tokens": 0,
            "total_accepted_tokens": 0,
            "acceptance_rates": deque(maxlen=self.config.acceptance_window),
        }
        self.current_gamma = self.config.gamma
