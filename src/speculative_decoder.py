"""Speculative Decoding for Whisper

This implements true speculative decoding with parallel token verification:
1. Generate draft tokens quickly with Whisper Tiny
2. Verify draft tokens in parallel with Whisper Large V3
3. Accept/reject using rejection sampling
4. Only continue where needed
"""

from typing import TYPE_CHECKING, Tuple, Optional
from collections import deque
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import SpeculativeConfig
from .draft_model import DraftModel

if TYPE_CHECKING:
    from whisper.model import Whisper
    from whisper.tokenizer import Tokenizer


class SpeculativeWhisperDecoder:
    """Speculative decoder for Whisper with parallel token verification.
    
    This implements the true speculative decoding algorithm:
    - Stage 1: Fast draft generation with smaller model
    - Stage 2: Parallel verification with target model
    - Stage 3: Rejection sampling for unmatched tokens
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
            target_model: The target Whisper model (e.g., Large V3)
            draft_model: The draft model for fast generation (e.g., Tiny)
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
        
        self.current_gamma = self.config.gamma
    
    def decode(
        self,
        mel: Tensor,
        initial_tokens: Tensor,
        max_length: int,
        eot_token: int,
        draft_mel: Tensor = None,
    ) -> Tuple[Tensor, dict]:
        """Perform speculative decoding with parallel verification.
        
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
        
        # Encode audio features once
        with torch.no_grad():
            target_audio_features = self.target_model.encoder(mel)
            
            if hasattr(self.draft_model, 'model'):
                draft_input = draft_mel if draft_mel is not None else mel
                draft_audio_features = self.draft_model.model.encoder(draft_input)
            else:
                draft_audio_features = target_audio_features
        
        # Main speculative decoding loop
        iteration = 0
        while current_tokens.shape[1] < max_length and iteration < self.config.max_iterations:
            # Check if EOT reached
            if (current_tokens[:, -1] == eot_token).all():
                break
            
            # Step 1: Generate draft tokens (gamma tokens)
            gamma = self._get_current_gamma()
            draft_tokens = self._generate_draft_tokens(
                current_tokens,
                draft_audio_features,
                gamma,
                eot_token
            )
            
            # Step 2: Verify draft tokens with target model in parallel
            accepted_tokens, n_accepted = self._verify_draft_parallel(
                current_tokens,
                draft_tokens,
                target_audio_features,
                draft_audio_features,
                eot_token
            )
            
            # Step 3: Update sequence
            current_tokens = torch.cat([current_tokens, accepted_tokens], dim=1)
            
            # Step 4: Update statistics
            self._update_stats(gamma, n_accepted)
            
            # Step 5: Adjust gamma if adaptive
            if self.config.use_adaptive_gamma:
                self._adjust_gamma()
            
            iteration += 1
        
        # Get final statistics
        stats = self._get_stats()
        
        return current_tokens, stats
    
    def _generate_draft_tokens(
        self,
        current_tokens: Tensor,
        audio_features: Tensor,
        n_tokens: int,
        eot_token: int,
    ) -> Tensor:
        """Generate draft tokens with the draft model.
        
        Args:
            current_tokens: Current sequence [batch_size, seq_len]
            audio_features: Audio features for draft model
            n_tokens: Number of tokens to generate
            eot_token: End-of-text token
            
        Returns:
            Draft tokens [batch_size, n_generated]
        """
        draft_tokens = []
        tokens = current_tokens.clone()
        
        with torch.no_grad():
            for _ in range(n_tokens):
                # Get logits from draft model
                if hasattr(self.draft_model, 'model'):
                    logits = self.draft_model.model.decoder(
                        tokens, audio_features, kv_cache=None
                    )
                else:
                    logits = self.draft_model.get_logits(tokens, audio_features)
                
                # Greedy sampling
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                draft_tokens.append(next_token)
                tokens = torch.cat([tokens, next_token], dim=1)
                
                # Stop if EOT
                if (next_token == eot_token).all():
                    break
        
        return torch.cat(draft_tokens, dim=1) if draft_tokens else torch.empty(current_tokens.shape[0], 0, dtype=torch.long, device=current_tokens.device)
    
    def _verify_draft_parallel(
        self,
        current_tokens: Tensor,
        draft_tokens: Tensor,
        target_audio_features: Tensor,
        draft_audio_features: Tensor,
        eot_token: int,
    ) -> Tuple[Tensor, int]:
        """Verify draft tokens using target model with parallel processing.
        
        This is the KEY optimization: process all draft tokens in ONE forward pass!
        
        Args:
            current_tokens: Current sequence [batch_size, seq_len]
            draft_tokens: Draft tokens to verify [batch_size, gamma]
            target_audio_features: Audio features for target model
            draft_audio_features: Audio features for draft model
            eot_token: End-of-text token
            
        Returns:
            Tuple of (accepted_tokens, n_accepted)
        """
        if draft_tokens.shape[1] == 0:
            # No draft tokens, generate one with target
            with torch.no_grad():
                logits = self.target_model.decoder(
                    current_tokens, target_audio_features, kv_cache=None
                )
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            return next_token, 1
        
        batch_size = current_tokens.shape[0]
        gamma = draft_tokens.shape[1]
        
        # Concatenate for parallel processing
        candidate_tokens = torch.cat([current_tokens, draft_tokens], dim=1)
        
        with torch.no_grad():
            # PARALLEL VERIFICATION: Process entire sequence at once!
            target_logits = self.target_model.decoder(
                candidate_tokens, target_audio_features, kv_cache=None
            )
            
            # Get target probabilities for verification positions
            verify_logits = target_logits[:, -gamma-1:-1, :]
            target_probs = F.softmax(verify_logits, dim=-1)
            
            # Get draft probabilities
            draft_candidate = torch.cat([current_tokens[:, -1:], draft_tokens[:, :-1]], dim=1)
            full_draft_seq = torch.cat([current_tokens, draft_candidate], dim=1)
            
            draft_logits = self._get_draft_logits(full_draft_seq, draft_audio_features)
            draft_verify_logits = draft_logits[:, -gamma:, :]
            draft_probs = F.softmax(draft_verify_logits, dim=-1)
            
            # Handle vocabulary mismatch
            if draft_probs.shape[-1] != target_probs.shape[-1]:
                vocab_diff = target_probs.shape[-1] - draft_probs.shape[-1]
                if vocab_diff > 0:
                    padding = torch.zeros(
                        draft_probs.shape[0], draft_probs.shape[1], vocab_diff,
                        device=draft_probs.device, dtype=draft_probs.dtype
                    )
                    draft_probs = torch.cat([draft_probs, padding], dim=-1)
        
        # Token-by-token verification with rejection sampling
        accepted_tokens_list = []
        n_accepted = 0
        
        for i in range(gamma):
            draft_token = draft_tokens[:, i]
            
            # Get probabilities for this token
            p_target = target_probs[torch.arange(batch_size), i, draft_token]
            p_draft = draft_probs[torch.arange(batch_size), i, draft_token]
            
            # Rejection sampling: accept with probability min(1, p_target/p_draft)
            acceptance_ratio = torch.minimum(
                torch.ones_like(p_target),
                p_target / (p_draft + 1e-10)
            )
            
            # Accept if ratio >= threshold or if EOT
            should_accept = acceptance_ratio >= self.config.acceptance_threshold
            is_eot = draft_token == eot_token
            
            if should_accept.all() or is_eot.any():
                accepted_tokens_list.append(draft_token.unsqueeze(1))
                n_accepted += 1
                if is_eot.any():
                    break
            else:
                # Rejection: sample from adjusted distribution
                adjusted_probs = torch.maximum(
                    torch.zeros_like(target_probs[:, i, :]),
                    target_probs[:, i, :] - draft_probs[:, i, :]
                )
                adjusted_probs = adjusted_probs / (adjusted_probs.sum(dim=-1, keepdim=True) + 1e-10)
                
                # Sample new token
                sampled_token = adjusted_probs.argmax(dim=-1, keepdim=True)
                accepted_tokens_list.append(sampled_token)
                n_accepted += 1
                break  # Stop after first rejection
        
        # If nothing accepted, sample from target
        if n_accepted == 0:
            last_logits = target_logits[:, -1, :]
            next_token = last_logits.argmax(dim=-1, keepdim=True)
            return next_token, 1
        
        return torch.cat(accepted_tokens_list, dim=1), n_accepted
    
    def _get_draft_logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        """Get logits from draft model."""
        with torch.no_grad():
            if hasattr(self.draft_model, 'model'):
                return self.draft_model.model.decoder(tokens, audio_features, kv_cache=None)
            else:
                return self.draft_model.get_logits(tokens, audio_features)
    
    def _get_current_gamma(self) -> int:
        """Get current gamma value."""
        return self.current_gamma if self.config.use_adaptive_gamma else self.config.gamma
    
    def _adjust_gamma(self):
        """Adjust gamma based on acceptance rate."""
        if len(self.stats["acceptance_rates"]) < self.config.acceptance_window // 2:
            return
        
        avg_acceptance = sum(self.stats["acceptance_rates"]) / len(self.stats["acceptance_rates"])
        
        if avg_acceptance > 0.8:
            self.current_gamma = min(self.current_gamma + 1, self.config.max_gamma)
        elif avg_acceptance < 0.5:
            self.current_gamma = max(self.current_gamma - 1, self.config.min_gamma)
    
    def _update_stats(self, gamma: int, n_accepted: int):
        """Update statistics."""
        self.stats["total_iterations"] += 1
        self.stats["total_draft_tokens"] += gamma
        self.stats["total_accepted_tokens"] += n_accepted
        
        acceptance_rate = n_accepted / gamma if gamma > 0 else 0
        self.stats["acceptance_rates"].append(acceptance_rate)
    
    def _get_stats(self) -> dict:
        """Get final statistics."""
        total_draft = self.stats["total_draft_tokens"]
        total_accepted = self.stats["total_accepted_tokens"]
        
        return {
            "total_iterations": self.stats["total_iterations"],
            "total_draft_tokens": total_draft,
            "total_accepted_tokens": total_accepted,
            "overall_acceptance_rate": total_accepted / total_draft if total_draft > 0 else 0,
            "avg_tokens_per_iteration": total_accepted / self.stats["total_iterations"] if self.stats["total_iterations"] > 0 else 0,
            "final_gamma": self.current_gamma,
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_iterations": 0,
            "total_draft_tokens": 0,
            "total_accepted_tokens": 0,
            "acceptance_rates": deque(maxlen=self.config.acceptance_window),
        }
        self.current_gamma = self.config.gamma
