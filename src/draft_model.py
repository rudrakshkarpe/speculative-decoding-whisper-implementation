"""Draft models for speculative decoding"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from whisper.model import Whisper


class DraftModel(ABC):
    """Abstract base class for draft models used in speculative decoding.
    
    A draft model is a smaller, faster model that generates candidate tokens
    which are then verified by the larger target model.
    """
    
    @abstractmethod
    def generate_draft(
        self,
        tokens: Tensor,
        audio_features: Tensor,
        n_tokens: int,
        temperature: float = 0.0,
    ) -> Tensor:
        """Generate draft tokens.
        
        Args:
            tokens: Current token sequence [batch_size, seq_len]
            audio_features: Encoded audio features [batch_size, n_ctx, dim]
            n_tokens: Number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Draft tokens [batch_size, n_tokens]
        """
        pass
    
    @abstractmethod
    def get_logits(
        self,
        tokens: Tensor,
        audio_features: Tensor,
    ) -> Tensor:
        """Get logits for the current tokens.
        
        Args:
            tokens: Token sequence [batch_size, seq_len]
            audio_features: Encoded audio features [batch_size, n_ctx, dim]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        pass
    
    @abstractmethod
    def reset_cache(self):
        """Reset any internal caching state."""
        pass


class DistilWhisperDraft(DraftModel):
    """Draft model using a distilled Whisper model.
    
    This implementation uses a smaller Whisper model (e.g., tiny or base)
    as the draft model for a larger target model (e.g., medium or large).
    """
    
    def __init__(self, draft_model: "Whisper"):
        """Initialize the draft model.
        
        Args:
            draft_model: A smaller Whisper model to use as the draft
        """
        self.model = draft_model
        self.kv_cache = {}
        self.hooks = []
        
    def generate_draft(
        self,
        tokens: Tensor,
        audio_features: Tensor,
        n_tokens: int,
        temperature: float = 0.0,
    ) -> Tensor:
        """Generate draft tokens using the smaller model.
        
        Args:
            tokens: Current token sequence [batch_size, seq_len]
            audio_features: Encoded audio features from TARGET model [batch_size, n_ctx, dim]
                           Note: We need to re-encode with draft model's encoder
            n_tokens: Number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Draft tokens [batch_size, n_tokens]
        """
        draft_tokens = []
        current_tokens = tokens.clone()
        
        # For simplicity and correctness, we'll generate tokens autoregressively
        # without KV caching in the draft stage. This is still fast since
        # the draft model is much smaller than the target model.
        
        with torch.no_grad():
            for _ in range(n_tokens):
                # Get logits for all tokens (no KV cache for draft generation)
                logits = self.model.decoder(
                    current_tokens, audio_features, kv_cache=None
                )
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :]
                
                # Sample next token
                if temperature == 0:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                draft_tokens.append(next_token)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        # Stack draft tokens
        draft_tokens = torch.cat(draft_tokens, dim=1)
        return draft_tokens
    
    def get_logits(
        self,
        tokens: Tensor,
        audio_features: Tensor,
    ) -> Tensor:
        """Get logits from the draft model.
        
        Args:
            tokens: Token sequence [batch_size, seq_len]
            audio_features: Encoded audio features [batch_size, n_ctx, dim]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        with torch.no_grad():
            # For getting logits, we don't use KV cache
            logits = self.model.decoder(tokens, audio_features, kv_cache=None)
        return logits
    
    def reset_cache(self):
        """Reset the KV cache."""
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        
        # Clear cache
        self.kv_cache = {}
        self.hooks = []


class LayerDropoutDraft(DraftModel):
    """Draft model using layer dropout on the target model.
    
    This implementation uses the same model but skips some decoder layers
    to create a faster draft model.
    """
    
    def __init__(self, target_model: "Whisper", layers_to_use: Optional[list] = None):
        """Initialize the draft model.
        
        Args:
            target_model: The target Whisper model
            layers_to_use: List of layer indices to use (e.g., [0, 2, 4] for layers 0, 2, 4)
                          If None, uses first half of layers
        """
        self.model = target_model
        self.kv_cache = {}
        self.hooks = []
        
        if layers_to_use is None:
            # Use first half of layers by default
            n_layers = len(self.model.decoder.blocks)
            self.layers_to_use = list(range(0, n_layers, 2))
        else:
            self.layers_to_use = layers_to_use
            
    def generate_draft(
        self,
        tokens: Tensor,
        audio_features: Tensor,
        n_tokens: int,
        temperature: float = 0.0,
    ) -> Tensor:
        """Generate draft tokens using layer dropout.
        
        Args:
            tokens: Current token sequence [batch_size, seq_len]
            audio_features: Encoded audio features [batch_size, n_ctx, dim]
            n_tokens: Number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Draft tokens [batch_size, n_tokens]
        """
        draft_tokens = []
        current_tokens = tokens.clone()
        
        # Install KV cache
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()
        
        with torch.no_grad():
            for _ in range(n_tokens):
                # Get logits with layer dropout
                if len(draft_tokens) == 0:
                    logits = self._forward_with_layer_dropout(
                        current_tokens, audio_features
                    )
                else:
                    logits = self._forward_with_layer_dropout(
                        current_tokens[:, -1:], audio_features
                    )
                
                next_token_logits = logits[:, -1, :]
                
                # Sample next token
                if temperature == 0:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                draft_tokens.append(next_token)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        draft_tokens = torch.cat(draft_tokens, dim=1)
        return draft_tokens
    
    def _forward_with_layer_dropout(
        self,
        tokens: Tensor,
        audio_features: Tensor,
    ) -> Tensor:
        """Forward pass using only selected layers."""
        # Get token embeddings
        x = self.model.decoder.token_embedding(tokens)
        x = x + self.model.decoder.positional_embedding[: x.shape[1]]
        
        # Apply only selected decoder blocks
        for i in self.layers_to_use:
            x = self.model.decoder.blocks[i](x, audio_features, kv_cache=self.kv_cache)
        
        # Apply final layer norm and projection
        x = self.model.decoder.ln(x)
        logits = (
            x @ torch.transpose(self.model.decoder.token_embedding.weight, 0, 1)
        ).float()
        
        return logits
    
    def get_logits(
        self,
        tokens: Tensor,
        audio_features: Tensor,
    ) -> Tensor:
        """Get logits using layer dropout.
        
        Args:
            tokens: Token sequence [batch_size, seq_len]
            audio_features: Encoded audio features [batch_size, n_ctx, dim]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        with torch.no_grad():
            logits = self._forward_with_layer_dropout(tokens, audio_features)
        return logits
    
    def reset_cache(self):
        """Reset the KV cache."""
        for hook in self.hooks:
            hook.remove()
        
        self.kv_cache = {}
        self.hooks = []
