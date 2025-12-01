"""Debug script to test speculative decoding step by step"""
import sys
sys.path.insert(0, "whisper")
sys.path.insert(0, ".")

import whisper
import torch
from src.draft_model import DistilWhisperDraft

# Load models
print("Loading models...")
target_model = whisper.load_model("large-v3")
draft_whisper = whisper.load_model("tiny")
draft_model = DistilWhisperDraft(draft_whisper)

# Get tokenizer
tokenizer = whisper.tokenizer.get_tokenizer(
    target_model.is_multilingual,
    num_languages=target_model.num_languages,
    task="transcribe"
)

# Load audio
print("\nLoading audio...")
audio = whisper.load_audio("tests/test_audio_samples/jfk.flac")
audio = whisper.pad_or_trim(audio)

# Create mel spectrograms
target_mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(target_model.device)
draft_mel = whisper.log_mel_spectrogram(audio, n_mels=80).to(target_model.device)

# Encode audio
print("Encoding audio...")
with torch.no_grad():
    target_features = target_model.encoder(target_mel.unsqueeze(0))
    draft_features = draft_whisper.encoder(draft_mel.unsqueeze(0))

# Initial tokens
initial_tokens = torch.tensor([tokenizer.sot_sequence], device=target_model.device)
print(f"\nInitial tokens: {initial_tokens}")
print(f"SOT sequence: {tokenizer.sot_sequence}")
print(f"EOT token: {tokenizer.eot}")

# Test draft model generation
print("\n" + "="*80)
print("TEST 1: Draft model generation")
print("="*80)
draft_tokens = draft_model.generate_draft(
    initial_tokens, draft_features, n_tokens=4, temperature=0.0
)
print(f"Draft tokens shape: {draft_tokens.shape}")
print(f"Draft tokens: {draft_tokens[0].tolist()}")
print(f"Draft tokens decoded: {[tokenizer.decode([t]) for t in draft_tokens[0].tolist()]}")

# Test target model prediction
print("\n" + "="*80)
print("TEST 2: Target model prediction")
print("="*80)
candidate_tokens = torch.cat([initial_tokens, draft_tokens], dim=1)
with torch.no_grad():
    target_logits = target_model.decoder(candidate_tokens, target_features, kv_cache=None)
    
# Get next token from target
next_token_logits = target_logits[:, -1, :]
next_token = next_token_logits.argmax(dim=-1)
print(f"Target next token: {next_token.item()}")
print(f"Target next token decoded: {tokenizer.decode([next_token.item()])}")

# Test standard Whisper for comparison
print("\n" + "="*80)
print("TEST 3: Standard Whisper tiny (for reference)")
print("="*80)
with torch.no_grad():
    tiny_logits = draft_whisper.decoder(initial_tokens, draft_features, kv_cache=None)
    tiny_next = tiny_logits[:, -1, :].argmax(dim=-1)
    print(f"Tiny next token: {tiny_next.item()}")
    print(f"Tiny next token decoded: {tokenizer.decode([tiny_next.item()])}")

print("\n" + "="*80)
print("TEST 4: Generate full sequence with tiny model")
print("="*80)
current = initial_tokens.clone()
for i in range(20):
    with torch.no_grad():
        logits = draft_whisper.decoder(current, draft_features, kv_cache=None)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current = torch.cat([current, next_tok], dim=1)
        if next_tok.item() == tokenizer.eot:
            break
            
transcription_tokens = current[0, len(tokenizer.sot_sequence):].tolist()
if tokenizer.eot in transcription_tokens:
    transcription_tokens = transcription_tokens[:transcription_tokens.index(tokenizer.eot)]
    
print(f"Generated {len(transcription_tokens)} tokens")
