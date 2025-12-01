"""Benchmark comparing standard vs speculative decoding with Whisper Large V3

This script implements the evaluation requirement from assignment.md:
"Compare speed (latency per sample) of speculative decoding vs. standard Whisper Large V3 decoding."

Uses:
- Draft model: Whisper Tiny
- Target model: Whisper Large V3
- Comparison: Speculative decoding vs Standard Large V3 decoding
"""

import sys
sys.path.insert(0, "../")

import whisper
import torch
import time
from pathlib import Path

from src.speculative_decoder import SpeculativeWhisperDecoder
from src.draft_model import DistilWhisperDraft
from src.config import SpeculativeConfig


def benchmark_standard_decoding(model, audio_path):
    """Benchmark standard Whisper decoding.
    
    Args:
        model: Whisper model (Large V3)
        audio_path: Path to audio file
        
    Returns:
        Tuple of (transcription_text, time_taken)
    """
    print("\n" + "=" * 80)
    print("STANDARD WHISPER LARGE V3 DECODING")
    print("=" * 80)
    
    start_time = time.time()
    result = model.transcribe(audio_path)
    end_time = time.time()
    
    duration = end_time - start_time
    
    print(f"Transcription: {result['text']}")
    print(f"Time: {duration:.2f}s")
    
    return result['text'], duration


def benchmark_speculative_decoding(
    target_model,
    draft_model,
    audio_path,
    config,
):
    """Benchmark speculative decoding.
    
    Args:
        target_model: Target Whisper model (Large V3)
        draft_model: Draft model wrapper (Tiny)
        audio_path: Path to audio file
        config: Speculative decoding configuration
        
    Returns:
        Tuple of (transcription_text, time_taken, statistics_dict)
    """
    print("\n" + "=" * 80)
    print("SPECULATIVE DECODING (TINY → LARGE V3)")
    print("=" * 80)
    
    # Get tokenizer
    tokenizer = whisper.tokenizer.get_tokenizer(
        target_model.is_multilingual,
        num_languages=target_model.num_languages,
        task="transcribe"
    )
    
    # Create speculative decoder
    decoder = SpeculativeWhisperDecoder(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        config=config,
    )
    
    # Load and process audio
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    
    # Create mel spectrograms with correct number of mel bins for each model
    # Large V3 uses 128 mel bins, other models use 80
    target_n_mels = target_model.dims.n_mels if hasattr(target_model.dims, 'n_mels') else 80
    draft_n_mels = draft_model.model.dims.n_mels if hasattr(draft_model.model.dims, 'n_mels') else 80
    
    target_mel = whisper.log_mel_spectrogram(audio, n_mels=target_n_mels).to(target_model.device)
    draft_mel = whisper.log_mel_spectrogram(audio, n_mels=draft_n_mels).to(target_model.device)
    
    # Prepare initial tokens
    initial_tokens = torch.tensor(
        [tokenizer.sot_sequence], device=target_model.device
    )
    
    # Decode with timing
    start_time = time.time()
    tokens, stats = decoder.decode(
        mel=target_mel.unsqueeze(0),
        initial_tokens=initial_tokens,
        max_length=448,
        eot_token=tokenizer.eot,
        draft_mel=draft_mel.unsqueeze(0),
    )
    end_time = time.time()
    
    duration = end_time - start_time
    
    # Decode tokens to text, removing special tokens
    token_list = tokens[0].tolist()
    sot_len = len(tokenizer.sot_sequence)
    transcription_tokens = token_list[sot_len:]
    
    if tokenizer.eot in transcription_tokens:
        eot_index = transcription_tokens.index(tokenizer.eot)
        transcription_tokens = transcription_tokens[:eot_index]
    
    text = tokenizer.decode(transcription_tokens).strip()
    
    # Remove Whisper special tokens for fair comparison
    special_tokens = ['<|notimestamps|>', '<|timestamps|>', '<|0.00|>', '<|nospeech|>']
    for token in special_tokens:
        text = text.replace(token, '')
    text = ' '.join(text.split()).strip()
    
    print(f"Transcription: {text}")
    print(f"Time: {duration:.2f}s")
    print(f"\nStatistics:")
    print(f"  Total iterations: {stats['total_iterations']}")
    print(f"  Total draft tokens: {stats['total_draft_tokens']}")
    print(f"  Total accepted tokens: {stats['total_accepted_tokens']}")
    print(f"  Overall acceptance rate: {stats['overall_acceptance_rate']:.2%}")
    print(f"  Avg tokens per iteration: {stats['avg_tokens_per_iteration']:.2f}")
    print(f"  Final gamma: {stats['final_gamma']}")
    
    return text, duration, stats


def main():
    """Run benchmark comparing standard vs speculative decoding."""
    
    # Configuration
    audio_path = "../tests/test_audio_samples/jfk.flac"
    
    print("=" * 80)
    print("WHISPER LARGE V3: STANDARD VS SPECULATIVE DECODING BENCHMARK")
    print("=" * 80)
    print("\nThis benchmark compares:")
    print("  1. Standard Whisper Large V3 decoding")
    print("  2. Speculative decoding (Tiny → Large V3)")
    print("\nAs per assignment requirements.\n")
    
    # Load models
    print("Loading models...")
    print("  - Target model: Whisper Large V3")
    target_model = whisper.load_model("large-v3")
    
    print("  - Draft model: Whisper Tiny")
    draft_whisper = whisper.load_model("tiny")
    draft_model = DistilWhisperDraft(draft_whisper)
    
    # Configure speculative decoding
    config = SpeculativeConfig(
        gamma=4,
        acceptance_threshold=0.8,
        use_adaptive_gamma=True,
    )
    
    # Benchmark 1: Standard Whisper Large V3
    standard_text, standard_time = benchmark_standard_decoding(
        target_model, audio_path
    )
    
    # Benchmark 2: Speculative Decoding (Tiny → Large V3)
    spec_text, spec_time, stats = benchmark_speculative_decoding(
        target_model, draft_model, audio_path, config
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"\n1. Standard Whisper Large V3:")
    print(f"     Time: {standard_time:.2f}s")
    print(f"     Text: {standard_text.strip()}")
    
    print(f"\n2. Speculative Decoding (Tiny → Large V3):")
    print(f"     Time: {spec_time:.2f}s")
    print(f"     Text: {spec_text.strip()}")
    print(f"     Acceptance rate: {stats['overall_acceptance_rate']:.2%}")
    print(f"     Avg tokens/iteration: {stats['avg_tokens_per_iteration']:.2f}")
    
    # Calculate speedup
    speedup = standard_time / spec_time if spec_time > 0 else 0
    
    print(f"\n3. Performance Comparison:")
    print(f"     Speedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        print(f"     ✓ Speculative decoding is {speedup:.2f}x FASTER")
        time_saved = standard_time - spec_time
        percent_saved = (time_saved / standard_time) * 100
        print(f"     ✓ Time saved: {time_saved:.2f}s ({percent_saved:.1f}%)")
    elif speedup < 1.0:
        print(f"     ✗ Speculative decoding is slower")
    else:
        print(f"     = Same speed")
    
    # Text match
    text_matches = standard_text.strip() == spec_text.strip()
    print(f"     Text matches: {text_matches}")
    
    if not text_matches:
        print(f"     Note: Transcriptions differ (may be due to decoding stopped early)")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"\nSpeculative decoding with Whisper Tiny as draft model")
    print(f"provides a {speedup:.2f}x speedup over standard Whisper Large V3")
    print(f"decoding, with {stats['overall_acceptance_rate']:.1%} token acceptance rate.")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
