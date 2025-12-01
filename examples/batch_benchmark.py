"""Batch Benchmark Script for Speculative Decoding

This script processes multiple audio files and compares:
- Standard Whisper Large V3 decoding
- Speculative decoding (Tiny → Large V3)

Outputs per-file and aggregate statistics with CSV export.
"""

import sys
sys.path.insert(0, "../")

import time
import csv
from pathlib import Path
from typing import List, Dict
import whisper
import torch

from src.speculative_decoder import SpeculativeWhisperDecoder
from src.draft_model import DistilWhisperDraft
from src.config import SpeculativeConfig


def discover_audio_files(directory: str) -> List[Path]:
    """Discover all audio files in directory.
    
    Args:
        directory: Path to directory containing audio files
        
    Returns:
        List of Path objects for audio files
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Support common audio formats
    audio_extensions = ['.m4a', '.mp3', '.wav', '.flac', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(dir_path.glob(f'*{ext}'))
    
    return sorted(audio_files)


def benchmark_single_file(
    audio_path: Path,
    target_model,
    draft_model,
    decoder,
    tokenizer,
) -> Dict:
    """Benchmark a single audio file.
    
    Args:
        audio_path: Path to audio file
        target_model: Whisper Large V3 model
        draft_model: Draft model wrapper
        decoder: Speculative decoder
        tokenizer: Whisper tokenizer
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\nProcessing: {audio_path.name}")
    
    # Standard Whisper decoding
    print("  Running standard decoding...", end=" ", flush=True)
    try:
        start_time = time.time()
        std_result = target_model.transcribe(str(audio_path))
        std_time = time.time() - start_time
        std_text = std_result['text'].strip()
        print(f"✓ ({std_time:.2f}s)")
    except Exception as e:
        print(f"✗ Error: {e}")
        return None
    
    # Speculative decoding
    print("  Running speculative decoding...", end=" ", flush=True)
    try:
        # Load and process audio
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)
        
        # Create mel spectrograms
        target_n_mels = target_model.dims.n_mels if hasattr(target_model.dims, 'n_mels') else 80
        draft_n_mels = draft_model.model.dims.n_mels if hasattr(draft_model.model.dims, 'n_mels') else 80
        
        target_mel = whisper.log_mel_spectrogram(audio, n_mels=target_n_mels).to(target_model.device)
        draft_mel = whisper.log_mel_spectrogram(audio, n_mels=draft_n_mels).to(target_model.device)
        
        # Prepare initial tokens
        initial_tokens = torch.tensor([tokenizer.sot_sequence], device=target_model.device)
        
        # Decode
        start_time = time.time()
        tokens, stats = decoder.decode(
            mel=target_mel.unsqueeze(0),
            initial_tokens=initial_tokens,
            max_length=448,
            eot_token=tokenizer.eot,
            draft_mel=draft_mel.unsqueeze(0),
        )
        spec_time = time.time() - start_time
        
        # Decode tokens to text
        token_list = tokens[0].tolist()
        sot_len = len(tokenizer.sot_sequence)
        transcription_tokens = token_list[sot_len:]
        
        if tokenizer.eot in transcription_tokens:
            eot_index = transcription_tokens.index(tokenizer.eot)
            transcription_tokens = transcription_tokens[:eot_index]
        
        spec_text = tokenizer.decode(transcription_tokens).strip()
        
        # Clean special tokens
        for token in ['<|notimestamps|>', '<|timestamps|>', '<|0.00|>', '<|nospeech|>']:
            spec_text = spec_text.replace(token, '')
        spec_text = ' '.join(spec_text.split()).strip()
        
        print(f"✓ ({spec_time:.2f}s)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None
    
    # Calculate metrics
    speedup = std_time / spec_time if spec_time > 0 else 0
    time_saved = std_time - spec_time
    
    return {
        'filename': audio_path.name,
        'std_time': std_time,
        'spec_time': spec_time,
        'speedup': speedup,
        'time_saved': time_saved,
        'std_text': std_text,
        'spec_text': spec_text,
        'acceptance_rate': stats['overall_acceptance_rate'],
        'avg_tokens_per_iter': stats['avg_tokens_per_iteration'],
        'total_iterations': stats['total_iterations'],
    }


def print_results_table(results: List[Dict]):
    """Print formatted results table.
    
    Args:
        results: List of benchmark results
    """
    print("\n" + "=" * 100)
    print("BATCH BENCHMARK RESULTS")
    print("=" * 100)
    
    # Per-file results
    print(f"\n{'File':<25} {'Standard':<12} {'Speculative':<12} {'Speedup':<10} {'Acceptance':<12}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['filename']:<25} "
              f"{result['std_time']:>10.2f}s  "
              f"{result['spec_time']:>10.2f}s  "
              f"{result['speedup']:>8.2f}x  "
              f"{result['acceptance_rate']:>10.1%}")
    
    # Aggregate statistics
    print("-" * 100)
    total_std_time = sum(r['std_time'] for r in results)
    total_spec_time = sum(r['spec_time'] for r in results)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    avg_acceptance = sum(r['acceptance_rate'] for r in results) / len(results)
    total_time_saved = sum(r['time_saved'] for r in results)
    
    print(f"{'TOTALS/AVERAGES':<25} "
          f"{total_std_time:>10.2f}s  "
          f"{total_spec_time:>10.2f}s  "
          f"{avg_speedup:>8.2f}x  "
          f"{avg_acceptance:>10.1%}")
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total files processed:     {len(results)}")
    print(f"Average speedup:           {avg_speedup:.2f}x")
    print(f"Average acceptance rate:   {avg_acceptance:.1%}")
    print(f"Total time saved:          {total_time_saved:.2f}s ({total_time_saved/total_std_time*100:.1f}%)")
    print(f"Standard total time:       {total_std_time:.2f}s")
    print(f"Speculative total time:    {total_spec_time:.2f}s")
    
    # Performance verdict
    print(f"\nPerformance: ", end="")
    if avg_speedup > 1.5:
        print(f"✓ EXCELLENT ({avg_speedup:.2f}x faster)")
    elif avg_speedup > 1.2:
        print(f"✓ GOOD ({avg_speedup:.2f}x faster)")
    elif avg_speedup > 1.0:
        print(f"✓ MODEST ({avg_speedup:.2f}x faster)")
    else:
        print(f"✗ SLOWER ({avg_speedup:.2f}x)")
    
    print("=" * 100)


def export_to_csv(results: List[Dict], output_path: str):
    """Export results to CSV file.
    
    Args:
        results: List of benchmark results
        output_path: Path to output CSV file
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename', 'std_time', 'spec_time', 'speedup', 'time_saved',
            'acceptance_rate', 'avg_tokens_per_iter', 'total_iterations',
            'std_text', 'spec_text'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Results exported to: {output_path}")


def main():
    """Run batch benchmark."""
    
    # Configuration
    batch_dir = "../tests/batch_tests"
    output_csv = "batch_benchmark_results.csv"
    
    print("=" * 100)
    print("BATCH BENCHMARK: Standard vs Speculative Decoding")
    print("=" * 100)
    
    # Discover audio files
    print(f"\nDiscovering audio files in: {batch_dir}")
    try:
        audio_files = discover_audio_files(batch_dir)
        print(f"Found {len(audio_files)} audio file(s):")
        for f in audio_files:
            print(f"  - {f.name}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    if not audio_files:
        print("\nNo audio files found!")
        return
    
    # Load models
    print("\nLoading models...")
    print("  - Target model: Whisper Large V3")
    target_model = whisper.load_model("large-v3")
    
    print("  - Draft model: Whisper Tiny")
    draft_whisper = whisper.load_model("tiny")
    draft_model = DistilWhisperDraft(draft_whisper)
    
    # Get tokenizer
    tokenizer = whisper.tokenizer.get_tokenizer(
        target_model.is_multilingual,
        num_languages=target_model.num_languages,
        task="transcribe"
    )
    
    # Configure speculative decoding
    config = SpeculativeConfig(
        gamma=4,
        acceptance_threshold=0.8,
        use_adaptive_gamma=True,
    )
    
    # Create decoder (reuse for all files)
    decoder = SpeculativeWhisperDecoder(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        config=config,
    )
    
    # Benchmark each file
    print("\n" + "=" * 100)
    print("BENCHMARKING FILES")
    print("=" * 100)
    
    results = []
    for audio_file in audio_files:
        result = benchmark_single_file(
            audio_file, target_model, draft_model, decoder, tokenizer
        )
        if result:
            results.append(result)
            # Reset decoder stats for next file
            decoder.reset_stats()
    
    if not results:
        print("\nNo successful results to display!")
        return
    
    # Display results
    print_results_table(results)
    
    # Export to CSV
    export_to_csv(results, output_csv)
    
    print()


if __name__ == "__main__":
    main()
