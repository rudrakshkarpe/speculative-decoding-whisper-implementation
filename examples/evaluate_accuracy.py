"""Evaluation script comparing accuracy and speed of standard vs speculative decoding"""

import sys
sys.path.insert(0, "../")

import time
from pathlib import Path

from src.api import SpeculativeWhisper
from src.metrics import calculate_wer, format_metrics


# Ground truth transcription for JFK audio
GROUND_TRUTH = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"


def evaluate_model(
    draft_model: str,
    final_model: str,
    audio_path: str,
    ground_truth: str,
):
    """Evaluate a model configuration.
    
    Args:
        draft_model: Draft model size
        final_model: Final model size
        audio_path: Path to audio file
        ground_truth: Ground truth transcription
    """
    print(f"\n{'=' * 80}")
    print(f"Evaluating: Draft={draft_model}, Final={final_model}")
    print('=' * 80)
    
    # Initialize model
    sw = SpeculativeWhisper(
        draft_model=draft_model,
        final_model=final_model,
        device=None,  # Auto-detect
    )
    
    # Transcribe with speculative decoding
    print("\nSpeculative Decoding:")
    start_time = time.time()
    spec_results = sw.transcribe([audio_path], return_stats=True)
    spec_time = time.time() - start_time
    
    spec_text = spec_results[0]["text"]
    spec_stats = spec_results[0]["stats"]
    
    print(f"  Transcription: {spec_text}")
    print(f"  Time: {spec_time:.2f}s")
    print(f"  Acceptance rate: {spec_stats['overall_acceptance_rate']:.2%}")
    
    # Transcribe with standard Whisper
    print("\nStandard Whisper:")
    start_time = time.time()
    std_results = sw.transcribe_standard([audio_path])
    std_time = time.time() - start_time
    
    std_text = std_results[0]
    
    print(f"  Transcription: {std_text}")
    print(f"  Time: {std_time:.2f}s")
    
    # Calculate metrics
    print("\nAccuracy Metrics:")
    
    spec_wer = calculate_wer(ground_truth, spec_text)
    std_wer = calculate_wer(ground_truth, std_text)
    speedup = std_time / spec_time if spec_time > 0 else 0
    
    print(f"  Standard WER: {std_wer:.2%}")
    print(f"  Speculative WER: {spec_wer:.2%}")
    print(f"  WER Difference: {(spec_wer - std_wer):.2%}")
    print(f"  Speedup: {speedup:.2f}x")
    
    return {
        "draft_model": draft_model,
        "final_model": final_model,
        "spec_wer": spec_wer,
        "std_wer": std_wer,
        "spec_time": spec_time,
        "std_time": std_time,
        "speedup": speedup,
        "spec_text": spec_text,
        "std_text": std_text,
        "stats": spec_stats,
    }


def main():
    """Run comprehensive evaluation."""
    
    audio_path = "../tests/test_audio_samples/jfk.flac"
    
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    print("=" * 80)
    print("SPECULATIVE DECODING ACCURACY EVALUATION")
    print("=" * 80)
    print(f"\nGround Truth: {GROUND_TRUTH}")
    print(f"Audio: {audio_path}")
    
    # Configuration: Tiny → Large V3 (as per assignment)
    results_1 = evaluate_model(
        draft_model="tiny",
        final_model="large-v3",
        audio_path=audio_path,
        ground_truth=GROUND_TRUTH,
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"\nConfiguration: Tiny → Large V3")
    print(f"  Standard Whisper:")
    print(f"    - WER: {results_1['std_wer']:.2%}")
    print(f"    - Time: {results_1['std_time']:.2f}s")
    
    print(f"\n  Speculative Decoding:")
    print(f"    - WER: {results_1['spec_wer']:.2%}")
    print(f"    - Time: {results_1['spec_time']:.2f}s")
    print(f"    - Speedup: {results_1['speedup']:.2f}x")
    print(f"    - Acceptance Rate: {results_1['stats']['overall_acceptance_rate']:.2%}")
    print(f"    - Avg Tokens/Iteration: {results_1['stats']['avg_tokens_per_iteration']:.2f}")
    
    print(f"\n  Analysis:")
    wer_diff = results_1['spec_wer'] - results_1['std_wer']
    if abs(wer_diff) < 0.05:  # Less than 5% difference
        quality_verdict = "✓ Quality maintained"
    elif wer_diff > 0:
        quality_verdict = "⚠ Slightly lower quality"
    else:
        quality_verdict = "✓ Better quality"
    
    print(f"    - Quality: {quality_verdict} (WER diff: {wer_diff:+.2%})")
    
    if results_1['speedup'] > 1.0:
        speed_verdict = f"✓ {results_1['speedup']:.2f}x faster"
    else:
        speed_verdict = "✗ Slower than standard"
    print(f"    - Speed: {speed_verdict}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
