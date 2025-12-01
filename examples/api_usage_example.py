"""Example matching the assignment's API usage"""

import sys
sys.path.insert(0, "../")

from src import SpeculativeWhisper

def main():
    """Demonstrate the API usage from assignment.md"""
    
    # Example from assignment
    sw = SpeculativeWhisper(
        draft_model="tiny",
        final_model="large-v3",
        device="cpu"  # or "cuda" if GPU available
    )
    
    audio_files = ["../tests/test_audio_samples/jfk.flac"]
    
    print("Transcribing audio files...")
    outputs = sw.transcribe(audio_files, max_tokens=200, batch_size=1)
    
    print("\nResults:")
    print("=" * 80)
    for audio, text in zip(audio_files, outputs):
        print(f"{audio}:")
        print(f"  {text}")
    print("=" * 80)
    
    # With statistics
    print("\nWith detailed statistics:")
    outputs_with_stats = sw.transcribe(audio_files, max_tokens=200, return_stats=True)
    
    for audio, result in zip(audio_files, outputs_with_stats):
        print(f"\n{audio}:")
        print(f"  Text: {result['text']}")
        print(f"  Stats:")
        for key, value in result['stats'].items():
            print(f"    - {key}: {value}")


if __name__ == "__main__":
    main()
