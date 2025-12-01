"""Basic usage example of speculative decoding with Whisper"""

import sys
sys.path.insert(0, "../")

import whisper
import torch

from src.speculative_decoder import SpeculativeWhisperDecoder
from src.draft_model import DistilWhisperDraft
from src.config import SpeculativeConfig


def main():
    """Demonstrate basic usage of speculative decoding."""
    
    # Load models
    print("Loading models...")
    target_model = whisper.load_model("large-v3")
    draft_whisper = whisper.load_model("tiny")
    
    # Create draft model wrapper
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
        temperature=0.0,
        use_adaptive_gamma=True,
    )
    
    # Create speculative decoder
    decoder = SpeculativeWhisperDecoder(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        config=config,
    )
    
    # Load and process audio
    print("Loading audio...")
    audio_path = "../tests/test_audio_samples/jfk.flac"
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    
    # Make mel spectrograms with correct number of mel bins for each model
    # Large V3 uses 128 mel bins, other models use 80
    target_n_mels = target_model.dims.n_mels if hasattr(target_model.dims, 'n_mels') else 80
    draft_n_mels = draft_whisper.dims.n_mels if hasattr(draft_whisper.dims, 'n_mels') else 80
    
    target_mel = whisper.log_mel_spectrogram(audio, n_mels=target_n_mels).to(target_model.device)
    draft_mel = whisper.log_mel_spectrogram(audio, n_mels=draft_n_mels).to(target_model.device)
    
    # Prepare initial tokens
    initial_tokens = torch.tensor(
        [tokenizer.sot_sequence], device=target_model.device
    )
    
    # Decode with speculative decoding
    print("Decoding with speculative decoding...")
    tokens, stats = decoder.decode(
        mel=target_mel.unsqueeze(0),
        initial_tokens=initial_tokens,
        max_length=448,  # Whisper's max length
        eot_token=tokenizer.eot,
        draft_mel=draft_mel.unsqueeze(0),
    )
    
    # Decode tokens to text, skipping special tokens at the start
    # Find where actual transcription starts (after sot_sequence)
    token_list = tokens[0].tolist()
    
    # Remove initial SOT sequence tokens
    sot_len = len(tokenizer.sot_sequence)
    transcription_tokens = token_list[sot_len:]
    
    # Remove EOT if present
    if tokenizer.eot in transcription_tokens:
        eot_index = transcription_tokens.index(tokenizer.eot)
        transcription_tokens = transcription_tokens[:eot_index]
    
    # Decode to text
    text = tokenizer.decode(transcription_tokens)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Transcription: {text}")
    print("\nStatistics:")
    print(f"  Total iterations: {stats['total_iterations']}")
    print(f"  Total draft tokens: {stats['total_draft_tokens']}")
    print(f"  Total accepted tokens: {stats['total_accepted_tokens']}")
    print(f"  Overall acceptance rate: {stats['overall_acceptance_rate']:.2%}")
    print(f"  Avg tokens per iteration: {stats['avg_tokens_per_iteration']:.2f}")
    print(f"  Final gamma: {stats['final_gamma']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
