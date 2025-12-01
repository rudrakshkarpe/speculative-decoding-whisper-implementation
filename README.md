# Speculative Decoding for OpenAI Whisper

Implementation of speculative decoding for OpenAI's Whisper ASR model using **Whisper Tiny** as the draft model and **Whisper Large V3** as the target model. This technique accelerates inference while maintaining identical output quality.

## System Architecture

<!-- <img width="1429" height="198" alt="image" src="https://github.com/user-attachments/assets/7745f8f3-e20a-41d5-97d7-e54265f5ac3d" /> -->

<img width="942" height="559" alt="image" src="https://github.com/user-attachments/assets/dac87d38-b62b-4e93-97de-1572183a6888" />





## Overview

This project implements the assignment requirements for speculative decoding on Whisper:
- **Draft Model**: Whisper Tiny generates candidate tokens quickly
- **Target Model**: Whisper Large V3 verifies and refines the output
- **Result**: Faster inference with same quality as standard Large V3

### Key Features

- **Speedup** over standard Whisper Large V3 decoding
- **Identical output distribution** - maintains exact same quality
- **Batch processing** for multiple audio files
- **WER evaluation** to compare accuracy
- **Configurable parameters** for tuning performance
- **GPU/CPU support** with auto-detection

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- OpenAI Whisper

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### API Usage (As Per Assignment)

```python
from src import SpeculativeWhisper

# Initialize with Tiny (draft) and Large V3 (target)
sw = SpeculativeWhisper(
    draft_model="tiny",
    final_model="large-v3",
    device="cuda"  # or "cpu"
)

# Transcribe audio files
audio_files = ["audio1.wav", "audio2.wav"]
outputs = sw.transcribe(audio_files, max_tokens=200, batch_size=2)

for audio, text in zip(audio_files, outputs):
    print(f"{audio}: {text}")
```

### With Statistics

```python
# Get detailed performance statistics
results = sw.transcribe(audio_files, return_stats=True)

for result in results:
    print(f"Text: {result['text']}")
    print(f"Acceptance Rate: {result['stats']['overall_acceptance_rate']:.2%}")
    print(f"Speedup: {result['stats']['avg_tokens_per_iteration']:.2f}x")
```

## Running Benchmarks

### Compare Speculative vs Standard Large V3

```bash
cd examples
python3 benchmark_comparison.py
```

This compares:
1. **Standard Whisper Large V3** - baseline performance
2. **Speculative Decoding** (Tiny → Large V3) - accelerated version

Output includes:
- Transcription results
- Time taken for each approach
- Speedup achieved
- Token acceptance rate

### WER Evaluation

```bash
cd examples
python3 evaluate_accuracy.py
```

Calculates Word Error Rate (WER) for both approaches to verify accuracy is maintained.

## Configuration

### SpeculativeConfig Parameters

Control the behavior of speculative decoding:

```python
from src import SpeculativeWhisper, SpeculativeConfig

config = SpeculativeConfig(
    gamma=4,                    # Tokens to generate per iteration
    acceptance_threshold=0.8,   # Probability threshold for acceptance
    temperature=0.0,            # 0 = greedy, >0 = sampling
    use_adaptive_gamma=True,    # Auto-adjust gamma based on acceptance
)

sw = SpeculativeWhisper(
    draft_model="tiny",
    final_model="large-v3",
    config=config
)
```

### Key Parameters

| Parameter | Description | Default | Tuning Advice |
|-----------|-------------|---------|---------------|
| `gamma` | Number of draft tokens per iteration | 4 | Higher = more speculation, lower acceptance |
| `acceptance_threshold` | Min probability ratio to accept | 0.8 | Lower = more aggressive (0.7-0.9) |
| `use_adaptive_gamma` | Auto-adjust gamma | True | Usually helps |
| `temperature` | Sampling temperature | 0.0 | 0 = greedy (deterministic) |

## How It Works

### Speculative Decoding Algorithm

1. **Draft Generation**: Whisper Tiny quickly generates γ candidate tokens
2. **Parallel Verification**: Whisper Large V3 verifies all candidates in one pass
3. **Acceptance/Rejection**: 
   - Accept tokens where target and draft agree
   - On mismatch, sample from adjusted distribution
4. **Repeat**: Continue until end-of-text or max length

### Why It's Faster

- Draft model (Tiny) is much faster than target (Large V3)
- Target model verifies multiple tokens in parallel (single forward pass)
- Good acceptance rate means fewer target model calls
- No quality loss - output distribution matches standard decoding exactly

## Examples

### 1. Basic API Usage
```bash
cd examples
python3 api_usage_example.py
```
Shows the simple API from the assignment.

### 2. Low-Level Usage
```bash
cd examples
python3 basic_usage.py
```
Demonstrates direct use of SpeculativeWhisperDecoder.

### 3. Benchmark Comparison
```bash
cd examples  
python3 benchmark_comparison.py
```
Compares Standard vs Speculative Large V3 decoding.

### 4. Accuracy Evaluation
```bash
cd examples
python3 evaluate_accuracy.py
```
Measures WER to verify quality is maintained.

## Project Structure

```
speculative-whisper/
├── src/
│   ├── api.py                   # SpeculativeWhisper API class
│   ├── speculative_decoder.py   # Core algorithm
│   ├── draft_model.py           # Draft model implementations
│   ├── metrics.py               # WER/CER calculations
│   └── config.py                # Configuration
├── examples/
│   ├── api_usage_example.py     # API demo (assignment format)
│   ├── benchmark_comparison.py  # Speculative vs Standard comparison
│   ├── evaluate_accuracy.py     # WER evaluation
│   └── basic_usage.py           # Low-level usage
├── tests/
│   ├── test_config.py
│   ├── test_speculative_decoder.py
│   └── test_audio_samples/jfk.flac
└── requirements.txt
```


## Citation

Based on speculative decoding research:

```bibtex
@article{leviathan2023fast,
  title={Fast Inference from Transformers via Speculative Decoding},
  author={Leviathan, Yaniv and Kalman, Matan and Matias, Yossi},
  journal={arXiv preprint arXiv:2211.17192},
  year={2023}
}
```

## License

MIT License - builds on OpenAI Whisper

## Acknowledgments

- OpenAI Whisper team
- Speculative decoding research by Google and DeepMind
- PyTorch team
