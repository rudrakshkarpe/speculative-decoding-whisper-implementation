"""Metrics for evaluating transcription quality"""

from typing import List, Tuple
import jiwer


def calculate_wer(
    reference: str,
    hypothesis: str,
    case_sensitive: bool = False,
) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis.
    
    Args:
        reference: Ground truth transcription
        hypothesis: Model-generated transcription
        case_sensitive: Whether to consider case in comparison
        
    Returns:
        WER as a float (0.0 = perfect, 1.0 = completely wrong)
    """
    if not case_sensitive:
        reference = reference.lower()
        hypothesis = hypothesis.lower()
    
    # Remove extra whitespace
    reference = " ".join(reference.split())
    hypothesis = " ".join(hypothesis.split())
    
    if not reference:
        return 0.0 if not hypothesis else 1.0
    
    try:
        wer = jiwer.wer(reference, hypothesis)
        return wer
    except Exception as e:
        print(f"Error calculating WER: {e}")
        return 1.0


def calculate_batch_wer(
    references: List[str],
    hypotheses: List[str],
) -> Tuple[float, List[float]]:
    """Calculate WER for a batch of transcriptions.
    
    Args:
        references: List of ground truth transcriptions
        hypotheses: List of model-generated transcriptions
        
    Returns:
        Tuple of (average_wer, list_of_individual_wers)
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Number of references ({len(references)}) must match "
            f"number of hypotheses ({len(hypotheses)})"
        )
    
    individual_wers = []
    for ref, hyp in zip(references, hypotheses):
        wer = calculate_wer(ref, hyp)
        individual_wers.append(wer)
    
    avg_wer = sum(individual_wers) / len(individual_wers) if individual_wers else 0.0
    
    return avg_wer, individual_wers


def calculate_cer(
    reference: str,
    hypothesis: str,
    case_sensitive: bool = False,
) -> float:
    """Calculate Character Error Rate (CER) between reference and hypothesis.
    
    Args:
        reference: Ground truth transcription
        hypothesis: Model-generated transcription
        case_sensitive: Whether to consider case in comparison
        
    Returns:
        CER as a float (0.0 = perfect, 1.0 = completely wrong)
    """
    if not case_sensitive:
        reference = reference.lower()
        hypothesis = hypothesis.lower()
    
    if not reference:
        return 0.0 if not hypothesis else 1.0
    
    try:
        cer = jiwer.cer(reference, hypothesis)
        return cer
    except Exception as e:
        print(f"Error calculating CER: {e}")
        return 1.0


def format_metrics(
    wer: float,
    cer: float = None,
    time_taken: float = None,
    speedup: float = None,
) -> str:
    """Format metrics for display.
    
    Args:
        wer: Word Error Rate
        cer: Character Error Rate (optional)
        time_taken: Time in seconds (optional)
        speedup: Speedup factor (optional)
        
    Returns:
        Formatted string with metrics
    """
    lines = []
    lines.append(f"WER: {wer:.2%}")
    
    if cer is not None:
        lines.append(f"CER: {cer:.2%}")
    
    if time_taken is not None:
        lines.append(f"Time: {time_taken:.2f}s")
    
    if speedup is not None:
        lines.append(f"Speedup: {speedup:.2f}x")
    
    return " | ".join(lines)
