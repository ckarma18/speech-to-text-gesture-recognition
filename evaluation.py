"""
Evaluation Metrics Module
Calculates WER, CER, and other metrics for ASR system evaluation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import difflib
from pathlib import Path
import json


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    wer: float  # Word Error Rate
    cer: float  # Character Error Rate
    num_words: int
    num_characters: int
    num_substitutions: int
    num_deletions: int
    num_insertions: int
    
    def __str__(self) -> str:
        return (
            f"WER: {self.wer:.2f}% | CER: {self.cer:.2f}% | "
            f"Words: {self.num_words} | Chars: {self.num_characters}"
        )


class Evaluator:
    """Evaluation metrics for ASR systems"""
    
    @staticmethod
    def calculate_wer(hypothesis: str, reference: str) -> Tuple[float, Dict]:
        """
        Calculate Word Error Rate (WER)
        
        WER = (S + D + I) / N * 100%
        where:
        S = number of substitutions
        D = number of deletions
        I = number of insertions
        N = total number of words in reference
        
        Args:
            hypothesis: Predicted transcription
            reference: Ground truth transcription
            
        Returns:
            Tuple of (WER percentage, detailed metrics)
        """
        # Split into words
        hypothesis_words = hypothesis.lower().split()
        reference_words = reference.lower().split()
        
        # Calculate edit distance
        matcher = difflib.SequenceMatcher(None, reference_words, hypothesis_words)
        matches = 0
        
        # Count operations using dynamic programming
        operations = Evaluator._levenshtein_distance_ops(
            reference_words,
            hypothesis_words
        )
        
        substitutions = operations['substitutions']
        deletions = operations['deletions']
        insertions = operations['insertions']
        matches = operations['matches']
        
        num_words = len(reference_words)
        
        if num_words == 0:
            return 0.0, {
                'substitutions': 0,
                'deletions': 0,
                'insertions': 0,
                'matches': 0,
                'total_words': 0
            }
        
        wer = ((substitutions + deletions + insertions) / num_words) * 100.0
        
        return wer, {
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'matches': matches,
            'total_words': num_words
        }
    
    @staticmethod
    def calculate_cer(hypothesis: str, reference: str) -> Tuple[float, Dict]:
        """
        Calculate Character Error Rate (CER)
        
        CER = (S + D + I) / N * 100%
        where operations are at character level
        
        Args:
            hypothesis: Predicted transcription
            reference: Ground truth transcription
            
        Returns:
            Tuple of (CER percentage, detailed metrics)
        """
        # Use characters instead of words
        hypothesis_chars = list(hypothesis.lower().replace(" ", ""))
        reference_chars = list(reference.lower().replace(" ", ""))
        
        operations = Evaluator._levenshtein_distance_ops(
            reference_chars,
            hypothesis_chars
        )
        
        substitutions = operations['substitutions']
        deletions = operations['deletions']
        insertions = operations['insertions']
        matches = operations['matches']
        
        num_chars = len(reference_chars)
        
        if num_chars == 0:
            return 0.0, {
                'substitutions': 0,
                'deletions': 0,
                'insertions': 0,
                'matches': 0,
                'total_characters': 0
            }
        
        cer = ((substitutions + deletions + insertions) / num_chars) * 100.0
        
        return cer, {
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'matches': matches,
            'total_characters': num_chars
        }
    
    @staticmethod
    def _levenshtein_distance_ops(reference: List[str], hypothesis: List[str]) -> Dict:
        """
        Calculate Levenshtein distance and count operations
        
        Args:
            reference: Reference sequence
            hypothesis: Hypothesis sequence
            
        Returns:
            Dictionary with operation counts
        """
        len_ref = len(reference)
        len_hyp = len(hypothesis)
        
        # Dynamic programming matrix
        dp = np.zeros((len_ref + 1, len_hyp + 1), dtype=int)
        
        # Initialize first row and column
        for i in range(len_ref + 1):
            dp[i][0] = i
        for j in range(len_hyp + 1):
            dp[0][j] = j
        
        # Fill matrix
        for i in range(1, len_ref + 1):
            for j in range(1, len_hyp + 1):
                if reference[i-1] == hypothesis[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # Deletion
                        dp[i][j-1],      # Insertion
                        dp[i-1][j-1]     # Substitution
                    )
        
        # Backtrack to count operations
        substitutions = 0
        deletions = 0
        insertions = 0
        matches = 0
        
        i, j = len_ref, len_hyp
        while i > 0 or j > 0:
            if i > 0 and j > 0 and reference[i-1] == hypothesis[j-1]:
                matches += 1
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i-1][j-1] + 1 == dp[i][j]:
                substitutions += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i-1][j] + 1 == dp[i][j]:
                deletions += 1
                i -= 1
            else:
                insertions += 1
                j -= 1
        
        return {
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'matches': matches
        }
    
    @staticmethod
    def evaluate_batch(
        hypotheses: List[str],
        references: List[str],
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate batch of predictions
        
        Args:
            hypotheses: List of predicted transcriptions
            references: List of reference transcriptions
            verbose: Print results
            
        Returns:
            Evaluation results including WER, CER, and statistics
        """
        if len(hypotheses) != len(references):
            raise ValueError("Number of hypotheses and references must match")
        
        wers = []
        cers = []
        
        for hyp, ref in zip(hypotheses, references):
            wer, _ = Evaluator.calculate_wer(hyp, ref)
            cer, _ = Evaluator.calculate_cer(hyp, ref)
            
            wers.append(wer)
            cers.append(cer)
        
        results = {
            'wer': {
                'mean': np.mean(wers),
                'std': np.std(wers),
                'min': np.min(wers),
                'max': np.max(wers),
                'values': wers
            },
            'cer': {
                'mean': np.mean(cers),
                'std': np.std(cers),
                'min': np.min(cers),
                'max': np.max(cers),
                'values': cers
            },
            'num_samples': len(hypotheses)
        }
        
        if verbose:
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            print(f"\n📊 Word Error Rate (WER):")
            print(f"  • Mean: {results['wer']['mean']:.2f}%")
            print(f"  • Std Dev: {results['wer']['std']:.2f}%")
            print(f"  • Range: [{results['wer']['min']:.2f}%, {results['wer']['max']:.2f}%]")
            
            print(f"\n📊 Character Error Rate (CER):")
            print(f"  • Mean: {results['cer']['mean']:.2f}%")
            print(f"  • Std Dev: {results['cer']['std']:.2f}%")
            print(f"  • Range: [{results['cer']['min']:.2f}%, {results['cer']['max']:.2f}%]")
            
            print(f"\n📈 Total Samples: {results['num_samples']}")
            print("="*60 + "\n")
        
        return results
    
    @staticmethod
    def calculate_accuracy(
        hypotheses: List[str],
        references: List[str]
    ) -> float:
        """
        Calculate exact match accuracy
        
        Args:
            hypotheses: List of predicted transcriptions
            references: List of reference transcriptions
            
        Returns:
            Accuracy percentage
        """
        matches = sum(
            1 for h, r in zip(hypotheses, references)
            if h.lower() == r.lower()
        )
        return (matches / len(hypotheses)) * 100.0 if hypotheses else 0.0
    
    @staticmethod
    def generate_confusion_matrix(
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[Tuple[str, str], int]:
        """
        Generate confusion matrix for character-level errors
        
        Args:
            hypotheses: List of predicted transcriptions
            references: List of reference transcriptions
            
        Returns:
            Dictionary of (reference_char, hypothesis_char) -> count
        """
        confusion = {}
        
        for hyp, ref in zip(hypotheses, references):
            hyp_chars = list(hyp.lower().replace(" ", ""))
            ref_chars = list(ref.lower().replace(" ", ""))
            
            # Align sequences
            matcher = difflib.SequenceMatcher(None, ref_chars, hyp_chars)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace':
                    for k in range(i2 - i1):
                        ref_char = ref_chars[i1 + k]
                        hyp_char = hyp_chars[j1 + k] if j1 + k < j2 else '∅'
                        key = (ref_char, hyp_char)
                        confusion[key] = confusion.get(key, 0) + 1
        
        return confusion
    
    @staticmethod
    def save_evaluation_results(
        results: Dict,
        output_path: str
    ):
        """Save evaluation results to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy values to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        with open(output_path, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)
        
        print(f"✅ Results saved to {output_path}")


def demonstrate_evaluator():
    """Demonstrate evaluator capabilities"""
    print("\n" + "="*60)
    print("EVALUATION METRICS DEMONSTRATION")
    print("="*60)
    
    # Example transcriptions
    references = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test",
        "automatic speech recognition"
    ]
    
    hypotheses = [
        "the quick brown fox jumps over the lazy dog",  # Perfect match
        "hello word this is a test",  # One substitution
        "automatic speech recopgnition test"  # One substitution, one insertion
    ]
    
    print("\n📝 Example Transcriptions:")
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        print(f"\n  Sample {i+1}:")
        print(f"    Reference:  {ref}")
        print(f"    Hypothesis: {hyp}")
        
        wer, wer_ops = Evaluator.calculate_wer(hyp, ref)
        cer, cer_ops = Evaluator.calculate_cer(hyp, ref)
        
        print(f"    WER: {wer:.2f}% | CER: {cer:.2f}%")
        print(f"    Operations - S:{wer_ops['substitutions']}, D:{wer_ops['deletions']}, I:{wer_ops['insertions']}")
    
    # Batch evaluation
    results = Evaluator.evaluate_batch(hypotheses, references, verbose=True)
    
    # Calculate accuracy
    accuracy = Evaluator.calculate_accuracy(hypotheses, references)
    print(f"✅ Exact Match Accuracy: {accuracy:.2f}%\n")


if __name__ == "__main__":
    demonstrate_evaluator()
