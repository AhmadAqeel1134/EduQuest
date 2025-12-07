"""
Evaluation metrics for question generation
Implements BLEU, ROUGE, and custom quality metrics
"""

from typing import List, Dict, Any
import re
from collections import Counter
import numpy as np


class QuestionEvaluator:
    """Evaluates generated questions using multiple metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_question(self, generated: str, reference: str = None) -> Dict[str, float]:
        """Evaluate a single generated question"""
        scores = {}
        
        # Format validation
        scores['format_valid'] = self._check_format(generated)
        
        # Quality metrics
        scores['has_question'] = self._has_question_mark(generated)
        scores['has_options'] = self._has_options(generated)
        scores['has_answer'] = self._has_correct_answer(generated)
        scores['length_score'] = self._length_score(generated)
        scores['structure_score'] = self._structure_score(generated)
        
        # If reference provided, calculate similarity
        if reference:
            scores['bleu_score'] = self._bleu_score(generated, reference)
            scores['rouge_score'] = self._rouge_score(generated, reference)
        
        # Overall quality score (0-1)
        scores['overall_quality'] = self._calculate_overall_quality(scores)
        
        return scores
    
    def _check_format(self, text: str) -> float:
        """Check if question follows expected format"""
        has_question = "Question:" in text or "?" in text
        has_options = any(opt in text for opt in ["A)", "B)", "C)", "D)"])
        has_answer = "Correct Answer:" in text or "Answer:" in text
        
        score = sum([has_question, has_options, has_answer]) / 3.0
        return score
    
    def _has_question_mark(self, text: str) -> float:
        """Check if question mark is present"""
        return 1.0 if "?" in text else 0.0
    
    def _has_options(self, text: str) -> float:
        """Check if multiple choice options are present"""
        options = len(re.findall(r'[A-D]\)', text))
        return min(options / 4.0, 1.0)  # Normalize to 0-1
    
    def _has_correct_answer(self, text: str) -> float:
        """Check if correct answer is specified"""
        return 1.0 if re.search(r'Correct Answer:\s*[A-D]', text, re.IGNORECASE) else 0.0
    
    def _length_score(self, text: str) -> float:
        """Score based on appropriate length (50-300 words ideal)"""
        word_count = len(text.split())
        if 50 <= word_count <= 300:
            return 1.0
        elif word_count < 50:
            return word_count / 50.0
        else:
            return max(0.0, 1.0 - (word_count - 300) / 200.0)
    
    def _structure_score(self, text: str) -> float:
        """Score based on structural elements"""
        elements = [
            "Question:" in text,
            any(opt in text for opt in ["A)", "B)", "C)", "D)"]),
            "Correct Answer:" in text,
            "Explanation:" in text or "Explanation" in text
        ]
        return sum(elements) / len(elements)
    
    def _bleu_score(self, generated: str, reference: str) -> float:
        """Simple BLEU score calculation"""
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()
        
        if len(gen_tokens) == 0:
            return 0.0
        
        # Unigram precision
        gen_counts = Counter(gen_tokens)
        ref_counts = Counter(ref_tokens)
        
        matches = sum(min(gen_counts[word], ref_counts[word]) for word in gen_counts)
        precision = matches / len(gen_tokens) if len(gen_tokens) > 0 else 0.0
        
        # Brevity penalty
        if len(gen_tokens) < len(ref_tokens):
            bp = np.exp(1 - len(ref_tokens) / len(gen_tokens))
        else:
            bp = 1.0
        
        return bp * precision
    
    def _rouge_score(self, generated: str, reference: str) -> float:
        """Simple ROUGE-L score (longest common subsequence)"""
        gen_words = generated.lower().split()
        ref_words = reference.lower().split()
        
        if len(gen_words) == 0 or len(ref_words) == 0:
            return 0.0
        
        # Calculate LCS
        lcs = self._lcs(gen_words, ref_words)
        
        precision = lcs / len(gen_words) if len(gen_words) > 0 else 0.0
        recall = lcs / len(ref_words) if len(ref_words) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _lcs(self, seq1: List[str], seq2: List[str]) -> int:
        """Longest Common Subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _calculate_overall_quality(self, scores: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics"""
        weights = {
            'format_valid': 0.2,
            'has_question': 0.1,
            'has_options': 0.2,
            'has_answer': 0.2,
            'length_score': 0.1,
            'structure_score': 0.2
        }
        
        weighted_sum = sum(scores.get(key, 0) * weight for key, weight in weights.items())
        return weighted_sum
    
    def compare_models(self, results: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compare multiple models' outputs"""
        comparison = {
            'model_scores': {},
            'average_quality': {},
            'format_compliance': {},
            'speed': {}
        }
        
        for model_name, questions in results.items():
            scores = []
            format_scores = []
            
            for question in questions:
                eval_result = self.evaluate_question(question)
                scores.append(eval_result['overall_quality'])
                format_scores.append(eval_result['format_valid'])
            
            comparison['model_scores'][model_name] = scores
            comparison['average_quality'][model_name] = np.mean(scores) if scores else 0.0
            comparison['format_compliance'][model_name] = np.mean(format_scores) if format_scores else 0.0
        
        return comparison
    
    def generate_report(self, comparison: Dict[str, Any]) -> str:
        """Generate human-readable evaluation report"""
        report = "=" * 60 + "\n"
        report += "MODEL COMPARISON REPORT\n"
        report += "=" * 60 + "\n\n"
        
        for model_name in comparison['average_quality'].keys():
            report += f"Model: {model_name}\n"
            report += f"  Average Quality Score: {comparison['average_quality'][model_name]:.3f}\n"
            report += f"  Format Compliance: {comparison['format_compliance'][model_name]:.3f}\n"
            report += f"  Number of Questions: {len(comparison['model_scores'][model_name])}\n"
            report += "\n"
        
        # Best model
        best_model = max(comparison['average_quality'], key=comparison['average_quality'].get)
        report += f"Best Performing Model: {best_model}\n"
        report += f"Quality Score: {comparison['average_quality'][best_model]:.3f}\n"
        
        return report
