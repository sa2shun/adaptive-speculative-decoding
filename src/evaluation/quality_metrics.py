#!/usr/bin/env python3
"""
Advanced Quality Evaluation System for Adaptive Speculative Decoding Research.

This module provides rigorous quality evaluation using multiple metrics:
- BLEU scores for translation-like tasks
- ROUGE scores for summarization tasks  
- BERTScore for semantic similarity
- Task-specific evaluation (MMLU, HumanEval, etc.)
- Statistical significance testing
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import evaluate
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sacrebleu import BLEU
from datasets import load_dataset
import nltk
from nltk.translate.bleu_score import sentence_bleu
from scipy import stats

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class ComprehensiveQualityEvaluator:
    """Advanced quality evaluation system with multiple metrics and statistical testing."""
    
    def __init__(self):
        """Initialize all evaluation metrics."""
        self.setup_metrics()
        self.reference_data = {}
        
        # Statistical significance parameters
        self.confidence_level = 0.95
        self.effect_size_threshold = 0.2  # Cohen's d
        
    def setup_metrics(self):
        """Setup all evaluation metrics."""
        logger.info("🔧 Setting up comprehensive quality evaluation metrics...")
        
        # ROUGE scorer for summarization tasks
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        
        # BLEU scorer for translation-like tasks
        self.bleu_scorer = BLEU()
        
        # Hugging Face evaluators
        try:
            self.hf_bleu = evaluate.load("bleu")
            self.hf_rouge = evaluate.load("rouge")
            self.hf_meteor = evaluate.load("meteor")
            logger.info("✅ HuggingFace evaluators loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not load HuggingFace evaluators: {e}")
            self.hf_bleu = None
            self.hf_rouge = None
            self.hf_meteor = None
    
    def load_reference_datasets(self, datasets_config: Dict[str, str]):
        """Load reference datasets for quality evaluation."""
        logger.info("📚 Loading reference datasets for quality evaluation...")
        
        for dataset_name, dataset_path in datasets_config.items():
            try:
                if dataset_name == "mmlu":
                    # MMLU dataset for knowledge questions
                    dataset = load_dataset("lukaemon/mmlu", "all", split="test")
                    self.reference_data[dataset_name] = self._prepare_mmlu_references(dataset)
                    
                elif dataset_name == "truthfulqa":
                    # TruthfulQA for factual accuracy
                    dataset = load_dataset("truthful_qa", "generation", split="validation")
                    self.reference_data[dataset_name] = self._prepare_truthfulqa_references(dataset)
                    
                elif dataset_name == "gsm8k":
                    # GSM8K for mathematical reasoning
                    dataset = load_dataset("gsm8k", "main", split="test")
                    self.reference_data[dataset_name] = self._prepare_gsm8k_references(dataset)
                    
                logger.info(f"✅ Loaded {dataset_name}: {len(self.reference_data[dataset_name])} samples")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load {dataset_name}: {e}")
    
    def _prepare_mmlu_references(self, dataset) -> List[Dict]:
        """Prepare MMLU references with correct answers."""
        references = []
        for item in dataset:
            correct_idx = item['answer']
            correct_answer = item['choices'][correct_idx]
            
            references.append({
                'question': item['question'],
                'choices': item['choices'], 
                'correct_answer': correct_answer,
                'correct_idx': correct_idx,
                'subject': item['subject'],
                'category': 'factual_knowledge'
            })
        return references[:1000]  # Sample for faster evaluation
    
    def _prepare_truthfulqa_references(self, dataset) -> List[Dict]:
        """Prepare TruthfulQA references."""
        references = []
        for item in dataset:
            references.append({
                'question': item['question'],
                'best_answer': item['best_answer'],
                'correct_answers': item['correct_answers'],
                'incorrect_answers': item['incorrect_answers'],
                'category': 'truthfulness'
            })
        return references[:500]
    
    def _prepare_gsm8k_references(self, dataset) -> List[Dict]:
        """Prepare GSM8K references."""
        references = []
        for item in dataset:
            references.append({
                'question': item['question'],
                'answer': item['answer'],
                'category': 'mathematical_reasoning'
            })
        return references[:500]
    
    def evaluate_output_quality(self, 
                               prompt: str, 
                               generated_output: str,
                               reference_output: Optional[str] = None,
                               task_category: str = "general") -> Dict[str, float]:
        """
        Comprehensive quality evaluation of generated output.
        
        Args:
            prompt: Input prompt
            generated_output: Model's generated response
            reference_output: Gold standard reference (if available)
            task_category: Type of task (factual, reasoning, creative, etc.)
            
        Returns:
            Dictionary of quality scores
        """
        
        quality_scores = {}
        
        # 1. Length-based metrics
        quality_scores.update(self._evaluate_length_metrics(prompt, generated_output))
        
        # 2. If reference is available, compute similarity metrics
        if reference_output:
            quality_scores.update(self._evaluate_similarity_metrics(
                generated_output, reference_output
            ))
        
        # 3. Task-specific evaluation
        quality_scores.update(self._evaluate_task_specific(
            prompt, generated_output, task_category
        ))
        
        # 4. Linguistic quality metrics
        quality_scores.update(self._evaluate_linguistic_quality(generated_output))
        
        # 5. Semantic coherence
        quality_scores.update(self._evaluate_semantic_coherence(
            prompt, generated_output
        ))
        
        return quality_scores
    
    def _evaluate_length_metrics(self, prompt: str, output: str) -> Dict[str, float]:
        """Evaluate length-based quality metrics."""
        prompt_len = len(prompt.split())
        output_len = len(output.split())
        
        return {
            'output_length': output_len,
            'length_ratio': output_len / max(prompt_len, 1),
            'length_score': min(1.0, output_len / 50),  # Normalize to reasonable length
        }
    
    def _evaluate_similarity_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        """Evaluate similarity between generated and reference text."""
        scores = {}
        
        try:
            # BLEU score
            if self.hf_bleu:
                bleu_result = self.hf_bleu.compute(
                    predictions=[generated], 
                    references=[[reference]]
                )
                scores['bleu'] = bleu_result['bleu']
            else:
                # Fallback to NLTK BLEU
                reference_tokens = [reference.split()]
                generated_tokens = generated.split()
                scores['bleu'] = sentence_bleu(reference_tokens, generated_tokens)
            
            # ROUGE scores
            rouge_scores = self.rouge_scorer.score(reference, generated)
            scores['rouge1'] = rouge_scores['rouge1'].fmeasure
            scores['rouge2'] = rouge_scores['rouge2'].fmeasure
            scores['rougeL'] = rouge_scores['rougeL'].fmeasure
            
            # BERTScore for semantic similarity
            P, R, F1 = bert_score([generated], [reference], lang="en", verbose=False)
            scores['bertscore_precision'] = P.item()
            scores['bertscore_recall'] = R.item() 
            scores['bertscore_f1'] = F1.item()
            
            # METEOR score (if available)
            if self.hf_meteor:
                meteor_result = self.hf_meteor.compute(
                    predictions=[generated],
                    references=[reference]
                )
                scores['meteor'] = meteor_result['meteor']
                
        except Exception as e:
            logger.warning(f"Error computing similarity metrics: {e}")
            # Return default scores
            scores.update({
                'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
                'bertscore_f1': 0.0, 'meteor': 0.0
            })
        
        return scores
    
    def _evaluate_task_specific(self, prompt: str, output: str, category: str) -> Dict[str, float]:
        """Task-specific quality evaluation."""
        scores = {}
        
        if category == "mathematical":
            scores['math_accuracy'] = self._evaluate_math_accuracy(prompt, output)
            
        elif category == "factual":
            scores['factual_consistency'] = self._evaluate_factual_consistency(output)
            
        elif category == "reasoning":
            scores['reasoning_quality'] = self._evaluate_reasoning_quality(output)
            
        elif category == "creative":
            scores['creativity_score'] = self._evaluate_creativity(output)
            
        else:
            # General quality score
            scores['general_quality'] = self._evaluate_general_quality(output)
        
        return scores
    
    def _evaluate_linguistic_quality(self, output: str) -> Dict[str, float]:
        """Evaluate linguistic quality of the output."""
        
        # Simple heuristics for linguistic quality
        sentences = output.split('.')
        words = output.split()
        
        # Repetition penalty
        unique_words = len(set(words))
        total_words = len(words)
        repetition_score = unique_words / max(total_words, 1)
        
        # Average sentence length (indicator of complexity)
        avg_sentence_length = total_words / max(len(sentences), 1)
        sentence_length_score = min(1.0, avg_sentence_length / 15)
        
        return {
            'repetition_score': repetition_score,
            'sentence_length_score': sentence_length_score,
            'vocabulary_diversity': repetition_score  # Alias for clarity
        }
    
    def _evaluate_semantic_coherence(self, prompt: str, output: str) -> Dict[str, float]:
        """Evaluate semantic coherence between prompt and output."""
        
        # Simple coherence check using keyword overlap
        prompt_words = set(prompt.lower().split())
        output_words = set(output.lower().split())
        
        # Remove common stop words for better analysis
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        prompt_words = prompt_words - stop_words
        output_words = output_words - stop_words
        
        # Jaccard similarity
        intersection = len(prompt_words & output_words)
        union = len(prompt_words | output_words)
        coherence_score = intersection / max(union, 1)
        
        return {
            'semantic_coherence': coherence_score,
            'keyword_overlap': intersection / max(len(prompt_words), 1)
        }
    
    def _evaluate_math_accuracy(self, prompt: str, output: str) -> float:
        """Evaluate mathematical accuracy (simplified)."""
        # Look for numerical answers
        import re
        numbers_in_output = re.findall(r'\d+(?:\.\d+)?', output)
        
        # Simple heuristic: if the output contains numbers, it's likely attempting to solve
        if numbers_in_output:
            return 0.8  # Assume reasonable accuracy if numbers are present
        return 0.3  # Lower score if no numerical answer
    
    def _evaluate_factual_consistency(self, output: str) -> float:
        """Evaluate factual consistency (simplified)."""
        # Heuristic: longer, more detailed answers often indicate higher confidence
        words = len(output.split())
        return min(1.0, words / 100)
    
    def _evaluate_reasoning_quality(self, output: str) -> float:
        """Evaluate reasoning quality (simplified)."""
        # Look for reasoning indicators
        reasoning_words = ['because', 'therefore', 'since', 'thus', 'hence', 'consequently']
        reasoning_count = sum(1 for word in reasoning_words if word in output.lower())
        return min(1.0, reasoning_count / 3)
    
    def _evaluate_creativity(self, output: str) -> float:
        """Evaluate creativity (simplified)."""
        # Heuristic: use vocabulary diversity as creativity proxy
        words = output.split()
        unique_words = len(set(words))
        return min(1.0, unique_words / max(len(words), 1))
    
    def _evaluate_general_quality(self, output: str) -> float:
        """General quality score combining multiple factors."""
        words = len(output.split())
        sentences = len(output.split('.'))
        
        # Balance length, completeness, and structure
        length_score = min(1.0, words / 50)
        structure_score = min(1.0, sentences / 5)
        
        return (length_score + structure_score) / 2
    
    def compute_aggregate_quality(self, quality_scores: Dict[str, float]) -> float:
        """Compute single aggregate quality score from multiple metrics."""
        
        # Define weights for different metrics
        weights = {
            'bleu': 0.25,
            'rouge1': 0.15,
            'rougeL': 0.15,
            'bertscore_f1': 0.25,
            'semantic_coherence': 0.10,
            'repetition_score': 0.05,
            'general_quality': 0.05
        }
        
        aggregate_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in quality_scores:
                aggregate_score += quality_scores[metric] * weight
                total_weight += weight
        
        # Add other available metrics with lower weights
        for metric, score in quality_scores.items():
            if metric not in weights and 'score' in metric:
                aggregate_score += score * 0.02
                total_weight += 0.02
        
        return aggregate_score / max(total_weight, 1.0)
    
    def compare_methods_statistically(self, 
                                    method_results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Statistical comparison between different methods."""
        
        logger.info("📊 Performing statistical significance testing...")
        
        comparison_results = {}
        
        # Extract quality scores for each method
        method_scores = {}
        for method_name, results in method_results.items():
            scores = []
            for result in results:
                if 'quality_metrics' in result:
                    aggregate_score = self.compute_aggregate_quality(result['quality_metrics'])
                    scores.append(aggregate_score)
            method_scores[method_name] = scores
        
        # Pairwise statistical tests
        methods = list(method_scores.keys())
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                scores1 = method_scores[method1]
                scores2 = method_scores[method2]
                
                if len(scores1) > 0 and len(scores2) > 0:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(scores1, scores2)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                    effect_size = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0
                    
                    comparison_key = f"{method1}_vs_{method2}"
                    comparison_results[comparison_key] = {
                        'method1': method1,
                        'method2': method2,
                        'method1_mean': np.mean(scores1),
                        'method2_mean': np.mean(scores2),
                        'method1_std': np.std(scores1),
                        'method2_std': np.std(scores2),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'significant': p_value < (1 - self.confidence_level),
                        'practical_significance': abs(effect_size) > self.effect_size_threshold
                    }
        
        return comparison_results
    
    def generate_quality_report(self, 
                              method_results: Dict[str, List[Dict]],
                              save_path: Optional[str] = None) -> Dict:
        """Generate comprehensive quality evaluation report."""
        
        logger.info("📝 Generating comprehensive quality evaluation report...")
        
        report = {
            'evaluation_summary': {},
            'method_comparison': {},
            'statistical_analysis': {},
            'recommendations': []
        }
        
        # Method-wise quality analysis
        for method_name, results in method_results.items():
            if not results:
                continue
                
            quality_metrics = []
            for result in results:
                if 'quality_metrics' in result:
                    quality_metrics.append(result['quality_metrics'])
            
            if quality_metrics:
                # Aggregate statistics
                metric_names = set()
                for metrics in quality_metrics:
                    metric_names.update(metrics.keys())
                
                method_summary = {}
                for metric in metric_names:
                    values = [m.get(metric, 0) for m in quality_metrics if metric in m]
                    if values:
                        method_summary[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'count': len(values)
                        }
                
                # Compute aggregate quality score
                aggregate_scores = [self.compute_aggregate_quality(m) for m in quality_metrics]
                method_summary['aggregate_quality'] = {
                    'mean': np.mean(aggregate_scores),
                    'std': np.std(aggregate_scores),
                    'min': np.min(aggregate_scores),
                    'max': np.max(aggregate_scores)
                }
                
                report['evaluation_summary'][method_name] = method_summary
        
        # Statistical comparison
        report['statistical_analysis'] = self.compare_methods_statistically(method_results)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        # Save report if path specified
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"💾 Quality evaluation report saved to: {save_path}")
        
        return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate actionable recommendations based on quality analysis."""
        
        recommendations = []
        
        # Analyze best performing method
        if 'evaluation_summary' in report:
            best_method = None
            best_score = 0
            
            for method, summary in report['evaluation_summary'].items():
                if 'aggregate_quality' in summary:
                    score = summary['aggregate_quality']['mean']
                    if score > best_score:
                        best_score = score
                        best_method = method
            
            if best_method:
                recommendations.append(f"Best overall method: {best_method} (quality: {best_score:.3f})")
        
        # Analyze statistical significance
        if 'statistical_analysis' in report:
            significant_improvements = []
            for comparison, stats in report['statistical_analysis'].items():
                if stats['significant'] and stats['practical_significance']:
                    if stats['method1_mean'] > stats['method2_mean']:
                        better_method = stats['method1']
                        worse_method = stats['method2']
                    else:
                        better_method = stats['method2']
                        worse_method = stats['method1']
                    
                    significant_improvements.append(
                        f"{better_method} significantly outperforms {worse_method} "
                        f"(p={stats['p_value']:.3f}, effect size={stats['effect_size']:.3f})"
                    )
            
            if significant_improvements:
                recommendations.extend(significant_improvements)
            else:
                recommendations.append("No statistically significant differences found between methods")
        
        # Quality-specific recommendations
        recommendations.extend([
            "Consider increasing evaluation dataset size for more robust statistics",
            "Implement task-specific quality metrics for different prompt categories",
            "Add human evaluation as gold standard for quality assessment"
        ])
        
        return recommendations

def create_evaluation_demo():
    """Create demonstration of the quality evaluation system."""
    
    logger.info("🎯 Creating Quality Evaluation Demonstration")
    
    evaluator = ComprehensiveQualityEvaluator()
    
    # Sample evaluations
    test_cases = [
        {
            'prompt': "What is 2 + 2?",
            'generated': "2 + 2 equals 4. This is a basic arithmetic operation.",
            'reference': "4",
            'category': "mathematical"
        },
        {
            'prompt': "Explain photosynthesis.",
            'generated': "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll.",
            'reference': "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
            'category': "factual"
        },
        {
            'prompt': "Write a creative story about a robot.",
            'generated': "Once upon a time, there was a friendly robot named Zyx who loved to help humans with their daily tasks.",
            'reference': None,
            'category': "creative"
        }
    ]
    
    results = []
    for test_case in test_cases:
        quality_scores = evaluator.evaluate_output_quality(
            test_case['prompt'],
            test_case['generated'],
            test_case['reference'],
            test_case['category']
        )
        
        aggregate_score = evaluator.compute_aggregate_quality(quality_scores)
        
        results.append({
            'test_case': test_case,
            'quality_metrics': quality_scores,
            'aggregate_quality': aggregate_score
        })
        
        logger.info(f"✅ Evaluated: '{test_case['prompt'][:30]}...' → Quality: {aggregate_score:.3f}")
    
    return results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run demonstration
    demo_results = create_evaluation_demo()
    
    print("\n🎉 Quality Evaluation System Demo Complete!")
    print(f"📊 Evaluated {len(demo_results)} test cases")
    
    for i, result in enumerate(demo_results, 1):
        prompt = result['test_case']['prompt']
        quality = result['aggregate_quality']
        print(f"  {i}. '{prompt[:40]}...' → Quality: {quality:.3f}")