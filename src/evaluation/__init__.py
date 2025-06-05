"""
Advanced Quality Evaluation Module for Adaptive Speculative Decoding Research.

This module provides comprehensive quality evaluation capabilities:
- Multiple quality metrics (BLEU, ROUGE, BERTScore)
- Statistical significance testing
- Task-specific evaluation
- Research-grade reporting
"""

from .quality_metrics import ComprehensiveQualityEvaluator

__all__ = ['ComprehensiveQualityEvaluator']