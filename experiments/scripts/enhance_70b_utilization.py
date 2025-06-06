#!/usr/bin/env python3
"""
Enhanced 70B Model Utilization System.

This module implements advanced complexity detection and routing
to improve 70B model utilization while maintaining efficiency.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import re

logger = logging.getLogger(__name__)

class AdvancedComplexityDetector:
    """Enhanced complexity detection for better 70B utilization."""
    
    def __init__(self):
        self.setup_complexity_patterns()
        
    def setup_complexity_patterns(self):
        """Setup patterns for detecting complex queries."""
        
        # High complexity indicators (force 70B)
        self.force_70b_patterns = [
            r'\b(implement|design|create).*(system|architecture|framework)',
            r'\b(analyze|compare|evaluate).*(multiple|several|various)',
            r'\b(machine learning|deep learning|neural network)',
            r'\b(research|academic|scholarly)',
            r'\b(comprehensive|detailed|thorough).*analysis',
            r'\b(multi-step|complex|sophisticated)',
            r'\b(optimization|algorithm|data structure)',
            r'\b(prove|theorem|mathematical proof)',
            r'\b(explain.*theory|theoretical framework)',
            r'\b(code review|technical specification)',
        ]
        
        # Medium complexity indicators (prefer 34B/70B)
        self.prefer_large_patterns = [
            r'\b(explain|describe).{50,}',  # Long explanations
            r'\b(how.*work|why.*important)',
            r'\b(advantages.*disadvantages|pros.*cons)',
            r'\b(step.*by.*step|process|procedure)',
            r'\b(technical|scientific|academic)',
            r'\b(programming|coding|development)',
        ]
        
        # Quality-critical indicators
        self.quality_critical_patterns = [
            r'\b(legal|medical|financial|safety)',
            r'\b(critical|important|essential)',
            r'\b(production|enterprise|business)',
            r'\b(security|privacy|compliance)',
            r'\b(accuracy|precision|correctness)',
        ]
        
    def detect_complexity_score(self, prompt: str) -> float:
        """Calculate complexity score (0-1) for prompt."""
        
        prompt_lower = prompt.lower()
        score = 0.0
        
        # Base score from length
        word_count = len(prompt.split())
        length_score = min(1.0, word_count / 100)
        score += length_score * 0.3
        
        # Force 70B patterns
        for pattern in self.force_70b_patterns:
            if re.search(pattern, prompt_lower):
                score += 0.4
        
        # Prefer large model patterns
        for pattern in self.prefer_large_patterns:
            if re.search(pattern, prompt_lower):
                score += 0.2
        
        # Quality critical patterns
        for pattern in self.quality_critical_patterns:
            if re.search(pattern, prompt_lower):
                score += 0.3
        
        # Technical complexity indicators
        technical_words = ['algorithm', 'architecture', 'framework', 'implementation', 
                          'optimization', 'scalability', 'performance', 'distributed']
        tech_count = sum(1 for word in technical_words if word in prompt_lower)
        score += min(0.3, tech_count * 0.1)
        
        # Question complexity
        question_words = prompt_lower.count('?')
        if question_words > 1:
            score += 0.1  # Multiple questions indicate complexity
        
        return min(1.0, score)
    
    def should_use_70b(self, prompt: str, quality_threshold: float = 0.6) -> bool:
        """Determine if 70B model should be used."""
        
        complexity_score = self.detect_complexity_score(prompt)
        return complexity_score >= quality_threshold

class Enhanced70BRouter:
    """Enhanced routing system to improve 70B utilization."""
    
    def __init__(self):
        self.complexity_detector = AdvancedComplexityDetector()
        self.utilization_target = 0.4  # Target 40% 70B usage
        self.current_utilization = 0.0
        self.recent_decisions = []
        
    def select_model(self, prompt: str, lambda_param: float = 1.0) -> str:
        """Select optimal model with enhanced 70B utilization."""
        
        # Get complexity score
        complexity = self.complexity_detector.detect_complexity_score(prompt)
        
        # Adjust thresholds based on current utilization
        if self.current_utilization < self.utilization_target:
            # Need more 70B usage, lower threshold
            threshold_70b = max(0.4, 0.6 - (self.utilization_target - self.current_utilization))
            threshold_34b = 0.3
        else:
            # 70B usage is adequate, normal thresholds
            threshold_70b = 0.6
            threshold_34b = 0.4
        
        # Lambda-based adjustment
        if lambda_param > 2.0:
            threshold_70b *= 0.8  # Favor quality, lower threshold
        elif lambda_param < 0.5:
            threshold_70b *= 1.3  # Favor speed, higher threshold
        
        # Model selection
        if complexity >= threshold_70b:
            selected = '70B'
        elif complexity >= threshold_34b:
            selected = '34B'
        else:
            selected = '13B'
        
        # Update utilization tracking
        self.recent_decisions.append(selected)
        if len(self.recent_decisions) > 100:
            self.recent_decisions = self.recent_decisions[-100:]
        
        self.current_utilization = sum(1 for d in self.recent_decisions if d == '70B') / len(self.recent_decisions)
        
        return selected

def demo_enhanced_70b_utilization():
    """Demo of enhanced 70B utilization system."""
    
    logger.info("🎯 Enhanced 70B Utilization Demo")
    
    detector = AdvancedComplexityDetector()
    router = Enhanced70BRouter()
    
    # Test prompts
    test_prompts = [
        # Should prefer 70B
        "Design a comprehensive machine learning system for fraud detection with real-time processing capabilities",
        "Implement a distributed consensus algorithm that handles Byzantine failures in a blockchain network",
        "Analyze the theoretical foundations of quantum computing and compare different qubit technologies",
        
        # Should prefer 34B
        "Explain how neural networks work and their applications in computer vision",
        "Compare the advantages and disadvantages of different database architectures",
        "Describe the process of implementing a REST API with proper authentication",
        
        # Should prefer 13B
        "What is the capital of France?",
        "Calculate 25 * 4",
        "Define machine learning",
    ]
    
    logger.info("\n🔍 Complexity Detection Results:")
    
    model_counts = {'13B': 0, '34B': 0, '70B': 0}
    
    for prompt in test_prompts:
        complexity = detector.detect_complexity_score(prompt)
        should_70b = detector.should_use_70b(prompt)
        selected_model = router.select_model(prompt)
        
        model_counts[selected_model] += 1
        
        logger.info(f"  Complexity: {complexity:.2f} | Model: {selected_model} | "
                   f"Force70B: {should_70b}")
        logger.info(f"    '{prompt[:60]}...'")
    
    logger.info(f"\n📊 Model Distribution:")
    total = len(test_prompts)
    for model, count in model_counts.items():
        percentage = count / total * 100
        logger.info(f"  {model}: {count}/{total} ({percentage:.1f}%)")
    
    logger.info(f"\n🎯 70B Utilization: {router.current_utilization*100:.1f}%")
    
    return router

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    router = demo_enhanced_70b_utilization()
    
    print(f"\n✅ Enhanced 70B utilization: {router.current_utilization*100:.1f}%")