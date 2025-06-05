"""
Essential baseline methods for comparison.

These baselines are theoretically motivated and provide
clear reference points for evaluating our approach.
"""

import numpy as np
import torch
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.minimal_adaptive_decoder import DecodingResult


class BaselineMethod(ABC):
    """Abstract base class for baseline methods."""
    
    @abstractmethod
    def select_stage(self, prompt: str, features: Optional[torch.Tensor] = None) -> int:
        """Select which model stage to use."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Method name for reporting."""
        pass


class OracleBaseline(BaselineMethod):
    """
    Oracle baseline with perfect knowledge of input difficulty.
    
    This provides an upper bound on performance - no practical
    algorithm can exceed this.
    """
    
    def __init__(self, cost_weight: float = 1.0):
        self.cost_weight = cost_weight
        self.stage_costs = [1.0, 2.0, 4.5, 10.0]
        self.stage_qualities = [0.7, 0.8, 0.85, 0.9]
    
    def select_stage(self, prompt: str, features: Optional[torch.Tensor] = None) -> int:
        """Select optimal stage with perfect knowledge."""
        # Oracle knows true difficulty
        difficulty = self._get_true_difficulty(prompt)
        
        # Select stage that maximizes quality - cost_weight * cost
        best_stage = 0
        best_value = -float('inf')
        
        for stage in range(4):
            # Can this stage handle the difficulty?
            if self.stage_qualities[stage] >= difficulty * 0.9:
                value = self.stage_qualities[stage] - self.cost_weight * self.stage_costs[stage]
                if value > best_value:
                    best_value = value
                    best_stage = stage
        
        return best_stage
    
    def _get_true_difficulty(self, prompt: str) -> float:
        """Oracle knowledge of true difficulty."""
        # Simulated oracle based on prompt analysis
        technical_words = sum(1 for word in prompt.split() 
                            if any(term in word.lower() for term in 
                                 ['algorithm', 'theorem', 'proof', 'optimize', 
                                  'implement', 'analyze', 'derive']))
        
        length_factor = min(len(prompt.split()) / 100, 0.3)
        complexity = min(technical_words * 0.15 + length_factor, 1.0)
        
        return complexity
    
    def name(self) -> str:
        return f"Oracle(λ={self.cost_weight})"


class RandomBaseline(BaselineMethod):
    """
    Random stage selection.
    
    This provides a lower bound - any reasonable algorithm
    should outperform random selection.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.stage_probs = [0.4, 0.3, 0.2, 0.1]  # Favor smaller models
    
    def select_stage(self, prompt: str, features: Optional[torch.Tensor] = None) -> int:
        """Randomly select stage."""
        return self.rng.choice(4, p=self.stage_probs)
    
    def name(self) -> str:
        return "Random"


class FixedStageBaseline(BaselineMethod):
    """
    Always use a fixed stage k.
    
    This represents the traditional approach of using
    a single model for all inputs.
    """
    
    def __init__(self, stage: int):
        assert 0 <= stage <= 3, "Stage must be in [0, 3]"
        self.stage = stage
        self.model_names = ["7B", "14B", "32B", "72B"]
    
    def select_stage(self, prompt: str, features: Optional[torch.Tensor] = None) -> int:
        """Always return fixed stage."""
        return self.stage
    
    def name(self) -> str:
        return f"Fixed-{self.model_names[self.stage]}"


class ThresholdBaseline(BaselineMethod):
    """
    Stop when confidence exceeds threshold.
    
    This is a simple but competitive baseline that doesn't
    require learning complex policies.
    """
    
    def __init__(self, threshold: float = 0.8, confidence_model=None):
        self.threshold = threshold
        self.confidence_model = confidence_model
    
    def select_stage(self, prompt: str, features: Optional[torch.Tensor] = None) -> int:
        """Select first stage with confidence above threshold."""
        for stage in range(4):
            # Simulate confidence estimation
            confidence = self._estimate_confidence(prompt, stage, features)
            
            if confidence >= self.threshold or stage == 3:
                return stage
        
        return 3  # Default to largest
    
    def _estimate_confidence(self, prompt: str, stage: int, features: Optional[torch.Tensor]) -> float:
        """Estimate confidence for stage on prompt."""
        if self.confidence_model is not None and features is not None:
            # Use learned confidence model
            with torch.no_grad():
                return self.confidence_model(features).item()
        else:
            # Simple heuristic: larger models are more confident
            base_confidence = 0.6 + stage * 0.1
            
            # Adjust based on prompt length (shorter = higher confidence)
            length_penalty = min(len(prompt.split()) / 100, 0.2)
            
            return min(base_confidence - length_penalty, 1.0)
    
    def name(self) -> str:
        return f"Threshold(θ={self.threshold})"


class CascadeBaseline(BaselineMethod):
    """
    Traditional cascade: try models in order until quality threshold met.
    
    This is similar to our approach but without learned policies.
    """
    
    def __init__(self, quality_threshold: float = 0.85):
        self.quality_threshold = quality_threshold
        self.stage_qualities = [0.7, 0.8, 0.85, 0.9]
    
    def select_stage(self, prompt: str, features: Optional[torch.Tensor] = None) -> int:
        """Select first stage meeting quality threshold."""
        for stage in range(4):
            # Estimate quality for this stage
            estimated_quality = self._estimate_quality(prompt, stage)
            
            if estimated_quality >= self.quality_threshold or stage == 3:
                return stage
        
        return 3
    
    def _estimate_quality(self, prompt: str, stage: int) -> float:
        """Estimate quality heuristically."""
        base_quality = self.stage_qualities[stage]
        
        # Simple prompt difficulty estimation
        difficulty = min(len(prompt.split()) / 50, 0.3)
        
        # Harder prompts reduce quality
        return base_quality * (1 - difficulty * 0.2)
    
    def name(self) -> str:
        return f"Cascade(τ={self.quality_threshold})"


def evaluate_baselines(test_prompts: List[str], 
                      lambda_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0]) -> Dict:
    """
    Evaluate all baselines on test prompts.
    
    Returns comparative results for analysis.
    """
    baselines = []
    
    # Oracle baselines for different lambda values
    for lam in lambda_values:
        baselines.append(OracleBaseline(cost_weight=lam))
    
    # Other baselines
    baselines.extend([
        RandomBaseline(),
        FixedStageBaseline(0),  # Always 7B
        FixedStageBaseline(1),  # Always 14B
        FixedStageBaseline(2),  # Always 32B
        FixedStageBaseline(3),  # Always 72B
        ThresholdBaseline(0.7),
        ThresholdBaseline(0.8),
        ThresholdBaseline(0.9),
        CascadeBaseline(0.8),
        CascadeBaseline(0.85),
    ])
    
    results = {}
    
    for baseline in baselines:
        stages = []
        costs = []
        
        for prompt in test_prompts:
            stage = baseline.select_stage(prompt)
            stages.append(stage)
            
            # Compute cost
            stage_costs = [1.0, 2.0, 4.5, 10.0]
            costs.append(stage_costs[stage])
        
        # Aggregate metrics
        results[baseline.name()] = {
            'avg_stage': np.mean(stages),
            'avg_cost': np.mean(costs),
            'stage_distribution': np.bincount(stages, minlength=4) / len(stages),
            'total_prompts': len(test_prompts)
        }
    
    return results