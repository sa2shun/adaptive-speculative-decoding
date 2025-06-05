#!/usr/bin/env python3
"""
Advanced Task Taxonomy System for Adaptive Speculative Decoding Research.

This module provides sophisticated task classification beyond simple complexity levels:
- Multi-dimensional task characterization
- Domain-specific classification
- Cognitive load assessment
- Optimal model selection guidance
"""

import logging
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class TaskDomain(Enum):
    """Primary domain classification."""
    MATHEMATICAL = "mathematical"
    LINGUISTIC = "linguistic"
    REASONING = "reasoning"
    FACTUAL = "factual"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"

class TaskComplexity(Enum):
    """Multi-level complexity classification."""
    TRIVIAL = "trivial"          # Single-step, immediate recall
    SIMPLE = "simple"            # 1-2 steps, basic operations
    MODERATE = "moderate"        # 3-5 steps, some reasoning
    COMPLEX = "complex"          # 6-10 steps, multi-hop reasoning
    EXPERT = "expert"            # 10+ steps, deep expertise
    RESEARCH = "research"        # Novel synthesis, creative solutions

class CognitiveLoad(Enum):
    """Cognitive processing requirements."""
    RECALL = "recall"            # Memory retrieval
    COMPREHENSION = "comprehension"  # Understanding concepts
    APPLICATION = "application"   # Applying knowledge
    ANALYSIS = "analysis"        # Breaking down information
    SYNTHESIS = "synthesis"      # Combining ideas creatively
    EVALUATION = "evaluation"    # Critical judgment

@dataclass
class TaskCharacteristics:
    """Comprehensive task characterization."""
    domain: TaskDomain
    complexity: TaskComplexity
    cognitive_load: CognitiveLoad
    
    # Detailed attributes
    requires_computation: bool = False
    requires_creativity: bool = False
    requires_factual_knowledge: bool = False
    requires_reasoning: bool = False
    requires_code_generation: bool = False
    
    # Quantitative measures
    estimated_tokens: int = 0
    estimated_steps: int = 1
    domain_expertise_level: float = 0.0  # 0-1 scale
    
    # Contextual factors
    has_constraints: bool = False
    requires_examples: bool = False
    benefits_from_iteration: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'domain': self.domain.value,
            'complexity': self.complexity.value,
            'cognitive_load': self.cognitive_load.value,
            'requires_computation': self.requires_computation,
            'requires_creativity': self.requires_creativity,
            'requires_factual_knowledge': self.requires_factual_knowledge,
            'requires_reasoning': self.requires_reasoning,
            'requires_code_generation': self.requires_code_generation,
            'estimated_tokens': self.estimated_tokens,
            'estimated_steps': self.estimated_steps,
            'domain_expertise_level': self.domain_expertise_level,
            'has_constraints': self.has_constraints,
            'requires_examples': self.requires_examples,
            'benefits_from_iteration': self.benefits_from_iteration
        }

class AdvancedTaskClassifier:
    """Sophisticated task classification system."""
    
    def __init__(self):
        """Initialize the task classifier with comprehensive patterns."""
        self.setup_classification_patterns()
        self.setup_domain_vocabularies()
        self.setup_complexity_indicators()
        
    def setup_classification_patterns(self):
        """Setup pattern matching for task classification."""
        
        # Domain classification patterns
        self.domain_patterns = {
            TaskDomain.MATHEMATICAL: [
                r'\b(calculate|solve|equation|formula|math|algebra|geometry|calculus)\b',
                r'\b(\d+\s*[+\-*/]\s*\d+|\d+\s*=|\d+\.\d+)\b',
                r'\b(derivative|integral|matrix|vector|probability|statistics)\b',
                r'\b(theorem|proof|lemma|axiom)\b'
            ],
            
            TaskDomain.TECHNICAL: [
                r'\b(implement|code|function|algorithm|api|database|system)\b',
                r'\b(python|javascript|java|c\+\+|sql|html|css)\b',
                r'\b(class|method|variable|object|inheritance|polymorphism)\b',
                r'\b(server|client|network|protocol|architecture)\b'
            ],
            
            TaskDomain.FACTUAL: [
                r'\b(what is|who is|when did|where is|define|explain)\b',
                r'\b(fact|information|data|knowledge|encyclopedia)\b',
                r'\b(history|geography|science|biology|physics|chemistry)\b',
                r'\b(date|year|location|person|event)\b'
            ],
            
            TaskDomain.CREATIVE: [
                r'\b(write|create|design|compose|generate|imagine)\b',
                r'\b(story|poem|essay|article|script|novel)\b',
                r'\b(creative|artistic|original|innovative|unique)\b',
                r'\b(brainstorm|ideate|conceptualize)\b'
            ],
            
            TaskDomain.REASONING: [
                r'\b(analyze|compare|evaluate|assess|judge|critique)\b',
                r'\b(because|therefore|thus|hence|consequently|since)\b',
                r'\b(logical|reasoning|argument|evidence|conclusion)\b',
                r'\b(pros and cons|advantages|disadvantages|trade-off)\b'
            ],
            
            TaskDomain.ANALYTICAL: [
                r'\b(analyze|breakdown|examine|investigate|study)\b',
                r'\b(data|statistics|trends|patterns|correlation)\b',
                r'\b(report|summary|analysis|interpretation)\b',
                r'\b(metrics|performance|optimization|efficiency)\b'
            ]
        }
        
        # Complexity indicators
        self.complexity_patterns = {
            TaskComplexity.TRIVIAL: [
                r'\b(what is \d+\s*[+\-*/]\s*\d+|yes or no|true or false)\b',
                r'\b(capital of|color of|name of)\b'
            ],
            
            TaskComplexity.SIMPLE: [
                r'\b(calculate|convert|translate|define)\b',
                r'^.{1,50}$',  # Very short prompts
                r'\b(simple|basic|elementary)\b'
            ],
            
            TaskComplexity.MODERATE: [
                r'\b(explain|describe|summarize|outline)\b',
                r'^.{51,200}$',  # Medium length prompts
                r'\b(steps|process|procedure|method)\b'
            ],
            
            TaskComplexity.COMPLEX: [
                r'\b(design|implement|develop|create system)\b',
                r'\b(multi-step|complex|advanced|sophisticated)\b',
                r'^.{201,500}$',  # Longer prompts
                r'\b(architecture|framework|comprehensive)\b'
            ],
            
            TaskComplexity.EXPERT: [
                r'\b(research|novel|innovative|cutting-edge)\b',
                r'\b(PhD|doctoral|expert|specialist)\b',
                r'^.{501,}$',  # Very long prompts
                r'\b(state-of-the-art|groundbreaking|revolutionary)\b'
            ]
        }
        
        # Cognitive load patterns
        self.cognitive_patterns = {
            CognitiveLoad.RECALL: [
                r'\b(what is|who is|when|where|define|list)\b',
                r'\b(remember|recall|name|identify)\b'
            ],
            
            CognitiveLoad.COMPREHENSION: [
                r'\b(explain|describe|interpret|summarize)\b',
                r'\b(understand|meaning|significance)\b'
            ],
            
            CognitiveLoad.APPLICATION: [
                r'\b(apply|use|implement|execute|solve)\b',
                r'\b(calculate|compute|demonstrate)\b'
            ],
            
            CognitiveLoad.ANALYSIS: [
                r'\b(analyze|examine|investigate|compare)\b',
                r'\b(breakdown|dissect|study|explore)\b'
            ],
            
            CognitiveLoad.SYNTHESIS: [
                r'\b(create|design|develop|combine|integrate)\b',
                r'\b(synthesize|merge|unify|construct)\b'
            ],
            
            CognitiveLoad.EVALUATION: [
                r'\b(evaluate|assess|judge|critique|review)\b',
                r'\b(better|worse|best|optimal|recommend)\b'
            ]
        }
    
    def setup_domain_vocabularies(self):
        """Setup domain-specific vocabulary for enhanced classification."""
        
        self.domain_vocabularies = {
            TaskDomain.MATHEMATICAL: {
                'keywords': ['algebra', 'geometry', 'calculus', 'statistics', 'probability',
                           'matrix', 'vector', 'equation', 'formula', 'theorem', 'proof'],
                'symbols': ['+', '-', '*', '/', '=', '∫', '∑', '∂', '√', 'π', '∞'],
                'complexity_weight': 1.2
            },
            
            TaskDomain.TECHNICAL: {
                'keywords': ['algorithm', 'data structure', 'API', 'database', 'framework',
                           'architecture', 'design pattern', 'optimization', 'scalability'],
                'languages': ['python', 'java', 'javascript', 'c++', 'sql', 'html', 'css'],
                'complexity_weight': 1.3
            },
            
            TaskDomain.CREATIVE: {
                'keywords': ['story', 'poem', 'narrative', 'character', 'plot', 'creative',
                           'original', 'innovative', 'artistic', 'imaginative'],
                'indicators': ['write', 'create', 'compose', 'design', 'generate'],
                'complexity_weight': 0.9
            },
            
            TaskDomain.FACTUAL: {
                'keywords': ['fact', 'information', 'knowledge', 'history', 'geography',
                           'science', 'definition', 'encyclopedia', 'reference'],
                'question_types': ['what', 'who', 'when', 'where', 'which'],
                'complexity_weight': 0.8
            }
        }
    
    def setup_complexity_indicators(self):
        """Setup indicators for complexity assessment."""
        
        self.complexity_indicators = {
            'length_thresholds': [20, 50, 150, 300, 500],  # Word count thresholds
            'step_keywords': ['first', 'then', 'next', 'finally', 'step', 'phase'],
            'constraint_keywords': ['must', 'should', 'requirement', 'constraint', 'limit'],
            'expertise_keywords': ['advanced', 'expert', 'professional', 'research', 'novel'],
            'iteration_keywords': ['iterate', 'refine', 'improve', 'optimize', 'enhance']
        }
    
    def classify_task(self, prompt: str) -> TaskCharacteristics:
        """Classify a task prompt comprehensively."""
        
        logger.debug(f"Classifying task: '{prompt[:100]}...'")
        
        # Normalize prompt
        prompt_lower = prompt.lower()
        word_count = len(prompt.split())
        
        # Classify domain
        domain = self._classify_domain(prompt_lower)
        
        # Classify complexity
        complexity = self._classify_complexity(prompt_lower, word_count)
        
        # Classify cognitive load
        cognitive_load = self._classify_cognitive_load(prompt_lower)
        
        # Extract detailed characteristics
        characteristics = TaskCharacteristics(
            domain=domain,
            complexity=complexity,
            cognitive_load=cognitive_load
        )
        
        # Set detailed attributes
        self._set_detailed_attributes(characteristics, prompt_lower)
        
        # Set quantitative measures
        self._set_quantitative_measures(characteristics, prompt, word_count)
        
        # Set contextual factors
        self._set_contextual_factors(characteristics, prompt_lower)
        
        return characteristics
    
    def _classify_domain(self, prompt_lower: str) -> TaskDomain:
        """Classify the primary domain of the task."""
        
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt_lower))
                score += matches
            domain_scores[domain] = score
        
        # If no patterns match, default based on content
        if max(domain_scores.values()) == 0:
            # Simple heuristics for fallback
            if any(word in prompt_lower for word in ['what', 'who', 'when', 'where']):
                return TaskDomain.FACTUAL
            elif any(word in prompt_lower for word in ['write', 'create', 'story']):
                return TaskDomain.CREATIVE
            else:
                return TaskDomain.CONVERSATIONAL
        
        return max(domain_scores, key=domain_scores.get)
    
    def _classify_complexity(self, prompt_lower: str, word_count: int) -> TaskComplexity:
        """Classify the complexity level of the task."""
        
        # Length-based initial assessment
        thresholds = self.complexity_indicators['length_thresholds']
        if word_count <= thresholds[0]:
            base_complexity = TaskComplexity.TRIVIAL
        elif word_count <= thresholds[1]:
            base_complexity = TaskComplexity.SIMPLE
        elif word_count <= thresholds[2]:
            base_complexity = TaskComplexity.MODERATE
        elif word_count <= thresholds[3]:
            base_complexity = TaskComplexity.COMPLEX
        else:
            base_complexity = TaskComplexity.EXPERT
        
        # Pattern-based adjustment
        complexity_scores = {}
        for complexity, patterns in self.complexity_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt_lower))
                score += matches
            complexity_scores[complexity] = score
        
        # Use pattern-based if strong signal, otherwise use length-based
        if max(complexity_scores.values()) >= 2:
            return max(complexity_scores, key=complexity_scores.get)
        else:
            return base_complexity
    
    def _classify_cognitive_load(self, prompt_lower: str) -> CognitiveLoad:
        """Classify the cognitive processing requirements."""
        
        cognitive_scores = {}
        
        for cognitive_type, patterns in self.cognitive_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt_lower))
                score += matches
            cognitive_scores[cognitive_type] = score
        
        # Default to comprehension if no clear signals
        if max(cognitive_scores.values()) == 0:
            return CognitiveLoad.COMPREHENSION
        
        return max(cognitive_scores, key=cognitive_scores.get)
    
    def _set_detailed_attributes(self, characteristics: TaskCharacteristics, prompt_lower: str):
        """Set detailed boolean attributes."""
        
        # Computation requirements
        computation_indicators = ['calculate', 'compute', 'solve', 'equation', '+', '-', '*', '/']
        characteristics.requires_computation = any(ind in prompt_lower for ind in computation_indicators)
        
        # Creativity requirements
        creativity_indicators = ['creative', 'original', 'innovative', 'write', 'design', 'story']
        characteristics.requires_creativity = any(ind in prompt_lower for ind in creativity_indicators)
        
        # Factual knowledge requirements
        factual_indicators = ['fact', 'history', 'science', 'who is', 'what is', 'when did']
        characteristics.requires_factual_knowledge = any(ind in prompt_lower for ind in factual_indicators)
        
        # Reasoning requirements
        reasoning_indicators = ['analyze', 'compare', 'because', 'therefore', 'reasoning']
        characteristics.requires_reasoning = any(ind in prompt_lower for ind in reasoning_indicators)
        
        # Code generation requirements
        code_indicators = ['implement', 'code', 'function', 'algorithm', 'python', 'java']
        characteristics.requires_code_generation = any(ind in prompt_lower for ind in code_indicators)
    
    def _set_quantitative_measures(self, characteristics: TaskCharacteristics, prompt: str, word_count: int):
        """Set quantitative measures."""
        
        # Estimate output tokens (rough heuristic)
        if characteristics.complexity == TaskComplexity.TRIVIAL:
            characteristics.estimated_tokens = 10
        elif characteristics.complexity == TaskComplexity.SIMPLE:
            characteristics.estimated_tokens = 50
        elif characteristics.complexity == TaskComplexity.MODERATE:
            characteristics.estimated_tokens = 150
        elif characteristics.complexity == TaskComplexity.COMPLEX:
            characteristics.estimated_tokens = 400
        else:
            characteristics.estimated_tokens = 800
        
        # Estimate processing steps
        step_keywords = self.complexity_indicators['step_keywords']
        step_count = sum(1 for keyword in step_keywords if keyword in prompt.lower())
        
        complexity_multiplier = {
            TaskComplexity.TRIVIAL: 1,
            TaskComplexity.SIMPLE: 2,
            TaskComplexity.MODERATE: 3,
            TaskComplexity.COMPLEX: 5,
            TaskComplexity.EXPERT: 8,
            TaskComplexity.RESEARCH: 12
        }[characteristics.complexity]
        
        characteristics.estimated_steps = max(1, step_count * complexity_multiplier)
        
        # Domain expertise level
        expertise_keywords = self.complexity_indicators['expertise_keywords']
        expertise_signals = sum(1 for keyword in expertise_keywords if keyword in prompt.lower())
        characteristics.domain_expertise_level = min(1.0, expertise_signals * 0.3)
    
    def _set_contextual_factors(self, characteristics: TaskCharacteristics, prompt_lower: str):
        """Set contextual factors."""
        
        # Constraint detection
        constraint_keywords = self.complexity_indicators['constraint_keywords']
        characteristics.has_constraints = any(keyword in prompt_lower for keyword in constraint_keywords)
        
        # Example requirements
        example_indicators = ['example', 'instance', 'sample', 'demonstrate']
        characteristics.requires_examples = any(ind in prompt_lower for ind in example_indicators)
        
        # Iteration benefits
        iteration_keywords = self.complexity_indicators['iteration_keywords']
        characteristics.benefits_from_iteration = any(keyword in prompt_lower for keyword in iteration_keywords)
    
    def get_optimal_model_recommendation(self, characteristics: TaskCharacteristics) -> Dict[str, float]:
        """Recommend optimal model based on task characteristics."""
        
        # Base scores for each model
        model_scores = {
            '13B': 1.0,
            '34B': 1.0,
            '70B': 1.0
        }
        
        # Adjust based on complexity
        complexity_adjustments = {
            TaskComplexity.TRIVIAL: {'13B': 1.5, '34B': 1.0, '70B': 0.8},
            TaskComplexity.SIMPLE: {'13B': 1.3, '34B': 1.1, '70B': 0.9},
            TaskComplexity.MODERATE: {'13B': 1.0, '34B': 1.3, '70B': 1.1},
            TaskComplexity.COMPLEX: {'13B': 0.7, '34B': 1.2, '70B': 1.4},
            TaskComplexity.EXPERT: {'13B': 0.5, '34B': 1.0, '70B': 1.6},
            TaskComplexity.RESEARCH: {'13B': 0.3, '34B': 0.8, '70B': 1.8}
        }
        
        # Apply complexity adjustments
        for model in model_scores:
            model_scores[model] *= complexity_adjustments[characteristics.complexity][model]
        
        # Adjust based on domain
        domain_adjustments = {
            TaskDomain.MATHEMATICAL: {'13B': 0.9, '34B': 1.1, '70B': 1.3},
            TaskDomain.TECHNICAL: {'13B': 0.8, '34B': 1.2, '70B': 1.4},
            TaskDomain.CREATIVE: {'13B': 1.2, '34B': 1.1, '70B': 1.0},
            TaskDomain.FACTUAL: {'13B': 1.3, '34B': 1.0, '70B': 0.9},
            TaskDomain.REASONING: {'13B': 0.7, '34B': 1.1, '70B': 1.5}
        }
        
        if characteristics.domain in domain_adjustments:
            for model in model_scores:
                model_scores[model] *= domain_adjustments[characteristics.domain][model]
        
        # Adjust based on specific requirements
        if characteristics.requires_code_generation:
            model_scores['13B'] *= 0.7
            model_scores['34B'] *= 1.2
            model_scores['70B'] *= 1.4
        
        if characteristics.requires_creativity:
            model_scores['13B'] *= 1.1
            model_scores['34B'] *= 1.0
            model_scores['70B'] *= 0.9
        
        # Normalize scores
        total_score = sum(model_scores.values())
        normalized_scores = {model: score/total_score for model, score in model_scores.items()}
        
        return normalized_scores
    
    def batch_classify_tasks(self, prompts: List[str]) -> List[TaskCharacteristics]:
        """Classify multiple tasks efficiently."""
        
        logger.info(f"🔍 Batch classifying {len(prompts)} tasks...")
        
        results = []
        for i, prompt in enumerate(prompts):
            try:
                characteristics = self.classify_task(prompt)
                results.append(characteristics)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"  Processed {i + 1}/{len(prompts)} tasks")
                    
            except Exception as e:
                logger.warning(f"Error classifying task {i}: {e}")
                # Create default characteristics
                default_chars = TaskCharacteristics(
                    domain=TaskDomain.CONVERSATIONAL,
                    complexity=TaskComplexity.MODERATE,
                    cognitive_load=CognitiveLoad.COMPREHENSION
                )
                results.append(default_chars)
        
        logger.info(f"✅ Batch classification complete: {len(results)} tasks classified")
        return results
    
    def analyze_task_distribution(self, tasks: List[TaskCharacteristics]) -> Dict[str, Any]:
        """Analyze distribution of task characteristics."""
        
        logger.info("📊 Analyzing task distribution...")
        
        # Count distributions
        domain_counts = {}
        complexity_counts = {}
        cognitive_counts = {}
        
        for task in tasks:
            # Domain distribution
            domain_counts[task.domain.value] = domain_counts.get(task.domain.value, 0) + 1
            
            # Complexity distribution
            complexity_counts[task.complexity.value] = complexity_counts.get(task.complexity.value, 0) + 1
            
            # Cognitive load distribution
            cognitive_counts[task.cognitive_load.value] = cognitive_counts.get(task.cognitive_load.value, 0) + 1
        
        # Calculate percentages
        total_tasks = len(tasks)
        
        distribution_analysis = {
            'total_tasks': total_tasks,
            'domain_distribution': {
                domain: {'count': count, 'percentage': count/total_tasks*100}
                for domain, count in domain_counts.items()
            },
            'complexity_distribution': {
                complexity: {'count': count, 'percentage': count/total_tasks*100}
                for complexity, count in complexity_counts.items()
            },
            'cognitive_load_distribution': {
                cognitive: {'count': count, 'percentage': count/total_tasks*100}
                for cognitive, count in cognitive_counts.items()
            }
        }
        
        # Calculate average quantitative measures
        if tasks:
            avg_tokens = np.mean([task.estimated_tokens for task in tasks])
            avg_steps = np.mean([task.estimated_steps for task in tasks])
            avg_expertise = np.mean([task.domain_expertise_level for task in tasks])
            
            distribution_analysis['quantitative_averages'] = {
                'avg_estimated_tokens': avg_tokens,
                'avg_estimated_steps': avg_steps,
                'avg_domain_expertise_level': avg_expertise
            }
        
        # Calculate requirement frequencies
        requirement_counts = {
            'requires_computation': sum(1 for task in tasks if task.requires_computation),
            'requires_creativity': sum(1 for task in tasks if task.requires_creativity),
            'requires_factual_knowledge': sum(1 for task in tasks if task.requires_factual_knowledge),
            'requires_reasoning': sum(1 for task in tasks if task.requires_reasoning),
            'requires_code_generation': sum(1 for task in tasks if task.requires_code_generation)
        }
        
        distribution_analysis['requirement_frequencies'] = {
            req: {'count': count, 'percentage': count/total_tasks*100}
            for req, count in requirement_counts.items()
        }
        
        return distribution_analysis

def create_task_taxonomy_demo():
    """Create demonstration of the advanced task taxonomy system."""
    
    logger.info("🎯 Creating Advanced Task Taxonomy Demonstration")
    
    # Create classifier
    classifier = AdvancedTaskClassifier()
    
    # Test cases covering different dimensions
    test_prompts = [
        # Mathematical tasks
        "Calculate the derivative of x^2 + 3x + 5",
        "Solve the system of equations: 2x + 3y = 7, x - y = 1",
        "Prove that the square root of 2 is irrational",
        
        # Technical tasks
        "Implement a binary search algorithm in Python",
        "Design a REST API for a social media platform",
        "Optimize a database query for better performance",
        
        # Creative tasks
        "Write a short story about a time traveler",
        "Create a poem about artificial intelligence",
        "Design a logo for a green energy company",
        
        # Factual tasks
        "What is the capital of Australia?",
        "Who invented the telephone?",
        "Explain the process of photosynthesis",
        
        # Reasoning tasks
        "Compare the advantages and disadvantages of renewable energy",
        "Analyze the causes of the 2008 financial crisis",
        "Evaluate the ethical implications of genetic engineering",
        
        # Complex multi-domain tasks
        "Design and implement a machine learning system for predicting stock prices, including data preprocessing, model selection, and evaluation metrics",
        "Create a comprehensive business plan for a sustainable food delivery startup, including market analysis, financial projections, and growth strategy"
    ]
    
    # Classify all tasks
    logger.info(f"🔍 Classifying {len(test_prompts)} diverse test tasks...")
    classified_tasks = []
    
    for i, prompt in enumerate(test_prompts, 1):
        characteristics = classifier.classify_task(prompt)
        classified_tasks.append(characteristics)
        
        # Get model recommendation
        model_rec = classifier.get_optimal_model_recommendation(characteristics)
        best_model = max(model_rec, key=model_rec.get)
        confidence = model_rec[best_model]
        
        logger.info(f"  {i:2d}. Domain: {characteristics.domain.value:12} | "
                   f"Complexity: {characteristics.complexity.value:8} | "
                   f"Cognitive: {characteristics.cognitive_load.value:12} | "
                   f"Best Model: {best_model} ({confidence:.2f})")
        logger.info(f"      '{prompt[:60]}...'")
    
    # Analyze distribution
    logger.info("\n📊 Analyzing task distribution...")
    distribution = classifier.analyze_task_distribution(classified_tasks)
    
    logger.info(f"\n🎯 Task Distribution Summary:")
    logger.info(f"Total tasks: {distribution['total_tasks']}")
    
    logger.info(f"\n📈 Domain Distribution:")
    for domain, stats in distribution['domain_distribution'].items():
        logger.info(f"  {domain:15}: {stats['count']:2d} ({stats['percentage']:5.1f}%)")
    
    logger.info(f"\n🔢 Complexity Distribution:")
    for complexity, stats in distribution['complexity_distribution'].items():
        logger.info(f"  {complexity:8}: {stats['count']:2d} ({stats['percentage']:5.1f}%)")
    
    logger.info(f"\n🧠 Cognitive Load Distribution:")
    for cognitive, stats in distribution['cognitive_load_distribution'].items():
        logger.info(f"  {cognitive:12}: {stats['count']:2d} ({stats['percentage']:5.1f}%)")
    
    return classifier, classified_tasks, distribution

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run demonstration
    classifier, tasks, distribution = create_task_taxonomy_demo()
    
    print("\n🎉 Advanced Task Taxonomy Demo Complete!")
    print(f"📊 Classified {len(tasks)} tasks across {len(set(task.domain for task in tasks))} domains")
    print(f"🔬 Identified {len(set(task.complexity for task in tasks))} complexity levels")
    print(f"🧠 Detected {len(set(task.cognitive_load for task in tasks))} cognitive load types")