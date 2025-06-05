#!/usr/bin/env python3
"""
Production Environment Simulator for Adaptive Speculative Decoding Research.

This module simulates realistic production workloads with:
- 24-hour continuous operation patterns
- Dynamic load variations
- Real-world query distributions
- System stress testing
- Performance monitoring
"""

import logging
import time
import threading
import queue
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import random
from pathlib import Path
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class LoadPattern:
    """Represents a traffic load pattern."""
    hour: int
    base_qps: float          # Base queries per second
    surge_multiplier: float  # Peak surge multiplier
    complexity_bias: float   # Bias towards complex queries (0-1)
    
class SystemMetrics:
    """System performance metrics collector."""
    
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.start_time = time.time()
        self.request_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0
        self.latencies = []
        self.throughput_samples = []
        self.gpu_utilization_samples = []
        self.memory_usage_samples = []
        self.cost_samples = []
        self.quality_samples = []
        self.stage_usage = {'13B': 0, '34B': 0, '70B': 0}
        
    def record_request(self, 
                      latency: float, 
                      success: bool, 
                      cost: float, 
                      quality: float, 
                      stage_used: str):
        """Record a request's metrics."""
        self.request_count += 1
        
        if success:
            self.successful_requests += 1
            self.total_latency += latency
            self.latencies.append(latency)
            self.cost_samples.append(cost)
            self.quality_samples.append(quality)
            self.stage_usage[stage_used] += 1
        else:
            self.failed_requests += 1
    
    def record_system_state(self):
        """Record current system state."""
        # GPU utilization (simulated)
        gpu_util = random.uniform(0.3, 0.95)
        self.gpu_utilization_samples.append(gpu_util)
        
        # Memory usage
        memory_info = psutil.virtual_memory()
        self.memory_usage_samples.append(memory_info.percent)
        
        # Calculate current throughput
        current_time = time.time()
        time_elapsed = current_time - self.start_time
        if time_elapsed > 0:
            current_qps = self.successful_requests / time_elapsed
            self.throughput_samples.append(current_qps)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if not self.latencies:
            return {'error': 'No successful requests recorded'}
        
        return {
            'total_requests': self.request_count,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.successful_requests / self.request_count if self.request_count > 0 else 0,
            'avg_latency': np.mean(self.latencies),
            'p50_latency': np.percentile(self.latencies, 50),
            'p95_latency': np.percentile(self.latencies, 95),
            'p99_latency': np.percentile(self.latencies, 99),
            'avg_throughput': np.mean(self.throughput_samples) if self.throughput_samples else 0,
            'avg_cost': np.mean(self.cost_samples),
            'avg_quality': np.mean(self.quality_samples),
            'avg_gpu_utilization': np.mean(self.gpu_utilization_samples) if self.gpu_utilization_samples else 0,
            'avg_memory_usage': np.mean(self.memory_usage_samples) if self.memory_usage_samples else 0,
            'stage_distribution': {
                stage: count / max(self.successful_requests, 1) * 100 
                for stage, count in self.stage_usage.items()
            }
        }

class TrafficGenerator:
    """Generate realistic traffic patterns."""
    
    def __init__(self):
        self.setup_load_patterns()
        self.setup_query_templates()
        
    def setup_load_patterns(self):
        """Setup 24-hour load patterns based on real-world usage."""
        
        # Typical daily pattern for a global service
        self.daily_patterns = [
            # Midnight to 6 AM (low traffic)
            LoadPattern(0, 2.0, 1.2, 0.2),
            LoadPattern(1, 1.5, 1.1, 0.2),
            LoadPattern(2, 1.2, 1.0, 0.3),
            LoadPattern(3, 1.0, 1.0, 0.3),
            LoadPattern(4, 1.2, 1.1, 0.2),
            LoadPattern(5, 2.0, 1.3, 0.2),
            
            # Morning rush (6 AM - 12 PM)
            LoadPattern(6, 5.0, 2.0, 0.4),
            LoadPattern(7, 8.0, 2.5, 0.5),
            LoadPattern(8, 12.0, 3.0, 0.6),
            LoadPattern(9, 15.0, 2.8, 0.7),
            LoadPattern(10, 18.0, 2.5, 0.8),
            LoadPattern(11, 20.0, 2.2, 0.8),
            
            # Afternoon peak (12 PM - 6 PM)
            LoadPattern(12, 25.0, 2.0, 0.9),
            LoadPattern(13, 22.0, 1.8, 0.9),
            LoadPattern(14, 20.0, 2.2, 0.8),
            LoadPattern(15, 18.0, 2.5, 0.7),
            LoadPattern(16, 16.0, 2.8, 0.6),
            LoadPattern(17, 14.0, 3.0, 0.5),
            
            # Evening (6 PM - midnight)
            LoadPattern(18, 12.0, 2.5, 0.4),
            LoadPattern(19, 10.0, 2.0, 0.4),
            LoadPattern(20, 8.0, 1.8, 0.3),
            LoadPattern(21, 6.0, 1.5, 0.3),
            LoadPattern(22, 4.0, 1.3, 0.2),
            LoadPattern(23, 3.0, 1.2, 0.2),
        ]
    
    def setup_query_templates(self):
        """Setup realistic query templates with different complexity levels."""
        
        self.query_templates = {
            'simple': [
                "What is {}?",
                "Define {}",
                "Who is {}?",
                "When was {} invented?",
                "Calculate {} + {}",
                "Convert {} to {}",
                "Is {} true or false?",
                "What color is {}?",
                "How many {} in {}?",
                "Name three types of {}"
            ],
            
            'moderate': [
                "Explain how {} works",
                "Compare {} and {}",
                "What are the advantages of {}?",
                "Describe the process of {}",
                "How do you troubleshoot {}?",
                "What factors affect {}?",
                "Summarize the key points about {}",
                "Why is {} important for {}?",
                "What are common mistakes when {}?",
                "How has {} evolved over time?"
            ],
            
            'complex': [
                "Design a comprehensive strategy for {} that addresses {} while considering {}",
                "Analyze the long-term implications of {} on {} and propose three alternative solutions",
                "Implement a scalable {} system that handles {} requirements and optimizes for {}",
                "Create a detailed technical specification for {} including {}, {}, and {} components",
                "Develop a machine learning pipeline for {} that processes {} and predicts {}",
                "Evaluate the trade-offs between {} and {} approaches for solving {} in {} context",
                "Design an architecture for {} that integrates {} and ensures {} while maintaining {}",
                "Build a comprehensive testing framework for {} that covers {}, {}, and {} scenarios"
            ]
        }
        
        # Topic pools for template filling
        self.topic_pools = {
            'general': ['artificial intelligence', 'climate change', 'renewable energy', 'space exploration', 
                       'quantum computing', 'biotechnology', 'cybersecurity', 'blockchain'],
            'technical': ['microservices', 'containerization', 'distributed systems', 'API design',
                         'database optimization', 'machine learning', 'data pipelines', 'cloud computing'],
            'business': ['customer acquisition', 'market analysis', 'product strategy', 'growth hacking',
                        'digital transformation', 'supply chain', 'risk management', 'innovation'],
            'scientific': ['neural networks', 'genetic algorithms', 'protein folding', 'climate modeling',
                          'particle physics', 'materials science', 'drug discovery', 'renewable energy']
        }
    
    def generate_query(self, complexity_level: str) -> str:
        """Generate a realistic query of specified complexity."""
        
        templates = self.query_templates[complexity_level]
        template = random.choice(templates)
        
        # Choose appropriate topic pool
        topic_pool = random.choice(list(self.topic_pools.values()))
        
        # Fill template with topics
        try:
            # Count number of {} placeholders
            placeholder_count = template.count('{}')
            topics = random.sample(topic_pool, min(placeholder_count, len(topic_pool)))
            
            # Handle cases where we need more topics than available
            while len(topics) < placeholder_count:
                topics.extend(random.sample(topic_pool, min(placeholder_count - len(topics), len(topic_pool))))
            
            filled_query = template.format(*topics[:placeholder_count])
            return filled_query
            
        except (ValueError, IndexError):
            # Fallback for template formatting issues
            return f"Explain {random.choice(topic_pool)} in detail"
    
    def get_current_load_pattern(self, current_hour: int) -> LoadPattern:
        """Get load pattern for current hour."""
        return self.daily_patterns[current_hour % 24]
    
    def should_generate_request(self, current_hour: int, elapsed_seconds: float) -> bool:
        """Determine if a request should be generated now."""
        
        pattern = self.get_current_load_pattern(current_hour)
        
        # Add some surge simulation
        surge_probability = 0.05  # 5% chance of surge per minute
        is_surge = random.random() < surge_probability
        multiplier = pattern.surge_multiplier if is_surge else 1.0
        
        current_qps = pattern.base_qps * multiplier
        
        # Probability of generating request in this second
        probability = current_qps * (1.0 / 60.0)  # Convert to per-second probability
        
        return random.random() < probability
    
    def select_complexity_level(self, complexity_bias: float) -> str:
        """Select complexity level based on bias."""
        
        # Adjust probabilities based on complexity bias
        base_probs = {'simple': 0.6, 'moderate': 0.3, 'complex': 0.1}
        
        # Shift probabilities based on bias
        if complexity_bias > 0.5:
            # Bias towards more complex
            adjustment = (complexity_bias - 0.5) * 2  # 0 to 1
            base_probs['simple'] *= (1 - adjustment * 0.5)
            base_probs['moderate'] *= (1 + adjustment * 0.3)
            base_probs['complex'] *= (1 + adjustment * 2.0)
        else:
            # Bias towards simpler
            adjustment = (0.5 - complexity_bias) * 2  # 0 to 1
            base_probs['simple'] *= (1 + adjustment * 0.5)
            base_probs['moderate'] *= (1 - adjustment * 0.3)
            base_probs['complex'] *= (1 - adjustment * 0.5)
        
        # Normalize probabilities
        total_prob = sum(base_probs.values())
        normalized_probs = {k: v/total_prob for k, v in base_probs.items()}
        
        # Select based on probabilities
        rand = random.random()
        cumulative = 0
        for complexity, prob in normalized_probs.items():
            cumulative += prob
            if rand < cumulative:
                return complexity
        
        return 'moderate'  # Fallback

class AdaptiveSystemSimulator:
    """Simulate the adaptive speculative decoding system."""
    
    def __init__(self):
        self.setup_system_parameters()
        
    def setup_system_parameters(self):
        """Setup system simulation parameters."""
        
        # Model characteristics (based on research results)
        self.model_params = {
            '13B': {
                'base_latency': 120,  # ms
                'base_cost': 1.0,
                'quality_range': (0.75, 0.90),
                'capacity': 50  # requests per second
            },
            '34B': {
                'base_latency': 180,  # ms
                'base_cost': 1.3,
                'quality_range': (0.85, 0.94),
                'capacity': 25  # requests per second
            },
            '70B': {
                'base_latency': 320,  # ms
                'base_cost': 1.8,
                'quality_range': (0.90, 0.96),
                'capacity': 12  # requests per second
            }
        }
        
        # System state
        self.current_load = {'13B': 0, '34B': 0, '70B': 0}
        self.queue_sizes = {'13B': 0, '34B': 0, '70B': 0}
        
        # Dynamic cost adjustment parameters
        self.dynamic_costs_enabled = True
        self.cost_adjustment_factor = 1.0
        
    def select_optimal_model(self, query_complexity: str, system_load: Dict[str, float]) -> str:
        """Select optimal model based on query and system state."""
        
        # Base model selection based on complexity
        complexity_preferences = {
            'simple': {'13B': 0.7, '34B': 0.2, '70B': 0.1},
            'moderate': {'13B': 0.3, '34B': 0.5, '70B': 0.2},
            'complex': {'13B': 0.1, '34B': 0.3, '70B': 0.6}
        }
        
        base_probs = complexity_preferences.get(query_complexity, 
                                              complexity_preferences['moderate'])
        
        # Adjust based on current system load
        adjusted_probs = {}
        for model, prob in base_probs.items():
            load_factor = system_load.get(model, 0.0)
            
            # Reduce probability if model is heavily loaded
            if load_factor > 0.8:
                adjusted_probs[model] = prob * 0.3
            elif load_factor > 0.6:
                adjusted_probs[model] = prob * 0.7
            else:
                adjusted_probs[model] = prob
        
        # Normalize probabilities
        total_prob = sum(adjusted_probs.values())
        if total_prob > 0:
            normalized_probs = {k: v/total_prob for k, v in adjusted_probs.items()}
        else:
            # Fallback if all models are overloaded
            normalized_probs = {'13B': 0.5, '34B': 0.3, '70B': 0.2}
        
        # Select model
        rand = random.random()
        cumulative = 0
        for model, prob in normalized_probs.items():
            cumulative += prob
            if rand < cumulative:
                return model
        
        return '34B'  # Fallback
    
    def simulate_request(self, query: str, complexity: str) -> Dict[str, Any]:
        """Simulate processing a single request."""
        
        # Calculate current system load
        system_load = {}
        for model in self.model_params:
            capacity = self.model_params[model]['capacity']
            current_usage = self.current_load[model]
            system_load[model] = current_usage / capacity
        
        # Select model
        selected_model = self.select_optimal_model(complexity, system_load)
        
        # Get model parameters
        model_config = self.model_params[selected_model]
        
        # Simulate latency with load-based variation
        load_factor = system_load[selected_model]
        latency_multiplier = 1.0 + (load_factor * 2.0)  # Latency increases with load
        latency = model_config['base_latency'] * latency_multiplier
        
        # Add random variation
        latency *= random.uniform(0.8, 1.3)
        latency += random.gauss(0, 10)  # Random noise
        latency = max(10, latency)  # Minimum latency
        
        # Simulate cost with dynamic adjustment
        base_cost = model_config['base_cost']
        if self.dynamic_costs_enabled:
            # Increase cost under high load
            cost_multiplier = 1.0 + (load_factor * 0.5)
            cost = base_cost * cost_multiplier * self.cost_adjustment_factor
        else:
            cost = base_cost
        
        # Simulate quality
        quality_min, quality_max = model_config['quality_range']
        quality = random.uniform(quality_min, quality_max)
        
        # Adjust quality based on load (slight degradation under high load)
        if load_factor > 0.8:
            quality *= 0.95
        elif load_factor > 0.6:
            quality *= 0.98
        
        # Simulate success/failure
        failure_rate = 0.001 + (load_factor * 0.02)  # Higher failure under load
        success = random.random() > failure_rate
        
        return {
            'latency': latency,
            'cost': cost,
            'quality': quality,
            'model_used': selected_model,
            'success': success,
            'system_load': system_load.copy(),
            'complexity': complexity
        }
    
    def update_system_load(self, model: str, delta: float):
        """Update system load for a model."""
        self.current_load[model] = max(0, self.current_load[model] + delta)
    
    def adjust_costs_dynamically(self, global_metrics: Dict[str, Any]):
        """Adjust costs based on system performance."""
        
        if not self.dynamic_costs_enabled:
            return
        
        # Adjust based on average latency
        avg_latency = global_metrics.get('avg_latency', 200)
        target_latency = 200
        
        if avg_latency > target_latency * 1.5:
            # System is struggling, increase cost adjustment to favor faster models
            self.cost_adjustment_factor = min(2.0, self.cost_adjustment_factor * 1.1)
        elif avg_latency < target_latency * 0.8:
            # System is performing well, reduce cost adjustment
            self.cost_adjustment_factor = max(0.5, self.cost_adjustment_factor * 0.95)

class ProductionSimulator:
    """Main production environment simulator."""
    
    def __init__(self, simulation_duration_hours: int = 1.0):
        """Initialize the production simulator."""
        self.simulation_duration = simulation_duration_hours * 3600  # Convert to seconds
        self.traffic_generator = TrafficGenerator()
        self.system_simulator = AdaptiveSystemSimulator()
        self.metrics = SystemMetrics()
        
        # Simulation state
        self.start_time = None
        self.is_running = False
        self.request_queue = queue.Queue()
        self.results = []
        
        # Monitoring
        self.monitoring_interval = 30  # seconds
        self.last_metrics_update = 0
        
    def run_simulation(self) -> Dict[str, Any]:
        """Run the complete production simulation."""
        
        logger.info(f"🚀 Starting production simulation ({self.simulation_duration/3600:.1f} hours)")
        
        self.start_time = time.time()
        self.is_running = True
        self.metrics.reset_metrics()
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        # Main simulation loop
        try:
            while self.is_running:
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Check if simulation should end
                if elapsed >= self.simulation_duration:
                    logger.info("⏰ Simulation duration reached")
                    break
                
                # Simulate current hour based on elapsed time
                current_hour = int(elapsed // 3600) % 24
                
                # Check if we should generate a request
                if self.traffic_generator.should_generate_request(current_hour, elapsed):
                    self._process_request(current_hour, elapsed)
                
                # Small delay to prevent busy waiting
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("🛑 Simulation interrupted by user")
        
        finally:
            self.is_running = False
        
        # Generate final report
        final_metrics = self.metrics.get_summary()
        simulation_report = self._generate_simulation_report(final_metrics)
        
        logger.info(f"✅ Simulation complete! Processed {final_metrics.get('total_requests', 0)} requests")
        
        return simulation_report
    
    def _process_request(self, current_hour: int, elapsed_time: float):
        """Process a single request."""
        
        # Get load pattern for current hour
        load_pattern = self.traffic_generator.get_current_load_pattern(current_hour)
        
        # Select query complexity
        complexity = self.traffic_generator.select_complexity_level(load_pattern.complexity_bias)
        
        # Generate query
        query = self.traffic_generator.generate_query(complexity)
        
        # Simulate request processing
        request_start = time.time()
        result = self.system_simulator.simulate_request(query, complexity)
        
        # Update system load simulation
        model_used = result['model_used']
        processing_time = result['latency'] / 1000.0  # Convert ms to seconds
        self.system_simulator.update_system_load(model_used, processing_time)
        
        # Record metrics
        self.metrics.record_request(
            latency=result['latency'],
            success=result['success'],
            cost=result['cost'],
            quality=result['quality'],
            stage_used=result['model_used']
        )
        
        # Store detailed result
        result['timestamp'] = time.time()
        result['elapsed_time'] = elapsed_time
        result['hour'] = current_hour
        result['query'] = query[:100]  # Truncate for storage
        self.results.append(result)
        
        # Decay system load over time
        for model in self.system_simulator.current_load:
            self.system_simulator.current_load[model] *= 0.999
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        
        while self.is_running:
            current_time = time.time()
            
            if current_time - self.last_metrics_update >= self.monitoring_interval:
                # Record system state
                self.metrics.record_system_state()
                
                # Get current metrics
                current_metrics = self.metrics.get_summary()
                
                # Adjust system parameters based on performance
                self.system_simulator.adjust_costs_dynamically(current_metrics)
                
                # Log status
                if current_metrics.get('total_requests', 0) > 0:
                    logger.info(f"📊 Status: {current_metrics['total_requests']} requests, "
                              f"P95: {current_metrics.get('p95_latency', 0):.0f}ms, "
                              f"QPS: {current_metrics.get('avg_throughput', 0):.1f}")
                
                self.last_metrics_update = current_time
            
            time.sleep(5)  # Monitor every 5 seconds
    
    def _generate_simulation_report(self, final_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive simulation report."""
        
        report = {
            'simulation_config': {
                'duration_hours': self.simulation_duration / 3600,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(time.time()).isoformat()
            },
            'performance_metrics': final_metrics,
            'temporal_analysis': self._analyze_temporal_patterns(),
            'load_analysis': self._analyze_load_patterns(),
            'recommendations': self._generate_recommendations(final_metrics)
        }
        
        return report
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in the simulation."""
        
        if not self.results:
            return {'error': 'No results to analyze'}
        
        # Group results by hour
        hourly_data = {}
        for result in self.results:
            hour = result['hour']
            if hour not in hourly_data:
                hourly_data[hour] = {
                    'latencies': [], 'costs': [], 'qualities': [],
                    'model_usage': {'13B': 0, '34B': 0, '70B': 0},
                    'request_count': 0
                }
            
            if result['success']:
                hourly_data[hour]['latencies'].append(result['latency'])
                hourly_data[hour]['costs'].append(result['cost'])
                hourly_data[hour]['qualities'].append(result['quality'])
                hourly_data[hour]['model_usage'][result['model_used']] += 1
            
            hourly_data[hour]['request_count'] += 1
        
        # Calculate hourly statistics
        hourly_stats = {}
        for hour, data in hourly_data.items():
            if data['latencies']:
                hourly_stats[hour] = {
                    'avg_latency': np.mean(data['latencies']),
                    'avg_cost': np.mean(data['costs']),
                    'avg_quality': np.mean(data['qualities']),
                    'request_count': data['request_count'],
                    'model_distribution': {
                        model: count / len(data['latencies']) * 100
                        for model, count in data['model_usage'].items()
                    }
                }
        
        return {
            'hourly_statistics': hourly_stats,
            'peak_hour': max(hourly_stats.keys(), key=lambda h: hourly_stats[h]['request_count']) if hourly_stats else None,
            'best_performance_hour': min(hourly_stats.keys(), key=lambda h: hourly_stats[h]['avg_latency']) if hourly_stats else None
        }
    
    def _analyze_load_patterns(self) -> Dict[str, Any]:
        """Analyze load patterns and system behavior."""
        
        if not self.results:
            return {'error': 'No results to analyze'}
        
        # Analyze by complexity
        complexity_stats = {}
        for result in self.results:
            complexity = result['complexity']
            if complexity not in complexity_stats:
                complexity_stats[complexity] = {
                    'latencies': [], 'costs': [], 'qualities': [],
                    'model_usage': {'13B': 0, '34B': 0, '70B': 0}
                }
            
            if result['success']:
                complexity_stats[complexity]['latencies'].append(result['latency'])
                complexity_stats[complexity]['costs'].append(result['cost'])
                complexity_stats[complexity]['qualities'].append(result['quality'])
                complexity_stats[complexity]['model_usage'][result['model_used']] += 1
        
        # Calculate statistics by complexity
        complexity_analysis = {}
        for complexity, data in complexity_stats.items():
            if data['latencies']:
                total_requests = len(data['latencies'])
                complexity_analysis[complexity] = {
                    'avg_latency': np.mean(data['latencies']),
                    'avg_cost': np.mean(data['costs']),
                    'avg_quality': np.mean(data['qualities']),
                    'request_count': total_requests,
                    'model_distribution': {
                        model: count / total_requests * 100
                        for model, count in data['model_usage'].items()
                    }
                }
        
        return {
            'complexity_analysis': complexity_analysis,
            'load_balancing_effectiveness': self._calculate_load_balancing_score()
        }
    
    def _calculate_load_balancing_score(self) -> float:
        """Calculate how well the system balanced load across models."""
        
        if not self.results:
            return 0.0
        
        # Count successful requests per model
        model_counts = {'13B': 0, '34B': 0, '70B': 0}
        for result in self.results:
            if result['success']:
                model_counts[result['model_used']] += 1
        
        total_requests = sum(model_counts.values())
        if total_requests == 0:
            return 0.0
        
        # Calculate distribution
        distribution = [count / total_requests for count in model_counts.values()]
        
        # Ideal distribution might be 60% 13B, 30% 34B, 10% 70B for cost-effectiveness
        ideal_distribution = [0.6, 0.3, 0.1]
        
        # Calculate similarity to ideal (inverse of L2 distance)
        distance = np.sqrt(sum((actual - ideal) ** 2 for actual, ideal in zip(distribution, ideal_distribution)))
        similarity = max(0, 1 - distance)
        
        return similarity
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on simulation results."""
        
        recommendations = []
        
        # Performance recommendations
        if metrics.get('p95_latency', 0) > 500:
            recommendations.append("High P95 latency detected. Consider scaling up GPU resources or optimizing model serving.")
        
        if metrics.get('success_rate', 0) < 0.99:
            recommendations.append(f"Success rate is {metrics.get('success_rate', 0)*100:.1f}%. Investigate error patterns and add retry mechanisms.")
        
        # Load balancing recommendations
        stage_dist = metrics.get('stage_distribution', {})
        if stage_dist.get('70B', 0) > 50:
            recommendations.append("70B model usage is high (>50%). Consider improving complexity detection to route more requests to smaller models.")
        
        if stage_dist.get('13B', 0) < 30:
            recommendations.append("13B model usage is low (<30%). Review routing logic to better utilize the fastest model.")
        
        # Cost optimization recommendations
        avg_cost = metrics.get('avg_cost', 0)
        if avg_cost > 1.5:
            recommendations.append(f"Average cost is {avg_cost:.2f}. Implement more aggressive cost optimization strategies.")
        
        # Quality recommendations
        avg_quality = metrics.get('avg_quality', 0)
        if avg_quality < 0.85:
            recommendations.append(f"Average quality is {avg_quality:.3f}. Consider raising quality thresholds or improving model selection.")
        
        return recommendations

def create_production_simulation_demo():
    """Create demonstration of production simulation."""
    
    logger.info("🎯 Creating Production Simulation Demo")
    
    # Run short simulation (5 minutes for demo)
    simulator = ProductionSimulator(simulation_duration_hours=0.08)  # ~5 minutes
    
    # Run simulation
    report = simulator.run_simulation()
    
    # Display results
    logger.info("\n📊 Production Simulation Results:")
    
    perf_metrics = report['performance_metrics']
    logger.info(f"  Total Requests: {perf_metrics.get('total_requests', 0)}")
    logger.info(f"  Success Rate: {perf_metrics.get('success_rate', 0)*100:.1f}%")
    logger.info(f"  Average Latency: {perf_metrics.get('avg_latency', 0):.0f}ms")
    logger.info(f"  P95 Latency: {perf_metrics.get('p95_latency', 0):.0f}ms")
    logger.info(f"  Average Throughput: {perf_metrics.get('avg_throughput', 0):.1f} QPS")
    logger.info(f"  Average Cost: {perf_metrics.get('avg_cost', 0):.2f}")
    logger.info(f"  Average Quality: {perf_metrics.get('avg_quality', 0):.3f}")
    
    logger.info(f"\n🎯 Model Usage Distribution:")
    stage_dist = perf_metrics.get('stage_distribution', {})
    for model, percentage in stage_dist.items():
        logger.info(f"  {model}: {percentage:.1f}%")
    
    logger.info(f"\n💡 Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        logger.info(f"  {i}. {rec}")
    
    return simulator, report

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run demonstration
    simulator, report = create_production_simulation_demo()
    
    print("\n🎉 Production Simulation Demo Complete!")
    print(f"📈 Simulated {report['performance_metrics'].get('total_requests', 0)} requests")
    print(f"⚡ Average latency: {report['performance_metrics'].get('avg_latency', 0):.0f}ms")
    print(f"💰 Average cost: {report['performance_metrics'].get('avg_cost', 0):.2f}")