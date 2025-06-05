#!/usr/bin/env python3
"""
Dynamic Cost Optimization System for Adaptive Speculative Decoding.

This module implements real-time cost optimization that adapts to:
- Current system load and GPU utilization
- Queue lengths and response times
- Time-of-day patterns and demand forecasting
- Quality requirements and SLA constraints
"""

import logging
import time
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
import json
import psutil
import torch

logger = logging.getLogger(__name__)

@dataclass
class SystemState:
    """Current system state for optimization decisions."""
    timestamp: float
    gpu_utilization: Dict[str, float]  # Per-GPU utilization
    memory_usage: Dict[str, float]     # Per-GPU memory usage
    queue_lengths: Dict[str, int]      # Per-model queue lengths
    active_requests: Dict[str, int]    # Currently processing requests
    recent_latencies: Dict[str, List[float]]  # Recent latency samples
    request_rate: float                # Current requests per second
    error_rate: float                  # Recent error rate
    
class PerformanceMonitor:
    """Monitor system performance in real-time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset_metrics()
        
    def reset_metrics(self):
        """Reset all monitoring metrics."""
        self.latency_history = deque(maxlen=self.window_size)
        self.throughput_history = deque(maxlen=self.window_size)
        self.error_history = deque(maxlen=self.window_size)
        self.quality_history = deque(maxlen=self.window_size)
        self.cost_history = deque(maxlen=self.window_size)
        
        # Per-model metrics
        self.model_metrics = {
            '13B': {'latencies': deque(maxlen=self.window_size), 'requests': 0, 'errors': 0},
            '34B': {'latencies': deque(maxlen=self.window_size), 'requests': 0, 'errors': 0},
            '70B': {'latencies': deque(maxlen=self.window_size), 'requests': 0, 'errors': 0}
        }
        
        self.last_update = time.time()
        
    def record_request(self, model: str, latency: float, success: bool, cost: float, quality: float):
        """Record metrics for a completed request."""
        
        # Global metrics
        self.latency_history.append(latency)
        self.cost_history.append(cost)
        
        if success:
            self.quality_history.append(quality)
            self.error_history.append(0)
        else:
            self.error_history.append(1)
        
        # Model-specific metrics
        if model in self.model_metrics:
            self.model_metrics[model]['requests'] += 1
            if success:
                self.model_metrics[model]['latencies'].append(latency)
            else:
                self.model_metrics[model]['errors'] += 1
        
        # Update throughput
        current_time = time.time()
        time_window = current_time - self.last_update
        if time_window >= 1.0:  # Update every second
            current_throughput = len(self.latency_history) / max(time_window, 1.0)
            self.throughput_history.append(current_throughput)
            self.last_update = current_time
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        
        if not self.latency_history:
            return {'error': 'No data available'}
        
        # Global metrics
        metrics = {
            'avg_latency': np.mean(self.latency_history),
            'p95_latency': np.percentile(self.latency_history, 95),
            'p99_latency': np.percentile(self.latency_history, 99),
            'avg_cost': np.mean(self.cost_history) if self.cost_history else 0,
            'avg_quality': np.mean(self.quality_history) if self.quality_history else 0,
            'error_rate': np.mean(self.error_history) if self.error_history else 0,
            'throughput': np.mean(self.throughput_history) if self.throughput_history else 0
        }
        
        # Model-specific metrics
        model_stats = {}
        for model, data in self.model_metrics.items():
            if data['latencies']:
                model_stats[model] = {
                    'avg_latency': np.mean(data['latencies']),
                    'request_count': data['requests'],
                    'error_count': data['errors'],
                    'utilization': len(data['latencies']) / max(self.window_size, 1)
                }
            else:
                model_stats[model] = {
                    'avg_latency': 0, 'request_count': 0, 'error_count': 0, 'utilization': 0
                }
        
        metrics['model_metrics'] = model_stats
        return metrics

class LoadPredictor:
    """Predict future system load based on historical patterns."""
    
    def __init__(self):
        self.hourly_patterns = {}  # Average load by hour
        self.trend_window = deque(maxlen=60)  # Last 60 minutes
        self.seasonal_adjustment = 1.0
        
    def update_patterns(self, current_hour: int, current_load: float):
        """Update load patterns with current data."""
        
        if current_hour not in self.hourly_patterns:
            self.hourly_patterns[current_hour] = []
        
        self.hourly_patterns[current_hour].append(current_load)
        
        # Keep only recent data (last 7 days worth)
        if len(self.hourly_patterns[current_hour]) > 7:
            self.hourly_patterns[current_hour] = self.hourly_patterns[current_hour][-7:]
        
        # Update trend
        self.trend_window.append(current_load)
    
    def predict_load(self, hours_ahead: int = 1) -> float:
        """Predict load for hours_ahead in the future."""
        
        current_hour = time.localtime().tm_hour
        target_hour = (current_hour + hours_ahead) % 24
        
        # Base prediction from historical patterns
        if target_hour in self.hourly_patterns and self.hourly_patterns[target_hour]:
            base_prediction = np.mean(self.hourly_patterns[target_hour])
        else:
            # Fallback to current load if no historical data
            base_prediction = self.trend_window[-1] if self.trend_window else 1.0
        
        # Apply trend adjustment
        if len(self.trend_window) >= 10:
            recent_trend = np.polyfit(range(len(self.trend_window)), list(self.trend_window), 1)[0]
            trend_adjustment = recent_trend * hours_ahead * 60  # Extrapolate trend
            base_prediction += trend_adjustment
        
        # Apply seasonal adjustment
        adjusted_prediction = base_prediction * self.seasonal_adjustment
        
        return max(0.1, adjusted_prediction)  # Minimum load threshold
    
    def get_load_forecast(self, hours_ahead: int = 12) -> Dict[int, float]:
        """Get load forecast for the next several hours."""
        
        forecast = {}
        for h in range(1, hours_ahead + 1):
            forecast[h] = self.predict_load(h)
        
        return forecast

class DynamicCostOptimizer:
    """Main dynamic cost optimization system."""
    
    def __init__(self, optimization_interval: float = 30.0):
        """Initialize the dynamic cost optimizer."""
        
        self.optimization_interval = optimization_interval
        self.performance_monitor = PerformanceMonitor()
        self.load_predictor = LoadPredictor()
        
        # Cost optimization parameters
        self.base_costs = {'13B': 1.0, '34B': 1.3, '70B': 1.8}
        self.current_cost_multipliers = {'13B': 1.0, '34B': 1.0, '70B': 1.0}
        self.lambda_adjustment = 1.0
        
        # Optimization targets
        self.target_latency = 200  # ms
        self.max_error_rate = 0.01  # 1%
        self.min_quality = 0.85
        
        # Optimization state
        self.optimization_history = deque(maxlen=100)
        self.last_optimization = 0
        self.is_optimizing = False
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        logger.info("🎯 Dynamic Cost Optimizer initialized")
    
    def record_request_completion(self, model: str, latency: float, success: bool, 
                                cost: float, quality: float):
        """Record completion of a request for optimization."""
        
        self.performance_monitor.record_request(model, latency, success, cost, quality)
        
        # Update load patterns
        current_hour = time.localtime().tm_hour
        current_metrics = self.performance_monitor.get_current_metrics()
        current_load = current_metrics.get('throughput', 0)
        self.load_predictor.update_patterns(current_hour, current_load)
    
    def get_optimized_costs(self) -> Dict[str, float]:
        """Get current optimized cost multipliers."""
        
        optimized_costs = {}
        for model, base_cost in self.base_costs.items():
            multiplier = self.current_cost_multipliers[model]
            optimized_costs[model] = base_cost * multiplier
        
        return optimized_costs
    
    def get_optimized_lambda(self) -> float:
        """Get current optimized lambda parameter."""
        return self.lambda_adjustment
    
    def _optimization_loop(self):
        """Main optimization loop running in background thread."""
        
        while True:
            try:
                current_time = time.time()
                
                if current_time - self.last_optimization >= self.optimization_interval:
                    self._perform_optimization()
                    self.last_optimization = current_time
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _perform_optimization(self):
        """Perform cost optimization based on current system state."""
        
        if self.is_optimizing:
            return  # Prevent concurrent optimization
        
        self.is_optimizing = True
        
        try:
            logger.debug("🔧 Performing dynamic cost optimization...")
            
            # Get current system metrics
            current_metrics = self.performance_monitor.get_current_metrics()
            
            if 'error' in current_metrics:
                logger.debug("No metrics available for optimization")
                return
            
            # Get system state
            system_state = self._get_system_state()
            
            # Optimize cost multipliers
            new_multipliers = self._optimize_cost_multipliers(current_metrics, system_state)
            
            # Optimize lambda parameter
            new_lambda = self._optimize_lambda_parameter(current_metrics, system_state)
            
            # Apply optimizations
            old_multipliers = self.current_cost_multipliers.copy()
            old_lambda = self.lambda_adjustment
            
            self.current_cost_multipliers = new_multipliers
            self.lambda_adjustment = new_lambda
            
            # Record optimization decision
            optimization_record = {
                'timestamp': time.time(),
                'old_multipliers': old_multipliers,
                'new_multipliers': new_multipliers,
                'old_lambda': old_lambda,
                'new_lambda': new_lambda,
                'metrics': current_metrics,
                'system_state': asdict(system_state)
            }
            
            self.optimization_history.append(optimization_record)
            
            # Log significant changes
            if abs(new_lambda - old_lambda) > 0.1:
                logger.info(f"🎯 Lambda adjusted: {old_lambda:.2f} → {new_lambda:.2f}")
            
            for model in self.base_costs:
                old_mult = old_multipliers[model]
                new_mult = new_multipliers[model]
                if abs(new_mult - old_mult) > 0.05:
                    logger.info(f"💰 {model} cost multiplier: {old_mult:.2f} → {new_mult:.2f}")
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
        
        finally:
            self.is_optimizing = False
    
    def _get_system_state(self) -> SystemState:
        """Get current system state."""
        
        # Simulate GPU monitoring (in real implementation, use nvidia-ml-py)
        gpu_utilization = {
            'gpu_0': np.random.uniform(0.3, 0.9),
            'gpu_1': np.random.uniform(0.3, 0.9),
            'gpu_2': np.random.uniform(0.3, 0.9),
            'gpu_3': np.random.uniform(0.3, 0.9)
        }
        
        memory_usage = {
            'gpu_0': np.random.uniform(0.4, 0.85),
            'gpu_1': np.random.uniform(0.4, 0.85),
            'gpu_2': np.random.uniform(0.4, 0.85),
            'gpu_3': np.random.uniform(0.4, 0.85)
        }
        
        # Simulate queue lengths
        queue_lengths = {
            '13B': np.random.randint(0, 10),
            '34B': np.random.randint(0, 15),
            '70B': np.random.randint(0, 20)
        }
        
        # Get performance metrics
        current_metrics = self.performance_monitor.get_current_metrics()
        
        return SystemState(
            timestamp=time.time(),
            gpu_utilization=gpu_utilization,
            memory_usage=memory_usage,
            queue_lengths=queue_lengths,
            active_requests={'13B': 5, '34B': 8, '70B': 12},  # Simulated
            recent_latencies={},  # Will be filled from metrics
            request_rate=current_metrics.get('throughput', 0),
            error_rate=current_metrics.get('error_rate', 0)
        )
    
    def _optimize_cost_multipliers(self, metrics: Dict[str, Any], 
                                 system_state: SystemState) -> Dict[str, float]:
        """Optimize cost multipliers based on system state."""
        
        new_multipliers = self.current_cost_multipliers.copy()
        
        # Get current performance indicators
        avg_latency = metrics.get('avg_latency', 0)
        error_rate = metrics.get('error_rate', 0)
        avg_quality = metrics.get('avg_quality', 0)
        
        # Calculate system pressure indicators
        avg_gpu_util = np.mean(list(system_state.gpu_utilization.values()))
        avg_queue_length = np.mean(list(system_state.queue_lengths.values()))
        
        # Optimization logic
        
        # 1. If latency is too high, favor faster models
        if avg_latency > self.target_latency * 1.2:
            # Reduce cost of faster models to encourage their use
            new_multipliers['13B'] *= 0.95
            new_multipliers['34B'] *= 0.98
            new_multipliers['70B'] *= 1.05
            
        elif avg_latency < self.target_latency * 0.8:
            # System is performing well, can afford more expensive models
            new_multipliers['13B'] *= 1.02
            new_multipliers['34B'] *= 1.01
            new_multipliers['70B'] *= 0.98
        
        # 2. If GPU utilization is high, increase costs to reduce load
        if avg_gpu_util > 0.85:
            for model in new_multipliers:
                new_multipliers[model] *= 1.05
        elif avg_gpu_util < 0.5:
            for model in new_multipliers:
                new_multipliers[model] *= 0.98
        
        # 3. If error rate is high, be more conservative
        if error_rate > self.max_error_rate:
            # Favor more reliable models
            new_multipliers['13B'] *= 0.9
            new_multipliers['34B'] *= 0.95
            new_multipliers['70B'] *= 1.1
        
        # 4. Queue-based adjustments
        for model, queue_len in system_state.queue_lengths.items():
            if queue_len > 10:
                # Model is overloaded, increase its cost
                new_multipliers[model] *= 1.1
            elif queue_len < 2:
                # Model is underutilized, decrease its cost
                new_multipliers[model] *= 0.98
        
        # 5. Quality-based adjustments
        if avg_quality < self.min_quality:
            # Need higher quality, favor larger models
            new_multipliers['13B'] *= 1.05
            new_multipliers['34B'] *= 0.98
            new_multipliers['70B'] *= 0.95
        
        # Apply constraints
        for model in new_multipliers:
            # Keep multipliers within reasonable bounds
            new_multipliers[model] = np.clip(new_multipliers[model], 0.5, 3.0)
        
        return new_multipliers
    
    def _optimize_lambda_parameter(self, metrics: Dict[str, Any], 
                                 system_state: SystemState) -> float:
        """Optimize lambda parameter for quality-speed tradeoff."""
        
        current_lambda = self.lambda_adjustment
        
        # Get performance indicators
        avg_latency = metrics.get('avg_latency', 0)
        avg_quality = metrics.get('avg_quality', 0)
        avg_cost = metrics.get('avg_cost', 0)
        
        # Calculate desired direction
        lambda_adjustment = 0.0
        
        # 1. Latency pressure
        if avg_latency > self.target_latency * 1.3:
            # Too slow, reduce lambda to favor speed
            lambda_adjustment -= 0.1
        elif avg_latency < self.target_latency * 0.7:
            # Fast enough, can increase lambda for quality
            lambda_adjustment += 0.05
        
        # 2. Quality requirements
        if avg_quality < self.min_quality:
            # Need more quality, increase lambda
            lambda_adjustment += 0.15
        elif avg_quality > 0.95:
            # Quality is very high, can reduce lambda for efficiency
            lambda_adjustment -= 0.05
        
        # 3. Cost pressure
        if avg_cost > 1.6:
            # Costs are high, reduce lambda to favor cheaper models
            lambda_adjustment -= 0.08
        elif avg_cost < 1.2:
            # Costs are low, can increase lambda
            lambda_adjustment += 0.03
        
        # 4. System load consideration
        avg_gpu_util = np.mean(list(system_state.gpu_utilization.values()))
        if avg_gpu_util > 0.9:
            # System under pressure, favor efficiency
            lambda_adjustment -= 0.05
        
        # 5. Predictive adjustment based on forecasted load
        load_forecast = self.load_predictor.get_load_forecast(hours_ahead=2)
        future_load_avg = np.mean(list(load_forecast.values()))
        current_load = system_state.request_rate
        
        if future_load_avg > current_load * 1.5:
            # Load is expected to increase, prepare by favoring efficiency
            lambda_adjustment -= 0.03
        elif future_load_avg < current_load * 0.7:
            # Load is expected to decrease, can favor quality
            lambda_adjustment += 0.02
        
        # Apply adjustment
        new_lambda = current_lambda + lambda_adjustment
        
        # Keep lambda within reasonable bounds
        new_lambda = np.clip(new_lambda, 0.1, 10.0)
        
        return new_lambda
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization performance report."""
        
        if not self.optimization_history:
            return {'error': 'No optimization history available'}
        
        # Calculate optimization effectiveness
        recent_optimizations = list(self.optimization_history)[-10:]  # Last 10 optimizations
        
        # Track how parameters have changed
        lambda_changes = []
        cost_changes = {'13B': [], '34B': [], '70B': []}
        
        for opt in recent_optimizations:
            lambda_change = abs(opt['new_lambda'] - opt['old_lambda'])
            lambda_changes.append(lambda_change)
            
            for model in cost_changes:
                old_mult = opt['old_multipliers'][model]
                new_mult = opt['new_multipliers'][model]
                change = abs(new_mult - old_mult)
                cost_changes[model].append(change)
        
        # Current system performance
        current_metrics = self.performance_monitor.get_current_metrics()
        
        report = {
            'optimization_frequency': len(self.optimization_history),
            'recent_lambda_volatility': np.std(lambda_changes) if lambda_changes else 0,
            'recent_cost_volatility': {
                model: np.std(changes) if changes else 0 
                for model, changes in cost_changes.items()
            },
            'current_multipliers': self.current_cost_multipliers.copy(),
            'current_lambda': self.lambda_adjustment,
            'current_performance': current_metrics,
            'optimization_targets': {
                'target_latency': self.target_latency,
                'max_error_rate': self.max_error_rate,
                'min_quality': self.min_quality
            },
            'load_forecast': self.load_predictor.get_load_forecast(6)
        }
        
        return report

def create_dynamic_cost_optimizer_demo():
    """Create demonstration of the dynamic cost optimizer."""
    
    logger.info("🎯 Creating Dynamic Cost Optimizer Demo")
    
    # Create optimizer
    optimizer = DynamicCostOptimizer(optimization_interval=5.0)  # Optimize every 5 seconds
    
    # Simulate some request completions
    logger.info("📊 Simulating request completions...")
    
    models = ['13B', '34B', '70B']
    complexities = ['simple', 'moderate', 'complex']
    
    # Simulate 100 requests over time
    for i in range(100):
        model = np.random.choice(models)
        complexity = np.random.choice(complexities)
        
        # Simulate request metrics based on model and complexity
        if model == '13B':
            latency = np.random.uniform(80, 150)
            cost = np.random.uniform(0.8, 1.2)
            quality = np.random.uniform(0.75, 0.88)
        elif model == '34B':
            latency = np.random.uniform(120, 200)
            cost = np.random.uniform(1.1, 1.5)
            quality = np.random.uniform(0.82, 0.92)
        else:  # 70B
            latency = np.random.uniform(200, 350)
            cost = np.random.uniform(1.5, 2.1)
            quality = np.random.uniform(0.88, 0.95)
        
        # Add complexity-based variation
        if complexity == 'complex':
            latency *= 1.3
            cost *= 1.2
            quality *= 1.02
        elif complexity == 'simple':
            latency *= 0.8
            cost *= 0.9
            quality *= 0.98
        
        success = np.random.random() > 0.005  # 0.5% failure rate
        
        # Record request
        optimizer.record_request_completion(model, latency, success, cost, quality)
        
        # Small delay to simulate time passage
        if i % 10 == 0:
            time.sleep(0.1)
            logger.debug(f"  Simulated {i+1} requests...")
    
    # Wait for optimization to occur
    logger.info("⏳ Waiting for optimization cycles...")
    time.sleep(8)  # Wait for at least one optimization cycle
    
    # Get optimization report
    report = optimizer.get_optimization_report()
    
    # Display results
    logger.info("\n🎯 Dynamic Cost Optimization Results:")
    
    current_perf = report['current_performance']
    logger.info(f"  Current Performance:")
    logger.info(f"    Average Latency: {current_perf.get('avg_latency', 0):.0f}ms")
    logger.info(f"    P95 Latency: {current_perf.get('p95_latency', 0):.0f}ms")
    logger.info(f"    Error Rate: {current_perf.get('error_rate', 0)*100:.2f}%")
    logger.info(f"    Average Quality: {current_perf.get('avg_quality', 0):.3f}")
    logger.info(f"    Throughput: {current_perf.get('throughput', 0):.1f} QPS")
    
    logger.info(f"\n🔧 Current Optimization Parameters:")
    logger.info(f"    Lambda: {report['current_lambda']:.2f}")
    
    optimized_costs = optimizer.get_optimized_costs()
    logger.info(f"    Optimized Costs:")
    for model, cost in optimized_costs.items():
        multiplier = report['current_multipliers'][model]
        logger.info(f"      {model}: {cost:.2f} (multiplier: {multiplier:.2f})")
    
    logger.info(f"\n📈 Load Forecast (next 6 hours):")
    forecast = report['load_forecast']
    for hour, predicted_load in forecast.items():
        logger.info(f"    +{hour}h: {predicted_load:.1f} QPS")
    
    logger.info(f"\n📊 Optimization Activity:")
    logger.info(f"    Optimization Cycles: {report['optimization_frequency']}")
    logger.info(f"    Lambda Volatility: {report['recent_lambda_volatility']:.3f}")
    logger.info(f"    Cost Volatility:")
    for model, volatility in report['recent_cost_volatility'].items():
        logger.info(f"      {model}: {volatility:.3f}")
    
    return optimizer, report

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run demonstration
    optimizer, report = create_dynamic_cost_optimizer_demo()
    
    print("\n🎉 Dynamic Cost Optimizer Demo Complete!")
    print(f"🎯 Lambda optimized to: {report['current_lambda']:.2f}")
    print(f"⚡ Average latency: {report['current_performance'].get('avg_latency', 0):.0f}ms")
    print(f"💰 Cost optimization active with {report['optimization_frequency']} cycles")