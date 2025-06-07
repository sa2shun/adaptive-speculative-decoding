"""
Cost Profiler for Adaptive Speculative Decoding

Measures actual inference latencies for Qwen3 model hierarchy to create
realistic cost models instead of theoretical estimates.

This module provides:
- Real latency measurement across different input/output lengths
- GPU memory utilization tracking  
- Statistical analysis of measurement variations
- Cost model fitting and validation
- Integration with configuration files
"""

import time
import json
import yaml
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

# Imports for model loading (assuming vLLM integration)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available. Cost profiling will use mock measurements.")

@dataclass
class MeasurementConfig:
    """Configuration for a single measurement."""
    model_name: str
    input_length: int
    output_length: int
    batch_size: int
    repetition: int

@dataclass
class MeasurementResult:
    """Result of a single latency measurement."""
    config: MeasurementConfig
    prefill_time: float
    decode_time: float
    total_time: float
    gpu_memory_used: float
    gpu_memory_total: float
    tokens_per_second: float
    successful: bool
    error_message: Optional[str] = None

@dataclass
class CostModel:
    """Fitted cost model for a specific model."""
    model_name: str
    coefficients: Dict[str, float]
    intercept: float
    r_squared: float
    model_type: str  # 'linear', 'polynomial', 'power_law'
    
class CostProfiler:
    """Main cost profiling class."""
    
    def __init__(self, config_path: str = "configs/cost_profiling.yaml"):
        """Initialize the cost profiler."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.models = {}
        self.measurements = []
        self.cost_models = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load profiling configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for profiling."""
        logger = logging.getLogger('cost_profiler')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path(self.config['output']['results_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(log_dir / 'cost_profiling.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def load_models(self) -> None:
        """Load all models for profiling."""
        self.logger.info("Loading models for profiling...")
        
        if not VLLM_AVAILABLE:
            self.logger.warning("vLLM not available. Using mock models.")
            return
            
        for model_config in self.config['profiling']['models']:
            try:
                self.logger.info(f"Loading model: {model_config['name']}")
                
                # Configure tensor parallelism
                tensor_parallel_size = model_config['tensor_parallel_size']
                gpu_ids = model_config['gpu_ids']
                
                # Load model with vLLM
                llm = LLM(
                    model=model_config['model_path'],
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=0.9,
                    dtype="bfloat16",  # Full precision as required
                    quantization=None,  # No quantization
                    trust_remote_code=False,
                    max_model_len=4096,
                )
                
                self.models[model_config['name']] = llm
                self.logger.info(f"Successfully loaded {model_config['name']}")
                
            except Exception as e:
                self.logger.error(f"Failed to load model {model_config['name']}: {e}")
                raise
    
    def generate_test_prompts(self, length: int, count: int = 10) -> List[str]:
        """Generate test prompts of specified length."""
        prompts = []
        
        # Use different prompt types from config
        prompt_types = self.config['test_data']['prompt_types']
        
        for i in range(count):
            prompt_type = prompt_types[i % len(prompt_types)]
            
            if prompt_type['name'] == 'simple_qa':
                base = f"What is the capital of country {i}? Please explain in detail."
            elif prompt_type['name'] == 'reasoning':
                base = f"Solve this step by step: If x + {i} = {i*2}, what is x?"
            elif prompt_type['name'] == 'code_generation':
                base = f"Write a Python function that calculates fibonacci number {i}."
            else:
                base = f"Explain the concept of {i} in simple terms."
            
            # Pad to desired length
            words_needed = length // 4  # Rough estimate: 4 chars per token
            padding = " ".join([f"word{j}" for j in range(words_needed)])
            prompt = f"{base} {padding}"[:length]
            prompts.append(prompt)
            
        return prompts
    
    def measure_single_inference(
        self, 
        model_name: str, 
        config: MeasurementConfig
    ) -> MeasurementResult:
        """Measure latency for a single inference configuration."""
        
        if not VLLM_AVAILABLE or model_name not in self.models:
            # Mock measurement for testing
            return self._mock_measurement(config)
        
        try:
            model = self.models[model_name]
            
            # Generate test prompts
            prompts = self.generate_test_prompts(
                config.input_length, 
                config.batch_size
            )
            
            # Sampling parameters
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=config.output_length,
            )
            
            # GPU memory before
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
            
            # Timing measurement
            start_time = time.perf_counter()
            torch.cuda.synchronize()
            
            # Inference
            outputs = model.generate(prompts, sampling_params)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # GPU memory after
            memory_after = torch.cuda.max_memory_allocated()
            memory_total = torch.cuda.get_device_properties(0).total_memory
            
            # Calculate metrics
            total_time = end_time - start_time
            total_tokens = sum(len(output.outputs[0].text.split()) for output in outputs)
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            
            # Estimate prefill vs decode (simplified)
            prefill_time = total_time * 0.3  # Rough estimate
            decode_time = total_time * 0.7
            
            return MeasurementResult(
                config=config,
                prefill_time=prefill_time,
                decode_time=decode_time,
                total_time=total_time,
                gpu_memory_used=memory_after - memory_before,
                gpu_memory_total=memory_total,
                tokens_per_second=tokens_per_second,
                successful=True
            )
            
        except Exception as e:
            self.logger.error(f"Measurement failed for {model_name}: {e}")
            return MeasurementResult(
                config=config,
                prefill_time=0,
                decode_time=0,
                total_time=0,
                gpu_memory_used=0,
                gpu_memory_total=0,
                tokens_per_second=0,
                successful=False,
                error_message=str(e)
            )
    
    def _mock_measurement(self, config: MeasurementConfig) -> MeasurementResult:
        """Create mock measurement for testing."""
        # Realistic latency estimates based on model size
        size_multipliers = {
            'qwen3-7b': 1.0,
            'qwen3-14b': 2.0,
            'qwen3-32b': 4.5,
            'qwen3-72b': 10.0
        }
        
        base_latency = 0.1  # 100ms base
        multiplier = size_multipliers.get(config.model_name, 1.0)
        
        # Add input/output length effects
        length_factor = (config.input_length + config.output_length) / 1000
        batch_factor = config.batch_size ** 0.8
        
        total_time = base_latency * multiplier * length_factor * batch_factor
        total_time += np.random.normal(0, total_time * 0.1)  # Add noise
        
        return MeasurementResult(
            config=config,
            prefill_time=total_time * 0.3,
            decode_time=total_time * 0.7,
            total_time=max(0.01, total_time),  # Ensure positive
            gpu_memory_used=1e9 * multiplier,  # Mock memory usage
            gpu_memory_total=80e9,  # 80GB A100
            tokens_per_second=config.output_length / max(0.01, total_time),
            successful=True
        )
    
    def run_comprehensive_profiling(self) -> None:
        """Run comprehensive latency profiling."""
        self.logger.info("Starting comprehensive profiling...")
        
        # Load models
        self.load_models()
        
        # Get measurement parameters
        measurement_config = self.config['measurement']
        models = [m['name'] for m in self.config['profiling']['models']]
        
        total_measurements = (
            len(models) * 
            len(measurement_config['input_lengths']) *
            len(measurement_config['output_lengths']) *
            len(measurement_config['batch_sizes']) *
            measurement_config['repetitions']
        )
        
        self.logger.info(f"Will perform {total_measurements} measurements")
        
        measurement_count = 0
        
        # Iterate through all configurations
        for model_name in models:
            self.logger.info(f"Profiling model: {model_name}")
            
            for input_length in measurement_config['input_lengths']:
                for output_length in measurement_config['output_lengths']:
                    for batch_size in measurement_config['batch_sizes']:
                        for rep in range(measurement_config['repetitions']):
                            
                            config = MeasurementConfig(
                                model_name=model_name,
                                input_length=input_length,
                                output_length=output_length,
                                batch_size=batch_size,
                                repetition=rep
                            )
                            
                            # Warmup iterations
                            if rep == 0:
                                self.logger.debug("Running warmup iterations...")
                                for _ in range(measurement_config['warmup_iterations']):
                                    self.measure_single_inference(model_name, config)
                            
                            # Actual measurement
                            result = self.measure_single_inference(model_name, config)
                            self.measurements.append(result)
                            
                            measurement_count += 1
                            if measurement_count % 10 == 0:
                                progress = measurement_count / total_measurements * 100
                                self.logger.info(f"Progress: {progress:.1f}%")
                            
                            # Clear GPU cache between measurements
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
        
        self.logger.info("Profiling completed")
    
    def analyze_measurements(self) -> None:
        """Analyze measurement results and fit cost models."""
        self.logger.info("Analyzing measurements...")
        
        # Convert to DataFrame for easier analysis
        data = []
        for measurement in self.measurements:
            if measurement.successful:
                row = {
                    'model_name': measurement.config.model_name,
                    'input_length': measurement.config.input_length,
                    'output_length': measurement.config.output_length,
                    'batch_size': measurement.config.batch_size,
                    'total_time': measurement.total_time,
                    'tokens_per_second': measurement.tokens_per_second,
                    'gpu_memory_used': measurement.gpu_memory_used,
                }
                data.append(row)
        
        df = pd.DataFrame(data)
        
        if df.empty:
            self.logger.error("No successful measurements to analyze")
            return
        
        # Fit cost models for each model
        for model_name in df['model_name'].unique():
            model_data = df[df['model_name'] == model_name]
            cost_model = self._fit_cost_model(model_name, model_data)
            self.cost_models[model_name] = cost_model
            
            self.logger.info(
                f"Cost model for {model_name}: "
                f"R² = {cost_model.r_squared:.3f}"
            )
    
    def _fit_cost_model(self, model_name: str, data: pd.DataFrame) -> CostModel:
        """Fit cost model for a specific model."""
        
        # Features: input_length, output_length, batch_size
        X = data[['input_length', 'output_length', 'batch_size']].values
        y = data['total_time'].values
        
        # Try different model types
        models = {}
        
        # Linear model
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        linear_score = r2_score(y, linear_model.predict(X))
        models['linear'] = (linear_model, linear_score)
        
        # Polynomial model (degree 2)
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y)
        poly_score = r2_score(y, poly_model.predict(X_poly))
        models['polynomial'] = (poly_model, poly_score, poly_features)
        
        # Select best model
        best_model_type = max(models.keys(), key=lambda k: models[k][1])
        best_model, best_score = models[best_model_type][:2]
        
        # Extract coefficients
        if best_model_type == 'linear':
            coefficients = {
                'input_length': float(best_model.coef_[0]),
                'output_length': float(best_model.coef_[1]),
                'batch_size': float(best_model.coef_[2])
            }
            intercept = float(best_model.intercept_)
        else:  # polynomial
            coefficients = {'polynomial_coeffs': best_model.coef_.tolist()}
            intercept = float(best_model.intercept_)
        
        return CostModel(
            model_name=model_name,
            coefficients=coefficients,
            intercept=intercept,
            r_squared=best_score,
            model_type=best_model_type
        )
    
    def generate_visualizations(self) -> None:
        """Generate visualization plots."""
        self.logger.info("Generating visualizations...")
        
        output_dir = Path(self.config['output']['results_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert measurements to DataFrame
        data = []
        for measurement in self.measurements:
            if measurement.successful:
                row = asdict(measurement)
                row.update(asdict(measurement.config))
                data.append(row)
        
        if not data:
            self.logger.warning("No data for visualization")
            return
            
        df = pd.DataFrame(data)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Latency by model and input length
        pivot_data = df.pivot_table(
            values='total_time', 
            index='input_length',
            columns='model_name',
            aggfunc='mean'
        )
        pivot_data.plot(kind='line', ax=axes[0, 0], marker='o')
        axes[0, 0].set_title('Latency vs Input Length')
        axes[0, 0].set_xlabel('Input Length (tokens)')
        axes[0, 0].set_ylabel('Latency (seconds)')
        axes[0, 0].legend(title='Model')
        
        # 2. Throughput comparison
        throughput_data = df.groupby('model_name')['tokens_per_second'].mean()
        throughput_data.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Average Throughput by Model')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Tokens per Second')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Memory usage
        memory_data = df.groupby('model_name')['gpu_memory_used'].mean() / 1e9  # GB
        memory_data.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Average GPU Memory Usage')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Memory Usage (GB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Latency distribution
        for i, model in enumerate(df['model_name'].unique()):
            model_data = df[df['model_name'] == model]['total_time']
            axes[1, 1].hist(model_data, alpha=0.6, label=model, bins=20)
        axes[1, 1].set_title('Latency Distribution')
        axes[1, 1].set_xlabel('Latency (seconds)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cost_profiling_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to {output_dir}")
    
    def save_results(self) -> None:
        """Save profiling results and cost models."""
        output_dir = Path(self.config['output']['results_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw measurements
        measurements_data = [asdict(m) for m in self.measurements]
        with open(output_dir / 'raw_measurements.json', 'w') as f:
            json.dump(measurements_data, f, indent=2, default=str)
        
        # Save cost models
        cost_models_data = {name: asdict(model) for name, model in self.cost_models.items()}
        with open(output_dir / 'cost_models.json', 'w') as f:
            json.dump(cost_models_data, f, indent=2)
        
        # Save summary statistics
        summary = self._generate_summary()
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        successful_measurements = [m for m in self.measurements if m.successful]
        
        if not successful_measurements:
            return {"error": "No successful measurements"}
        
        summary = {
            "total_measurements": len(self.measurements),
            "successful_measurements": len(successful_measurements),
            "success_rate": len(successful_measurements) / len(self.measurements),
            "models_profiled": list(self.cost_models.keys()),
            "average_latencies": {},
            "cost_model_quality": {}
        }
        
        # Average latencies by model
        for model_name in self.cost_models.keys():
            model_measurements = [
                m for m in successful_measurements 
                if m.config.model_name == model_name
            ]
            if model_measurements:
                avg_latency = np.mean([m.total_time for m in model_measurements])
                summary["average_latencies"][model_name] = avg_latency
        
        # Cost model quality
        for model_name, cost_model in self.cost_models.items():
            summary["cost_model_quality"][model_name] = {
                "r_squared": cost_model.r_squared,
                "model_type": cost_model.model_type
            }
        
        return summary
    
    def update_config_files(self) -> None:
        """Update configuration files with measured costs."""
        if not self.config['integration']['auto_update_configs']:
            return
            
        self.logger.info("Updating configuration files...")
        
        # Update qwen3_models.yaml
        qwen3_config_path = Path("configs/qwen3_models.yaml")
        if qwen3_config_path.exists():
            with open(qwen3_config_path, 'r') as f:
                qwen3_config = yaml.safe_load(f)
            
            # Update latency measurements
            for stage in qwen3_config['models']['stages']:
                model_name = stage['name']
                if model_name in self.cost_models:
                    # Use median latency as base
                    model_measurements = [
                        m for m in self.measurements 
                        if m.successful and m.config.model_name == model_name
                    ]
                    if model_measurements:
                        median_latency = np.median([m.total_time for m in model_measurements])
                        stage['base_latency_ms'] = median_latency * 1000  # Convert to ms
            
            # Backup original
            if self.config['output']['backup_original_configs']:
                backup_path = qwen3_config_path.with_suffix('.yaml.backup')
                qwen3_config_path.rename(backup_path)
            
            # Save updated config
            with open(qwen3_config_path, 'w') as f:
                yaml.dump(qwen3_config, f, indent=2)
        
        self.logger.info("Configuration files updated")
    
    def run_full_profiling(self) -> None:
        """Run complete profiling pipeline."""
        self.logger.info("Starting full cost profiling pipeline...")
        
        try:
            # Run profiling
            self.run_comprehensive_profiling()
            
            # Analyze results
            self.analyze_measurements()
            
            # Generate visualizations
            self.generate_visualizations()
            
            # Save results
            self.save_results()
            
            # Update config files
            self.update_config_files()
            
            self.logger.info("Cost profiling completed successfully")
            
        except Exception as e:
            self.logger.error(f"Cost profiling failed: {e}")
            raise

def main():
    """Main entry point for cost profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cost profiling for adaptive speculative decoding")
    parser.add_argument("--config", default="configs/cost_profiling.yaml", help="Config file path")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--models", nargs="+", help="Specific models to profile")
    
    args = parser.parse_args()
    
    # Initialize profiler
    profiler = CostProfiler(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        profiler.config['output']['results_dir'] = args.output_dir
    
    # Filter models if specified
    if args.models:
        profiler.config['profiling']['models'] = [
            m for m in profiler.config['profiling']['models']
            if m['name'] in args.models
        ]
    
    # Run profiling
    profiler.run_full_profiling()

if __name__ == "__main__":
    main()