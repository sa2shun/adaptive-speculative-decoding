"""
Adaptive speculative decoding pipeline
"""

from typing import List, Dict, Any, Optional, Tuple
import time
import uuid
import numpy as np
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..models.stage import Stage, StageManager
from ..models.predictor import QualityPredictor, FeatureExtractor
from ..algorithms.dp_solver import optimal_stopping_rule, bayesian_adjustment
from .cache_manager import KVCacheManager
from ..config.cost_config import get_measured_cost

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the adaptive pipeline"""
    lambda_value: float = 1.0
    risk_adjustment: bool = True
    risk_alpha: float = 1.0
    risk_beta: float = 1.0
    enable_caching: bool = True
    max_concurrent_requests: int = 100
    batch_timeout_ms: float = 50.0


@dataclass
class RequestResult:
    """Result of processing a single request"""
    request_id: str
    output: str
    stopped_at_stage: int
    latency_ms: float
    stage_probabilities: List[float]
    stage_costs: List[float]
    cache_hits: int
    total_tokens: int
    tokens_per_second: float


class AdaptiveSpeculativePipeline:
    """
    Main pipeline for adaptive speculative decoding
    """
    
    def __init__(
        self,
        stage_manager: StageManager,
        predictor: QualityPredictor,
        feature_extractor: FeatureExtractor,
        config: PipelineConfig,
        cache_manager: Optional[KVCacheManager] = None
    ):
        self.stage_manager = stage_manager
        self.predictor = predictor
        self.feature_extractor = feature_extractor
        self.config = config
        
        # Initialize cache manager
        if cache_manager is None and config.enable_caching:
            cache_manager = KVCacheManager()
        self.cache_manager = cache_manager
        
        # Pipeline statistics
        self.stats = {
            "total_requests": 0,
            "stage_stops": [0] * 4,
            "avg_latency": 0.0,
            "avg_tokens_per_second": 0.0,
            "total_tokens": 0,
            "avg_stage_probabilities": [0.0] * 4,
            "error_count": 0
        }
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
        
        # Request tracking
        self.active_requests = {}
        
        logger.info("AdaptiveSpeculativePipeline initialized")
    
    def process_request(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        request_id: Optional[str] = None
    ) -> RequestResult:
        """
        Process a single request synchronously
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            request_id: Optional request ID
            
        Returns:
            RequestResult with generation details
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        try:
            # Track active request
            self.active_requests[request_id] = {
                "start_time": start_time,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt
            }
            
            result = self._process_stages(
                request_id=request_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Update statistics
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            self.stats["error_count"] += 1
            raise
        finally:
            # Cleanup
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            if self.cache_manager:
                self.cache_manager.cleanup_request(request_id)
    
    async def process_request_async(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        request_id: Optional[str] = None
    ) -> RequestResult:
        """
        Process a request asynchronously
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.process_request,
            prompt,
            max_tokens,
            temperature,
            request_id
        )
        return result
    
    def _process_stages(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> RequestResult:
        """
        Process through stages with dynamic stopping
        """
        stage_names = ["8b", "13b", "34b", "70b"]
        current_prompt = prompt
        
        probabilities = []
        costs = []
        outputs = []
        cache_hits = 0
        total_tokens = 0
        
        for stage_idx, stage_name in enumerate(stage_names):
            stage = self.stage_manager.get_stage(stage_name)
            stage_start_time = time.time()
            
            # Check cache first
            cached_output = None
            if self.cache_manager:
                cached_data = self.cache_manager.get_cache(request_id, stage_idx)
                if cached_data:
                    cached_output = cached_data.get("output")
                    cache_hits += 1
            
            if cached_output:
                # Use cached result
                stage_output = [cached_output]
                stage_logprobs = [np.array([])]
                stage_stats = {"generation_time_ms": 0}
                logger.debug(f"Using cached output for stage {stage_idx}")
            else:
                # Generate with stage
                stage_output, stage_logprobs, stage_stats = stage.generate(
                    prompts=[current_prompt],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    return_logprobs=True
                )
                
                # Cache the result
                if self.cache_manager:
                    cache_data = {
                        "output": stage_output[0],
                        "logprobs": stage_logprobs[0] if stage_logprobs else np.array([])
                    }
                    self.cache_manager.allocate(request_id, stage_idx, cache_data)
            
            outputs.append(stage_output[0])
            # Use measured cost instead of parameter-based cost
            measured_cost = get_measured_cost(stage_name)
            costs.append(measured_cost)
            total_tokens += len(stage_output[0].split())
            
            # Predict acceptance probability (except for last stage)
            if stage_idx < len(stage_names) - 1:
                prob = self.predictor.predict(
                    prompt=current_prompt,
                    draft_output=stage_output[0],
                    draft_logprobs=stage_logprobs[0] if stage_logprobs else None,
                    stage_id=stage_idx,
                    feature_extractor=self.feature_extractor
                )
                
                # Apply risk adjustment
                if self.config.risk_adjustment:
                    n_obs = max(100, self.stats["total_requests"])
                    prob = bayesian_adjustment(
                        prob, n_obs, self.config.risk_alpha, self.config.risk_beta
                    )
                
                probabilities.append(prob)
            else:
                probabilities.append(1.0)  # Last stage always accepts
            
            logger.debug(f"Stage {stage_idx}: prob={probabilities[-1]:.3f}, "
                        f"time={stage_stats['generation_time_ms']:.1f}ms")
            
            # Determine optimal stopping point
            current_probs = probabilities[:stage_idx + 1]
            current_costs = costs[:stage_idx + 1]
            
            k_star, _ = optimal_stopping_rule(
                p=current_probs,
                C=current_costs,
                lam=self.config.lambda_value,
                risk_adjustment=False  # Already applied above
            )
            
            # Check if we should stop at current stage
            if k_star == stage_idx:
                logger.debug(f"Stopping at stage {stage_idx}")
                break
            
            # Prepare for next stage
            if stage_idx < len(stage_names) - 1:
                # Use current output as context for next stage
                current_prompt = prompt + " " + stage_output[0]
        
        # Calculate final metrics
        total_time_ms = (time.time() - start_time) * 1000
        tokens_per_second = total_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0
        
        # Cleanup cache for unused stages
        if self.cache_manager:
            self.cache_manager.truncate_at_stage(request_id, k_star)
        
        return RequestResult(
            request_id=request_id,
            output=outputs[k_star],
            stopped_at_stage=k_star,
            latency_ms=total_time_ms,
            stage_probabilities=probabilities,
            stage_costs=costs[:k_star + 1],
            cache_hits=cache_hits,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second
        )
    
    def _update_stats(self, result: RequestResult):
        """Update pipeline statistics"""
        self.stats["total_requests"] += 1
        self.stats["stage_stops"][result.stopped_at_stage] += 1
        self.stats["total_tokens"] += result.total_tokens
        
        # Update running averages
        alpha = 0.01  # Smoothing factor
        self.stats["avg_latency"] = (
            (1 - alpha) * self.stats["avg_latency"] + 
            alpha * result.latency_ms
        )
        
        self.stats["avg_tokens_per_second"] = (
            (1 - alpha) * self.stats["avg_tokens_per_second"] + 
            alpha * result.tokens_per_second
        )
        
        # Update stage probabilities
        for i, prob in enumerate(result.stage_probabilities):
            if i < len(self.stats["avg_stage_probabilities"]):
                self.stats["avg_stage_probabilities"][i] = (
                    (1 - alpha) * self.stats["avg_stage_probabilities"][i] + 
                    alpha * prob
                )
    
    def batch_process(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> List[RequestResult]:
        """
        Process multiple requests in batch
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per request
            temperature: Generation temperature
            
        Returns:
            List of RequestResults
        """
        # TODO: Implement intelligent batching based on predicted stages
        # For now, process sequentially
        results = []
        for prompt in prompts:
            result = self.process_request(prompt, max_tokens, temperature)
            results.append(result)
        
        return results
    
    def update_lambda(self, new_lambda: float):
        """Update lambda parameter for quality-speed tradeoff"""
        old_lambda = self.config.lambda_value
        self.config.lambda_value = new_lambda
        logger.info(f"Updated lambda: {old_lambda:.3f} -> {new_lambda:.3f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = self.stats.copy()
        
        # Add derived metrics
        if stats["total_requests"] > 0:
            stats["stage_distribution"] = [
                count / stats["total_requests"] 
                for count in stats["stage_stops"]
            ]
            stats["avg_tokens_per_request"] = (
                stats["total_tokens"] / stats["total_requests"]
            )
        else:
            stats["stage_distribution"] = [0.0] * 4
            stats["avg_tokens_per_request"] = 0.0
        
        # Add cache stats if available
        if self.cache_manager:
            stats["cache_stats"] = self.cache_manager.get_stats()
        
        # Add active request info
        stats["active_requests"] = len(self.active_requests)
        
        return stats
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            "total_requests": 0,
            "stage_stops": [0] * 4,
            "avg_latency": 0.0,
            "avg_tokens_per_second": 0.0,
            "total_tokens": 0,
            "avg_stage_probabilities": [0.0] * 4,
            "error_count": 0
        }
        logger.info("Pipeline statistics reset")
    
    def warmup(self, num_requests: int = 5):
        """Warmup the pipeline with dummy requests"""
        logger.info(f"Warming up pipeline with {num_requests} requests...")
        
        dummy_prompts = [
            "Hello, how are you today?",
            "What is the capital of France?",
            "Explain machine learning in simple terms.",
            "Write a short poem about nature.",
            "What are the benefits of renewable energy?"
        ]
        
        for i in range(num_requests):
            prompt = dummy_prompts[i % len(dummy_prompts)]
            try:
                result = self.process_request(
                    prompt=prompt,
                    max_tokens=50,
                    temperature=0.7
                )
                logger.debug(f"Warmup {i+1}: {result.latency_ms:.1f}ms, "
                           f"stopped at stage {result.stopped_at_stage}")
            except Exception as e:
                logger.warning(f"Warmup request {i+1} failed: {e}")
        
        logger.info("Pipeline warmup completed")
    
    def shutdown(self):
        """Shutdown the pipeline gracefully"""
        logger.info("Shutting down pipeline...")
        
        # Wait for active requests to complete
        if self.active_requests:
            logger.info(f"Waiting for {len(self.active_requests)} active requests...")
            # In production, implement proper graceful shutdown
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("Pipeline shutdown completed")