"""
Demo pipeline for adaptive speculative decoding with mock components
"""

import time
import uuid
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..models.predictor import MockQualityPredictor
from ..algorithms.dp_solver import optimal_stopping_rule

logger = logging.getLogger(__name__)


@dataclass
class DemoRequestResult:
    """Result of processing a demo request"""
    request_id: str
    output: str
    stopped_at_stage: int
    latency_ms: float
    stage_probabilities: List[float]
    stage_costs: List[float]
    total_tokens: int
    prompt: str


class MockStage:
    """Mock stage for demonstration"""
    
    def __init__(self, stage_id: str, cost_per_token: float):
        self.stage_id = stage_id
        self.cost_per_token = cost_per_token
        
        # Mock model responses based on prompt complexity
        self.responses = {
            "simple": "The capital of France is Paris.",
            "medium": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This occurs in the chloroplasts using chlorophyll.",
            "complex": """def quicksort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort(arr, low, pivot_index - 1)
        quicksort(arr, pivot_index + 1, high)
    
    return arr

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1"""
        }
    
    def generate(self, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        """Mock generation"""
        # Simulate processing time based on stage size
        stage_latencies = {"8b": 50, "13b": 100, "34b": 200, "70b": 400}
        latency = stage_latencies.get(self.stage_id, 100)
        
        # Add some random variation
        latency += np.random.normal(0, latency * 0.1)
        time.sleep(latency / 1000.0)  # Convert to seconds
        
        # Determine response complexity based on prompt
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ["capital", "what is", "simple"]):
            response_type = "simple"
        elif any(word in prompt_lower for word in ["explain", "process", "how"]):
            response_type = "medium"  
        else:
            response_type = "complex"
        
        output = self.responses[response_type]
        output_tokens = len(output.split())
        
        return {
            "text": output,
            "prompt_tokens": len(prompt.split()),
            "output_tokens": output_tokens,
            "latency_ms": latency,
            "cost": output_tokens * self.cost_per_token,
            "stage_id": self.stage_id
        }


class DemoPipeline:
    """Demo pipeline using mock components"""
    
    def __init__(self, lambda_value: float = 1.0):
        self.lambda_value = lambda_value
        
        # Initialize mock stages
        self.stages = {
            "8b": MockStage("8b", 1.0),
            "13b": MockStage("13b", 1.6), 
            "34b": MockStage("34b", 4.2),
            "70b": MockStage("70b", 8.8)
        }
        
        self.stage_order = ["8b", "13b", "34b", "70b"]
        
        # Initialize mock predictor
        self.predictor = MockQualityPredictor()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "stage_stops": [0] * 4,
            "avg_latency": 0.0,
            "stage_distribution": {}
        }
        
        logger.info("Demo pipeline initialized")
    
    def process_request(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> DemoRequestResult:
        """Process a single request through the demo pipeline"""
        
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Processing request {request_id}: {prompt[:50]}...")
        
        # Get quality predictions for all stages
        stage_probabilities = self.predictor.predict_all_stages(prompt)
        
        # Use dynamic programming to find optimal stopping point
        stage_costs = [stage.cost_per_token for stage in self.stages.values()]
        
        optimal_stage, _ = optimal_stopping_rule(
            p=stage_probabilities,
            C=stage_costs,
            lam=self.lambda_value
        )
        
        # Execute generation at the optimal stage
        stage_id = self.stage_order[optimal_stage]
        stage = self.stages[stage_id]
        
        result = stage.generate(prompt, max_tokens)
        
        # Calculate total latency
        total_latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Create result
        demo_result = DemoRequestResult(
            request_id=request_id,
            output=result["text"],
            stopped_at_stage=optimal_stage,
            latency_ms=total_latency,
            stage_probabilities=stage_probabilities,
            stage_costs=stage_costs,
            total_tokens=result["output_tokens"],
            prompt=prompt
        )
        
        # Update statistics
        self._update_stats(demo_result)
        
        logger.info(f"Request {request_id} completed in {total_latency:.2f}ms at stage {stage_id}")
        
        return demo_result
    
    def _update_stats(self, result: DemoRequestResult):
        """Update pipeline statistics"""
        self.stats["total_requests"] += 1
        self.stats["stage_stops"][result.stopped_at_stage] += 1
        
        # Update average latency
        old_avg = self.stats["avg_latency"]
        n = self.stats["total_requests"]
        self.stats["avg_latency"] = (old_avg * (n - 1) + result.latency_ms) / n
        
        # Update stage distribution
        stage_name = self.stage_order[result.stopped_at_stage]
        self.stats["stage_distribution"][stage_name] = self.stats["stage_distribution"].get(stage_name, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.stats.copy()
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of demo pipeline"""
        return {stage_id: True for stage_id in self.stage_order}


def create_demo_pipeline(lambda_value: float = 1.0) -> DemoPipeline:
    """Create a demo pipeline instance"""
    return DemoPipeline(lambda_value=lambda_value)