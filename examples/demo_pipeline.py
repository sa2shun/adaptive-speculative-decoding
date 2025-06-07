"""
Demo Pipeline for Adaptive Speculative Decoding
Educational demonstration with mock components - SEPARATE from research pipeline

⚠️ WARNING: This is for demonstration/education only
Research experiments MUST use src/serving/real_model_pipeline.py
"""

import time
import uuid
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

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
    demo_only: bool = True

class MockStage:
    """Mock stage for demonstration ONLY"""
    
    def __init__(self, stage_id: str, cost_per_token: float):
        self.stage_id = stage_id
        self.cost_per_token = cost_per_token
        
        # Mock model responses for educational purposes
        self.responses = {
            "simple": "The capital of France is Paris.",
            "medium": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
            "complex": """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)"""
        }
        
        logger.warning(f"🚨 MockStage {stage_id} initialized - DEMO ONLY, not for research")
    
    def generate(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Mock generation for demo purposes"""
        
        # Determine complexity for demo
        if len(prompt) < 50:
            complexity = "simple"
        elif len(prompt) < 100:
            complexity = "medium"
        else:
            complexity = "complex"
        
        # Mock response selection
        response = self.responses.get(complexity, "This is a demo response.")
        
        # Mock latency based on stage
        stage_num = int(self.stage_id.split('-')[0])
        mock_latency = 0.1 * (stage_num + 1) + np.random.normal(0, 0.02)
        
        return {
            "text": response,
            "tokens": len(response.split()),
            "latency": max(0.01, mock_latency),
            "cost": self.cost_per_token * len(response.split()),
            "quality_estimate": 0.7 + stage_num * 0.05,
            "demo_only": True
        }

class DemoPipeline:
    """Educational demo pipeline - NOT for research use"""
    
    def __init__(self):
        logger.warning("🚨 DemoPipeline initialized - EDUCATIONAL USE ONLY")
        logger.warning("   For research, use src.serving.real_model_pipeline.RealModelPipeline")
        
        # Mock stages for demonstration
        self.stages = {
            0: MockStage("0-demo-7b", 0.1),
            1: MockStage("1-demo-14b", 0.2),
            2: MockStage("2-demo-32b", 0.4),
            3: MockStage("3-demo-72b", 0.8)
        }
        
        self.stage_names = ["Demo-7B", "Demo-14B", "Demo-32B", "Demo-72B"]
        
    def process_request(self, prompt: str, lambda_param: float = 1.0) -> DemoRequestResult:
        """Process demo request with mock adaptive inference"""
        
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        logger.info(f"🎭 Processing demo request {request_id} (λ={lambda_param})")
        logger.warning("   This is DEMO ONLY - results are not research-grade")
        
        stage_results = []
        total_cost = 0
        selected_stage = -1
        final_output = ""
        
        # Mock adaptive inference through stages
        for stage_id in range(len(self.stages)):
            stage = self.stages[stage_id]
            
            # Mock generation
            result = stage.generate(prompt)
            stage_results.append(result)
            total_cost += result["cost"]
            
            # Mock stopping decision based on lambda
            quality_threshold = 0.75 if lambda_param < 1.0 else 0.85 if lambda_param <= 5.0 else 0.95
            
            should_stop = (
                result["quality_estimate"] > quality_threshold or 
                stage_id == len(self.stages) - 1
            )
            
            if should_stop:
                selected_stage = stage_id
                final_output = result["text"]
                break
        
        end_time = time.time()
        total_latency = (end_time - start_time) * 1000  # ms
        
        # Mock stage probabilities
        stage_probs = [0.25, 0.35, 0.25, 0.15]  # Demo distribution
        stage_costs = [r["cost"] for r in stage_results]
        
        return DemoRequestResult(
            request_id=request_id,
            output=final_output,
            stopped_at_stage=selected_stage,
            latency_ms=total_latency,
            stage_probabilities=stage_probs,
            stage_costs=stage_costs,
            total_tokens=sum(r["tokens"] for r in stage_results),
            prompt=prompt,
            demo_only=True
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get demo pipeline statistics"""
        return {
            "demo_warning": "This is for demonstration only - not research data",
            "stage_count": len(self.stages),
            "stage_names": self.stage_names,
            "use_for_research": "src.serving.real_model_pipeline.RealModelPipeline"
        }

def main():
    """Demo main function"""
    print("🎭 Adaptive Speculative Decoding - DEMO PIPELINE")
    print("⚠️  WARNING: This is for educational purposes only")
    print("   For research, use: src.serving.real_model_pipeline.RealModelPipeline")
    print()
    
    # Initialize demo pipeline
    pipeline = DemoPipeline()
    
    # Demo prompts
    demo_prompts = [
        "What is AI?",
        "Explain machine learning in simple terms for beginners.",
        "Write a Python function to calculate the fibonacci sequence efficiently with memoization and error handling."
    ]
    
    # Process demo requests
    for i, prompt in enumerate(demo_prompts):
        print(f"\n--- Demo Request {i+1} ---")
        print(f"Prompt: {prompt}")
        
        result = pipeline.process_request(prompt, lambda_param=1.0)
        
        print(f"Selected Stage: {result.stopped_at_stage} ({pipeline.stage_names[result.stopped_at_stage]})")
        print(f"Output: {result.output}")
        print(f"Latency: {result.latency_ms:.1f}ms")
        print(f"Demo Only: {result.demo_only}")
    
    print("\n🚨 REMINDER: This demo uses mock components")
    print("   Real research must use src.serving.real_model_pipeline.RealModelPipeline")

if __name__ == "__main__":
    main()