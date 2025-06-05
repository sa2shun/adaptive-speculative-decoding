"""
FastAPI server for adaptive speculative decoding
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.stage import StageManager, StageConfig
from src.models.predictor import QualityPredictor, FeatureExtractor
from src.serving.pipeline import AdaptiveSpeculativePipeline, PipelineConfig
from src.serving.cache_manager import KVCacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
pipeline: Optional[AdaptiveSpeculativePipeline] = None
cache_manager: Optional[KVCacheManager] = None


class GenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(512, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")
    stream: bool = Field(False, description="Whether to stream the response")
    request_id: Optional[str] = Field(None, description="Optional request ID")


class GenerationResponse(BaseModel):
    """Response model for text generation"""
    request_id: str = Field(..., description="Request identifier")
    output: str = Field(..., description="Generated text")
    stopped_at_stage: int = Field(..., description="Stage where generation stopped")
    latency_ms: float = Field(..., description="Total latency in milliseconds")
    stage_probabilities: List[float] = Field(..., description="Acceptance probabilities")
    stage_costs: List[float] = Field(..., description="Costs for executed stages")
    total_tokens: int = Field(..., description="Total tokens generated")
    tokens_per_second: float = Field(..., description="Generation speed")
    cache_hits: int = Field(..., description="Number of cache hits")


class BatchGenerationRequest(BaseModel):
    """Request model for batch generation"""
    prompts: List[str] = Field(..., description="List of input prompts")
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class LambdaUpdateRequest(BaseModel):
    """Request model for updating lambda parameter"""
    lambda_value: float = Field(..., ge=0.01, le=100.0, description="New lambda value")


class StatsResponse(BaseModel):
    """Response model for pipeline statistics"""
    total_requests: int
    stage_distribution: List[float]
    avg_latency: float
    avg_tokens_per_second: float
    avg_tokens_per_request: float
    active_requests: int
    error_count: int
    cache_stats: Optional[Dict[str, Any]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Adaptive Speculative Decoding Server...")
    
    global pipeline, cache_manager
    
    try:
        # Load configuration
        config_path = os.getenv("CONFIG_PATH", "configs/serving.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize cache manager
        cache_config = config.get("pipeline", {}).get("cache", {})
        if cache_config.get("enable_kv_cache", True):
            cache_manager = KVCacheManager(
                max_cache_size_gb=cache_config.get("max_cache_size_gb", 40),
                cleanup_interval=cache_config.get("cache_cleanup_interval", 300)
            )
        
        # Initialize models
        await initialize_models(config)
        
        # Warmup
        if pipeline:
            pipeline.warmup()
        
        logger.info("Server startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down server...")
        if pipeline:
            pipeline.shutdown()
        logger.info("Server shutdown completed")


async def initialize_models(config: Dict[str, Any]):
    """Initialize all models and pipeline"""
    global pipeline
    
    # Load model configurations
    models_config_path = "configs/models.yaml"
    with open(models_config_path, 'r') as f:
        models_config = yaml.safe_load(f)
    
    # H100 GPU allocation (8 GPUs total)
    gpu_allocation = {
        "8b": [0],           # 1 GPU
        "13b": [1, 2],       # 2 GPUs  
        "34b": [3, 4],       # 2 GPUs
        "70b": [5, 6, 7]     # 3 GPUs (could use 4 if needed)
    }
    
    # Create stage configurations
    stage_configs = []
    for size, model_config in models_config["models"].items():
        stage_config = StageConfig(
            model_name=model_config["name"],
            model_size=size,
            tensor_parallel_size=model_config["tensor_parallel_size"],
            gpu_memory_utilization=model_config["gpu_memory_utilization"],
            quantized=model_config["quantization"]["enabled"],
            cost_per_token=model_config["cost_per_token"]
        )
        stage_configs.append(stage_config)
    
    # Initialize stage manager
    logger.info("Initializing stage manager...")
    stage_manager = StageManager(stage_configs, gpu_allocation)
    stage_manager.warmup_all()
    
    # Load quality predictor
    logger.info("Loading quality predictor...")
    predictor = QualityPredictor(feature_dim=256)
    
    # Try to load trained weights
    predictor_path = "checkpoints/predictor.pt"
    if os.path.exists(predictor_path):
        predictor.load_state_dict(torch.load(predictor_path, map_location="cpu"))
        logger.info(f"Loaded predictor from {predictor_path}")
    else:
        logger.warning(f"Predictor weights not found at {predictor_path}, using random weights")
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Create pipeline configuration
    pipeline_config_dict = config.get("pipeline", {})
    pipeline_config = PipelineConfig(
        lambda_value=pipeline_config_dict.get("lambda_value", 1.0),
        risk_adjustment=pipeline_config_dict.get("risk_adjustment", {}).get("enabled", True),
        risk_alpha=pipeline_config_dict.get("risk_adjustment", {}).get("alpha", 1.0),
        risk_beta=pipeline_config_dict.get("risk_adjustment", {}).get("beta", 1.0),
        enable_caching=pipeline_config_dict.get("cache", {}).get("enable_kv_cache", True),
        max_concurrent_requests=config.get("safety", {}).get("max_concurrent_requests", 100)
    )
    
    # Initialize pipeline
    logger.info("Initializing adaptive pipeline...")
    pipeline = AdaptiveSpeculativePipeline(
        stage_manager=stage_manager,
        predictor=predictor,
        feature_extractor=feature_extractor,
        config=pipeline_config,
        cache_manager=cache_manager
    )
    
    logger.info("All models initialized successfully")


# Create FastAPI app
app = FastAPI(
    title="Adaptive Speculative Decoding API",
    description="Multi-stage Draft-Verify pipeline with input-dependent depth optimization",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text using adaptive speculative decoding"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Process request
        result = await pipeline.process_request_async(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            request_id=request.request_id
        )
        
        return GenerationResponse(
            request_id=result.request_id,
            output=result.output,
            stopped_at_stage=result.stopped_at_stage,
            latency_ms=result.latency_ms,
            stage_probabilities=result.stage_probabilities,
            stage_costs=result.stage_costs,
            total_tokens=result.total_tokens,
            tokens_per_second=result.tokens_per_second,
            cache_hits=result.cache_hits
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_generate")
async def batch_generate(request: BatchGenerationRequest):
    """Generate text for multiple prompts"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Process batch
        results = pipeline.batch_process(
            prompts=request.prompts,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Convert to response format
        responses = []
        for result in results:
            responses.append(GenerationResponse(
                request_id=result.request_id,
                output=result.output,
                stopped_at_stage=result.stopped_at_stage,
                latency_ms=result.latency_ms,
                stage_probabilities=result.stage_probabilities,
                stage_costs=result.stage_costs,
                total_tokens=result.total_tokens,
                tokens_per_second=result.tokens_per_second,
                cache_hits=result.cache_hits
            ))
        
        return {"results": responses}
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get pipeline statistics"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    stats = pipeline.get_stats()
    
    return StatsResponse(
        total_requests=stats["total_requests"],
        stage_distribution=stats["stage_distribution"],
        avg_latency=stats["avg_latency"],
        avg_tokens_per_second=stats["avg_tokens_per_second"],
        avg_tokens_per_request=stats["avg_tokens_per_request"],
        active_requests=stats["active_requests"],
        error_count=stats["error_count"],
        cache_stats=stats.get("cache_stats")
    )


@app.post("/update_lambda")
async def update_lambda(request: LambdaUpdateRequest):
    """Update lambda parameter for quality-speed tradeoff"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        old_lambda = pipeline.config.lambda_value
        pipeline.update_lambda(request.lambda_value)
        
        return {
            "message": "Lambda updated successfully",
            "old_lambda": old_lambda,
            "new_lambda": request.lambda_value
        }
        
    except Exception as e:
        logger.error(f"Lambda update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset_stats")
async def reset_stats():
    """Reset pipeline statistics"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    pipeline.reset_stats()
    return {"message": "Statistics reset successfully"}


@app.get("/models")
async def get_model_info():
    """Get information about loaded models"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    model_info = {}
    for size in ["8b", "13b", "34b", "70b"]:
        try:
            stage = pipeline.stage_manager.get_stage(size)
            model_info[size] = stage.get_model_info()
        except Exception as e:
            model_info[size] = {"error": str(e)}
    
    return {"models": model_info}


@app.get("/cache_stats")
async def get_cache_stats():
    """Get detailed cache statistics"""
    if cache_manager is None:
        raise HTTPException(status_code=404, detail="Cache manager not available")
    
    return cache_manager.get_stats()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Speculative Decoding Server")
    parser.add_argument("--config", default="configs/serving.yaml", help="Config file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    
    args = parser.parse_args()
    
    # Set config path environment variable
    os.environ["CONFIG_PATH"] = args.config
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Run server
    uvicorn.run(
        "src.serving.server:app",
        host=args.host,
        port=args.port,
        workers=1,  # Single worker for GPU sharing
        log_level=args.log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()