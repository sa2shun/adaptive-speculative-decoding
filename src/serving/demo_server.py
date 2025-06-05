"""
Demo server for adaptive speculative decoding
"""

import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:
    from .demo_pipeline import create_demo_pipeline, DemoPipeline
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, parent_dir)
    from src.serving.demo_pipeline import create_demo_pipeline, DemoPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[DemoPipeline] = None


class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    """Response model for text generation"""
    output: str
    stopped_at_stage: int
    latency_ms: float
    stage_probabilities: List[float]
    stage_costs: List[float]
    total_tokens: int
    request_id: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    stages: Dict[str, bool]


class StatsResponse(BaseModel):
    """Statistics response"""
    total_requests: int
    avg_latency: float
    stage_stops: List[int]
    stage_distribution: Dict[str, int]


# Create FastAPI app
app = FastAPI(
    title="Adaptive Speculative Decoding Demo",
    description="Demo API for adaptive speculative decoding pipeline",
    version="0.1.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup"""
    global pipeline
    
    logger.info("Starting demo server...")
    
    try:
        # Create demo pipeline with default lambda
        pipeline = create_demo_pipeline(lambda_value=1.0)
        logger.info("Demo pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        stage_health = pipeline.health_check()
        
        return HealthResponse(
            status="healthy" if all(stage_health.values()) else "degraded",
            stages=stage_health
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using adaptive speculative decoding"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        logger.info(f"Generating text for prompt: {request.prompt[:50]}...")
        
        # Process the request
        result = pipeline.process_request(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return GenerateResponse(
            output=result.output,
            stopped_at_stage=result.stopped_at_stage,
            latency_ms=result.latency_ms,
            stage_probabilities=result.stage_probabilities,
            stage_costs=result.stage_costs,
            total_tokens=result.total_tokens,
            request_id=result.request_id
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Get pipeline statistics"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        stats = pipeline.get_stats()
        
        return StatsResponse(
            total_requests=stats["total_requests"],
            avg_latency=stats["avg_latency"],
            stage_stops=stats["stage_stops"],
            stage_distribution=stats["stage_distribution"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Adaptive Speculative Decoding Demo API",
        "version": "0.1.0",
        "endpoints": [
            "/health",
            "/generate", 
            "/stats"
        ]
    }


def main():
    """Main entry point for the demo server"""
    parser = argparse.ArgumentParser(description="Demo server for adaptive speculative decoding")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    logger.info(f"Starting demo server on {args.host}:{args.port}")
    
    # Start the server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()