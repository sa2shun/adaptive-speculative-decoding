#!/usr/bin/env python3
"""
Download Qwen3 model family for experiments.

This script downloads all required Qwen3 models to the raid storage.
Models will be saved in HuggingFace format for easy loading.
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Qwen3 model configurations
QWEN3_MODELS = {
    "7b": {
        "repo_id": "Qwen/Qwen2.5-7B",  # Using Qwen2.5 as Qwen3 proxy
        "revision": "main",
        "size_gb": 15
    },
    "14b": {
        "repo_id": "Qwen/Qwen2.5-14B",
        "revision": "main", 
        "size_gb": 28
    },
    "32b": {
        "repo_id": "Qwen/Qwen2.5-32B",
        "revision": "main",
        "size_gb": 65
    },
    "72b": {
        "repo_id": "Qwen/Qwen2.5-72B",
        "revision": "main",
        "size_gb": 145
    }
}


def check_disk_space(required_gb: int, path: Path) -> bool:
    """Check if enough disk space is available."""
    import shutil
    stat = shutil.disk_usage(path)
    available_gb = stat.free / (1024**3)
    
    logger.info(f"Available space: {available_gb:.1f} GB")
    logger.info(f"Required space: {required_gb} GB")
    
    return available_gb > required_gb * 1.2  # 20% buffer


def download_model(model_size: str, base_path: Path, token: str = None) -> bool:
    """Download a single Qwen model."""
    config = QWEN3_MODELS[model_size]
    model_path = base_path / f"qwen3-{model_size}"
    
    if model_path.exists() and any(model_path.iterdir()):
        logger.info(f"Model qwen3-{model_size} already exists at {model_path}")
        return True
    
    logger.info(f"Downloading {config['repo_id']} to {model_path}")
    
    try:
        # Create directory
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Download model
        snapshot_download(
            repo_id=config['repo_id'],
            revision=config['revision'],
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
            token=token,
            resume_download=True,
            max_workers=4
        )
        
        logger.info(f"Successfully downloaded qwen3-{model_size}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download qwen3-{model_size}: {e}")
        return False


def download_all_models(base_path: Path, token: str = None, models: List[str] = None) -> Dict[str, bool]:
    """Download all required Qwen3 models."""
    if models is None:
        models = list(QWEN3_MODELS.keys())
    
    # Check total required space
    total_required_gb = sum(QWEN3_MODELS[m]["size_gb"] for m in models)
    
    if not check_disk_space(total_required_gb, base_path):
        logger.error(f"Insufficient disk space. Need {total_required_gb} GB")
        return {m: False for m in models}
    
    # Download each model
    results = {}
    for model_size in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading Qwen3-{model_size.upper()}")
        logger.info(f"{'='*60}")
        
        success = download_model(model_size, base_path, token)
        results[model_size] = success
        
        if not success:
            logger.warning(f"Failed to download qwen3-{model_size}, continuing with others...")
    
    return results


def verify_downloads(base_path: Path) -> Dict[str, bool]:
    """Verify that all models are downloaded correctly."""
    results = {}
    
    for model_size in QWEN3_MODELS:
        model_path = base_path / f"qwen3-{model_size}"
        
        # Check if key files exist
        required_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json"
        ]
        
        exists = model_path.exists()
        has_files = all((model_path / f).exists() for f in required_files)
        has_weights = any(f.name.startswith("model") and f.suffix in [".safetensors", ".bin"] 
                         for f in model_path.glob("*") if f.is_file())
        
        is_valid = exists and has_files and has_weights
        results[model_size] = is_valid
        
        status = "✓" if is_valid else "✗"
        logger.info(f"qwen3-{model_size}: {status}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3 models")
    parser.add_argument("--base-path", type=str, 
                       default=f"/raid/{os.environ.get('USER', 'sasaki')}/models",
                       help="Base path for model storage")
    parser.add_argument("--models", nargs="+", 
                       choices=list(QWEN3_MODELS.keys()),
                       help="Specific models to download (default: all)")
    parser.add_argument("--token", type=str, 
                       help="HuggingFace token for gated models")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing downloads")
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Model storage path: {base_path}")
    
    if args.verify_only:
        logger.info("\nVerifying existing downloads...")
        results = verify_downloads(base_path)
        
        all_valid = all(results.values())
        if all_valid:
            logger.info("\n✓ All models are properly downloaded!")
        else:
            logger.error("\n✗ Some models are missing or incomplete!")
            
    else:
        logger.info("\nStarting model downloads...")
        results = download_all_models(base_path, args.token, args.models)
        
        # Verify after download
        logger.info("\nVerifying downloads...")
        verify_results = verify_downloads(base_path)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("="*60)
        
        for model_size in QWEN3_MODELS:
            if model_size in results:
                status = "✓ Downloaded" if verify_results.get(model_size, False) else "✗ Failed"
                logger.info(f"qwen3-{model_size}: {status}")
    
    # Save model paths configuration
    config_path = base_path / "model_paths.json"
    import json
    
    model_paths = {
        f"qwen3_{size}": str(base_path / f"qwen3-{size}")
        for size in QWEN3_MODELS
    }
    
    with open(config_path, 'w') as f:
        json.dump(model_paths, f, indent=2)
    
    logger.info(f"\nModel paths saved to: {config_path}")


if __name__ == "__main__":
    main()