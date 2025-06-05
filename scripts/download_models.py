#!/usr/bin/env python3
"""
Download and prepare models for adaptive speculative decoding
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "8b": {
        "name": "meta-llama/Llama-3.1-8B",
        "size_gb": 16,
        "auth_required": True
    },
    "13b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct", 
        "size_gb": 16,
        "auth_required": True
    },
    "34b": {
        "name": "codellama/CodeLlama-34b-hf",
        "size_gb": 68,
        "auth_required": False
    },
    "70b": {
        "name": "meta-llama/Llama-3.1-70B-Instruct",
        "size_gb": 140,
        "auth_required": True
    }
}


def check_disk_space(required_gb: float, path: str = ".") -> bool:
    """Check if enough disk space is available"""
    import shutil
    
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)
    
    logger.info(f"Available disk space: {free_gb:.1f}GB, Required: {required_gb:.1f}GB")
    
    return free_gb >= required_gb * 1.2  # 20% buffer


def download_model(model_size: str, save_dir: str, force: bool = False):
    """Download and save a specific model"""
    
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}")
    
    config = MODEL_CONFIGS[model_size]
    model_name = config["name"]
    save_path = Path(save_dir) / model_size
    
    # Check if already exists
    if save_path.exists() and not force:
        logger.info(f"Model {model_size} already exists at {save_path}")
        return
    
    # Check disk space
    if not check_disk_space(config["size_gb"], save_dir):
        logger.error(f"Insufficient disk space for {model_size}")
        return
    
    logger.info(f"Downloading {model_size} model: {model_name}")
    
    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Download model
        logger.info("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        
        # Save to local directory
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving to {save_path}...")
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        # Clean up GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Successfully downloaded {model_size} model")
        
    except Exception as e:
        logger.error(f"Failed to download {model_size}: {e}")
        
        # Clean up partial download
        if save_path.exists():
            import shutil
            shutil.rmtree(save_path)
        
        raise


def verify_model(model_size: str, save_dir: str) -> bool:
    """Verify that a downloaded model works correctly"""
    
    save_path = Path(save_dir) / model_size
    
    if not save_path.exists():
        logger.error(f"Model {model_size} not found at {save_path}")
        return False
    
    try:
        logger.info(f"Verifying {model_size} model...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        
        # Load model (CPU only for verification)
        model = AutoModelForCausalLM.from_pretrained(
            save_path,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        
        # Test generation
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + 10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test generation: {generated_text}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        logger.info(f"Model {model_size} verification successful")
        return True
        
    except Exception as e:
        logger.error(f"Model {model_size} verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download models for adaptive speculative decoding")
    parser.add_argument(
        "--models",
        default="8b,13b,34b,70b",
        help="Comma-separated list of model sizes to download"
    )
    parser.add_argument(
        "--save-dir",
        default="./checkpoints",
        help="Directory to save models"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify models after downloading"
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token for authentication"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    
    args = parser.parse_args()
    
    # Parse model list
    model_sizes = [size.strip() for size in args.models.split(",")]
    
    # Validate model sizes
    for size in model_sizes:
        if size not in MODEL_CONFIGS:
            logger.error(f"Unknown model size: {size}")
            sys.exit(1)
    
    # Check authentication requirements
    auth_required = any(
        MODEL_CONFIGS[size]["auth_required"] for size in model_sizes
    )
    
    if auth_required:
        if args.hf_token:
            login(args.hf_token)
        else:
            logger.info("Some models require HuggingFace authentication.")
            logger.info("Please provide --hf-token or run 'huggingface-cli login'")
            
            # Try to use existing token
            try:
                from huggingface_hub import HfFolder
                token = HfFolder.get_token()
                if token:
                    logger.info("Using existing HuggingFace token")
                else:
                    logger.error("No HuggingFace token found")
                    sys.exit(1)
            except:
                logger.error("Please provide HuggingFace token")
                sys.exit(1)
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate total space required
    total_size_gb = sum(MODEL_CONFIGS[size]["size_gb"] for size in model_sizes)
    
    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info(f"Would download {len(model_sizes)} models:")
        for size in model_sizes:
            config = MODEL_CONFIGS[size]
            logger.info(f"  {size}: {config['name']} ({config['size_gb']}GB)")
        logger.info(f"Total size: {total_size_gb:.1f}GB")
        return
    
    logger.info(f"Downloading {len(model_sizes)} models to {args.save_dir}")
    logger.info(f"Total estimated size: {total_size_gb:.1f}GB")
    
    # Check total disk space
    if not check_disk_space(total_size_gb, args.save_dir):
        logger.error("Insufficient disk space for all models")
        sys.exit(1)
    
    # Download each model
    success_count = 0
    for i, size in enumerate(model_sizes):
        logger.info(f"\n=== Downloading model {i+1}/{len(model_sizes)}: {size} ===")
        
        try:
            download_model(size, args.save_dir, args.force)
            
            if args.verify:
                if verify_model(size, args.save_dir):
                    success_count += 1
                else:
                    logger.error(f"Model {size} failed verification")
            else:
                success_count += 1
                
        except Exception as e:
            logger.error(f"Failed to download {size}: {e}")
            continue
    
    # Summary
    logger.info(f"\n=== Download Summary ===")
    logger.info(f"Successfully downloaded: {success_count}/{len(model_sizes)} models")
    
    if success_count == len(model_sizes):
        logger.info("All models downloaded successfully!")
        
        # Create model info file
        info_file = Path(args.save_dir) / "model_info.txt"
        with open(info_file, "w") as f:
            f.write("Downloaded Models:\n")
            for size in model_sizes:
                config = MODEL_CONFIGS[size]
                f.write(f"{size}: {config['name']}\n")
            f.write(f"\nTotal size: {total_size_gb:.1f}GB\n")
            f.write(f"Download date: {__import__('datetime').datetime.now()}\n")
        
    else:
        logger.error("Some models failed to download")
        sys.exit(1)


if __name__ == "__main__":
    main()