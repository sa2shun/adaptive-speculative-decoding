#!/usr/bin/env python3
"""
Check Meta Llama model access permissions on HuggingFace
"""

import requests
from huggingface_hub import HfApi, whoami
from transformers import AutoTokenizer, AutoConfig
import sys
from typing import Dict, List, Tuple
import time

# Models we want to check for adaptive speculative decoding
TARGET_MODELS = {
    "8B Base": "meta-llama/Meta-Llama-3-8B",
    "8B Instruct": "meta-llama/Meta-Llama-3-8B-Instruct", 
    "70B Base": "meta-llama/Llama-3.1-70B",
    "70B Instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "8B Latest": "meta-llama/Llama-3.2-8B",
    "70B Latest": "meta-llama/Llama-3.3-70B-Instruct"
}

# Alternative models that don't require special permissions
ALTERNATIVE_MODELS = {
    "Mistral 7B": "mistralai/Mistral-7B-v0.1",
    "Falcon 7B": "tiiuae/falcon-7b",
    "Falcon 40B": "tiiuae/falcon-40b",
    "CodeLlama 34B": "codellama/CodeLlama-34b-hf"
}


def check_huggingface_login() -> Dict[str, str]:
    """Check HuggingFace login status and token info"""
    try:
        user_info = whoami()
        return {
            "status": "✅ Logged in",
            "username": user_info.get("name", "Unknown"),
            "user_id": user_info.get("id", "Unknown"),
            "token_type": user_info.get("auth", {}).get("type", "Unknown")
        }
    except Exception as e:
        return {
            "status": "❌ Not logged in",
            "error": str(e)
        }


def check_model_access(model_name: str) -> Tuple[str, str, str]:
    """
    Check if we can access a specific model
    Returns: (status, method_used, details)
    """
    print(f"  Testing {model_name}...")
    
    # Method 1: Try to load tokenizer (lightest test)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        del tokenizer  # Free memory
        return "✅ Accessible", "Tokenizer", "Full access confirmed"
    except Exception as tokenizer_error:
        pass
    
    # Method 2: Try to load config only
    try:
        config = AutoConfig.from_pretrained(model_name)
        del config
        return "🟡 Config Only", "Config", "Partial access - may need gated repo permission"
    except Exception as config_error:
        pass
    
    # Method 3: Check if it's a permission issue vs not found
    try:
        api = HfApi()
        repo_info = api.repo_info(model_name)
        if repo_info.gated:
            return "🔒 Gated", "API", "Requires access request approval"
        else:
            return "❌ Error", "API", "Unknown access issue"
    except Exception as api_error:
        if "403" in str(api_error) or "Forbidden" in str(api_error):
            return "🔒 Permission Denied", "API", "Requires access request"
        elif "404" in str(api_error) or "not found" in str(api_error).lower():
            return "❓ Not Found", "API", "Model may not exist"
        else:
            return "❌ Network Error", "API", f"Error: {str(api_error)[:50]}..."


def print_access_urls(models: Dict[str, str]):
    """Print URLs for requesting access"""
    print("\n🔗 Request Access URLs:")
    for name, model_id in models.items():
        print(f"   {name}: https://huggingface.co/{model_id}")


def print_token_setup_instructions():
    """Print instructions for setting up HuggingFace token"""
    print("\n🔧 HuggingFace Token Setup Instructions:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create new token or edit existing one")
    print("3. Enable these permissions:")
    print("   ✓ Read access to contents of all public repos")
    print("   ✓ Read access to contents of all repos you can access")
    print("   ✓ Access to gated repos you can access")
    print("4. Copy the token and run: huggingface-cli login")
    print("5. Paste your token when prompted")


def main():
    print("🔍 Meta Llama Model Access Checker")
    print("=" * 80)
    
    # Check HuggingFace login status
    print("\n📋 HuggingFace Login Status:")
    login_info = check_huggingface_login()
    
    for key, value in login_info.items():
        if key != "error":
            print(f"   {key}: {value}")
        else:
            print(f"   Error: {value}")
    
    if "❌" in login_info["status"]:
        print("\n⚠️  Please login to HuggingFace first:")
        print("   huggingface-cli login")
        return
    
    # Check target Meta Llama models
    print(f"\n🎯 Target Meta Llama Models ({len(TARGET_MODELS)} models):")
    print("-" * 60)
    
    accessible_models = []
    gated_models = []
    error_models = []
    
    for name, model_id in TARGET_MODELS.items():
        status, method, details = check_model_access(model_id)
        print(f"   {name:15} | {status:20} | {details}")
        
        if "✅" in status:
            accessible_models.append((name, model_id))
        elif "🔒" in status:
            gated_models.append((name, model_id))
        else:
            error_models.append((name, model_id))
        
        time.sleep(0.5)  # Be nice to HuggingFace API
    
    # Check alternative models
    print(f"\n🔄 Alternative Open Source Models ({len(ALTERNATIVE_MODELS)} models):")
    print("-" * 60)
    
    accessible_alternatives = []
    
    for name, model_id in ALTERNATIVE_MODELS.items():
        status, method, details = check_model_access(model_id)
        print(f"   {name:15} | {status:20} | {details}")
        
        if "✅" in status:
            accessible_alternatives.append((name, model_id))
        
        time.sleep(0.5)
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   ✅ Accessible Meta Llama models: {len(accessible_models)}/{len(TARGET_MODELS)}")
    print(f"   🔒 Gated Meta Llama models: {len(gated_models)}")
    print(f"   ❌ Error Meta Llama models: {len(error_models)}")
    print(f"   ✅ Accessible alternatives: {len(accessible_alternatives)}/{len(ALTERNATIVE_MODELS)}")
    
    # Detailed recommendations
    print(f"\n💡 Recommendations:")
    
    if len(accessible_models) == len(TARGET_MODELS):
        print("   🎉 All Meta Llama models are accessible! You can proceed with experiments.")
    elif len(accessible_models) > 0:
        print(f"   ✅ You have access to {len(accessible_models)} Meta Llama models.")
        print("   🔄 You can start experiments with accessible models.")
        if gated_models:
            print("   📝 Request access for remaining gated models below.")
    else:
        print("   🔒 No Meta Llama models are accessible.")
        print("   📝 You need to request access for all target models.")
        if accessible_alternatives:
            print(f"   🔄 Consider starting with {len(accessible_alternatives)} alternative models.")
    
    # Print access request URLs for gated models
    if gated_models:
        print(f"\n🔑 Request Access for Gated Models:")
        gated_dict = {name: model_id for name, model_id in gated_models}
        print_access_urls(gated_dict)
    
    # Print token setup if needed
    if error_models or not accessible_models:
        print_token_setup_instructions()
    
    # Export results for further use
    results = {
        "accessible_meta_llama": accessible_models,
        "gated_meta_llama": gated_models,
        "error_meta_llama": error_models,
        "accessible_alternatives": accessible_alternatives
    }
    
    print(f"\n💾 Results saved for programmatic use:")
    print(f"   accessible_meta_llama: {len(accessible_models)} models")
    print(f"   accessible_alternatives: {len(accessible_alternatives)} models")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        
        # Exit code based on accessibility
        if results["accessible_meta_llama"]:
            sys.exit(0)  # Success - some models accessible
        else:
            sys.exit(1)  # No Meta Llama models accessible
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Check interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)