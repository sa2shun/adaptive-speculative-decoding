#!/usr/bin/env python3
"""
Test the updated cost model with real measurements
"""

import sys
import yaml
sys.path.append('/home/sasaki/adaptive-speculative-decoding/src')

from config.cost_config import get_measured_cost, get_cost_vector, MEASURED_COSTS

def test_cost_model():
    """Test the real cost model integration."""
    
    print("🧪 Testing Real Cost Model Integration")
    print("=" * 50)
    
    # Test individual cost retrieval
    print("\n📊 Individual Model Costs:")
    for model_name in ["qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b"]:
        cost = get_measured_cost(model_name)
        print(f"  {model_name:12}: {cost:.2f}x")
    
    # Test legacy name mapping
    print("\n🔄 Legacy Name Mapping:")
    legacy_names = ["7b", "14b", "32b", "72b", "qwen3-7b", "qwen3-32b"]
    for name in legacy_names:
        cost = get_measured_cost(name)
        print(f"  {name:12}: {cost:.2f}x")
    
    # Test cost vector
    print("\n📈 Cost Vector:")
    cost_vector = get_cost_vector()
    print(f"  [7B, 14B, 32B, 72B] = {cost_vector}")
    
    # Verify correct ordering (larger models should be more expensive)
    print("\n✅ Validation:")
    costs = list(MEASURED_COSTS.values())
    is_ordered = all(costs[i] <= costs[i+1] for i in range(len(costs)-1))
    print(f"  Costs properly ordered: {is_ordered}")
    
    # Calculate speedup potential
    baseline_cost = MEASURED_COSTS["qwen2.5-72b"]  # Most expensive
    print(f"\n⚡ Speedup Potential:")
    for model_name, cost in MEASURED_COSTS.items():
        speedup = baseline_cost / cost
        print(f"  {model_name} vs 72B: {speedup:.2f}x faster")
    
    print("\n🎯 Real vs Theoretical Comparison:")
    theoretical_costs = {"qwen2.5-7b": 1.0, "qwen2.5-14b": 2.1, "qwen2.5-32b": 4.7, "qwen2.5-72b": 10.0}
    
    for model_name in MEASURED_COSTS:
        real_cost = MEASURED_COSTS[model_name]
        theo_cost = theoretical_costs[model_name]
        ratio = real_cost / theo_cost
        print(f"  {model_name}: Real={real_cost:.2f}x, Theoretical={theo_cost:.1f}x, Ratio={ratio:.2f}")
    
    print("\n" + "=" * 50)
    print("✅ Cost model integration test completed!")

def test_config_loading():
    """Test loading the updated configuration."""
    
    print("\n🔧 Testing Configuration Loading")
    print("=" * 50)
    
    try:
        with open('/home/sasaki/adaptive-speculative-decoding/configs/qwen2.5_models.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Configuration loaded successfully")
        print("\n📋 Model Stages:")
        
        for i, stage in enumerate(config['models']['stages']):
            name = stage['name']
            latency = stage['base_latency_ms']
            rel_cost = stage['relative_cost']
            print(f"  Stage {i}: {name} - {latency}ms ({rel_cost:.2f}x cost)")
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")

if __name__ == "__main__":
    test_cost_model()
    test_config_loading()