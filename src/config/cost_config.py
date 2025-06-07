"""
Cost configuration based on real measurements
"""

# Real measured costs from actual inference latency
# Measured on H100 GPUs with Qwen2.5 models
MEASURED_COSTS = {
    "qwen2.5-7b": 1.00,   # Baseline: 1.474s
    "qwen2.5-14b": 2.00,  # 2.947s (2.00x slower)
    "qwen2.5-32b": 4.20,  # 6.189s (4.20x slower)
    "qwen2.5-72b": 8.50,  # 12.525s (8.50x slower)
}

# Alternative name mappings
COST_MAPPINGS = {
    "7b": "qwen2.5-7b",
    "14b": "qwen2.5-14b", 
    "32b": "qwen2.5-32b",
    "72b": "qwen2.5-72b",
    "qwen3-7b": "qwen2.5-7b",  # Legacy support
    "qwen3-14b": "qwen2.5-14b",
    "qwen3-32b": "qwen2.5-32b", 
    "qwen3-72b": "qwen2.5-72b",
}

def get_measured_cost(model_name: str) -> float:
    """Get measured cost for a model."""
    # Normalize model name
    normalized_name = COST_MAPPINGS.get(model_name, model_name)
    
    if normalized_name in MEASURED_COSTS:
        return MEASURED_COSTS[normalized_name]
    
    # Fallback: extract size and estimate
    for size in ["7b", "14b", "32b", "72b"]:
        if size in model_name.lower():
            mapped_name = COST_MAPPINGS.get(size)
            if mapped_name and mapped_name in MEASURED_COSTS:
                return MEASURED_COSTS[mapped_name]
    
    # Default fallback
    return 1.0

def get_cost_vector() -> list:
    """Get the cost vector for all stages."""
    return [
        MEASURED_COSTS["qwen2.5-7b"],
        MEASURED_COSTS["qwen2.5-14b"], 
        MEASURED_COSTS["qwen2.5-32b"],
        MEASURED_COSTS["qwen2.5-72b"],
    ]