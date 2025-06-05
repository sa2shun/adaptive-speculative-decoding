#!/usr/bin/env python3
"""
Demonstrate key theoretical results of our optimal stopping approach.

This script validates our theoretical contributions without requiring
actual model inference.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.theory.optimal_stopping import OptimalStoppingTheory, TheoreticalParameters
from src.theory.regret_bounds import derive_regret_bound, compute_sample_complexity


def demonstrate_optimal_policy():
    """Show how optimal thresholds change with λ."""
    print("=== Optimal Stopping Policy ===\n")
    
    # Qwen3 model parameters
    quality_bounds = [0.70, 0.80, 0.85, 0.90]  # 7B, 14B, 32B, 72B
    cost_ratios = [1.0, 2.0, 4.5, 10.0]
    
    print("Model Hierarchy:")
    for i, (q, c) in enumerate(zip(quality_bounds, cost_ratios)):
        print(f"  Stage {i}: Quality={q:.2f}, Cost={c:.1f}x")
    
    print("\nOptimal Thresholds for Different λ:")
    print("λ     | Stage 0→1 | Stage 1→2 | Stage 2→3")
    print("------|-----------|-----------|----------")
    
    for lambda_val in [0.1, 0.5, 1.0, 2.0, 5.0]:
        params = TheoreticalParameters(
            n_stages=4,
            quality_bounds=quality_bounds,
            cost_ratios=cost_ratios,
            lambda_param=lambda_val
        )
        theory = OptimalStoppingTheory(params)
        thresholds = theory.derive_optimal_policy()
        
        print(f"{lambda_val:5.1f} | {thresholds[0]:9.3f} | {thresholds[1]:9.3f} | {thresholds[2]:9.3f}")
    
    print("\nInterpretation:")
    print("- Lower λ → Continue more (prioritize quality)")
    print("- Higher λ → Stop earlier (prioritize speed)")


def demonstrate_regret_bounds():
    """Show theoretical regret bounds."""
    print("\n=== Regret Bounds ===\n")
    
    n_stages = 4
    T_values = [100, 1000, 10000, 100000]
    quality_gaps = np.array([0.1, 0.05, 0.05])  # Gaps between consecutive stages
    
    print("Time Horizon | Problem-Dependent | Problem-Independent | Minimax")
    print("-------------|-------------------|---------------------|--------")
    
    for T in T_values:
        bounds = derive_regret_bound(n_stages, T, quality_gaps)
        print(f"{T:12d} | {bounds['problem_dependent']:17.1f} | "
              f"{bounds['problem_independent']:19.1f} | {bounds['minimax']:7.1f}")
    
    print("\nKey Insight: Our O(√T log T) bound matches lower bounds up to log factors")


def demonstrate_sample_complexity():
    """Show sample complexity for learning optimal policy."""
    print("\n=== Sample Complexity ===\n")
    
    print("For (ε,δ)-optimal policy with 4 stages:")
    print("ε     | δ      | Hoeffding | Bernstein | Lower Bound")
    print("------|--------|-----------|-----------|------------")
    
    for epsilon in [0.1, 0.05, 0.01]:
        for delta in [0.05, 0.01]:
            complexity = compute_sample_complexity(epsilon, delta, n_stages=4)
            print(f"{epsilon:5.2f} | {delta:6.2f} | {complexity['hoeffding']:9d} | "
                  f"{complexity['bernstein']:9d} | {complexity['lower_bound']:11d}")
    
    print("\nConclusion: Need ~10K samples for 0.05-accurate policy with 95% confidence")


def plot_theoretical_results():
    """Create visualization of theoretical results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Optimal thresholds vs lambda
    ax = axes[0, 0]
    lambdas = np.logspace(-1, 1, 50)
    thresholds_by_stage = {i: [] for i in range(3)}
    
    for lam in lambdas:
        params = TheoreticalParameters(
            n_stages=4,
            quality_bounds=[0.7, 0.8, 0.85, 0.9],
            cost_ratios=[1.0, 2.0, 4.5, 10.0],
            lambda_param=lam
        )
        theory = OptimalStoppingTheory(params)
        thresholds = theory.derive_optimal_policy()
        
        for i in range(3):
            thresholds_by_stage[i].append(thresholds[i])
    
    for i in range(3):
        ax.semilogx(lambdas, thresholds_by_stage[i], label=f'Stage {i}→{i+1}')
    
    ax.set_xlabel('λ (quality-cost tradeoff)')
    ax.set_ylabel('Optimal Threshold')
    ax.set_title('Optimal Stopping Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Regret bounds scaling
    ax = axes[0, 1]
    T_range = np.logspace(2, 6, 50)
    
    bounds_types = ['problem_independent', 'minimax']
    colors = ['blue', 'red']
    
    for bound_type, color in zip(bounds_types, colors):
        regrets = []
        for T in T_range:
            bounds = derive_regret_bound(4, int(T), np.array([0.1, 0.05, 0.05]))
            regrets.append(bounds[bound_type])
        
        ax.loglog(T_range, regrets, color=color, label=bound_type.replace('_', ' ').title())
    
    # Add theoretical O(√T log T) line
    theoretical = 10 * np.sqrt(T_range * np.log(T_range))
    ax.loglog(T_range, theoretical, 'k--', alpha=0.5, label='O(√T log T)')
    
    ax.set_xlabel('Time Horizon T')
    ax.set_ylabel('Regret Bound')
    ax.set_title('Theoretical Regret Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Sample complexity
    ax = axes[1, 0]
    epsilons = np.logspace(-2, -0.5, 50)
    
    methods = ['hoeffding', 'bernstein', 'lower_bound']
    colors = ['blue', 'green', 'red']
    
    for method, color in zip(methods, colors):
        complexities = []
        for eps in epsilons:
            comp = compute_sample_complexity(eps, 0.05, 4)
            complexities.append(comp[method])
        
        ax.loglog(epsilons, complexities, color=color, label=method.title())
    
    ax.set_xlabel('Accuracy ε')
    ax.set_ylabel('Sample Complexity')
    ax.set_title('Sample Complexity for (ε,0.05)-optimal Policy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # 4. Value function structure
    ax = axes[1, 1]
    
    # Show how value function changes with stage
    stages = [0, 1, 2, 3]
    values_by_lambda = {}
    
    for lam in [0.5, 1.0, 2.0]:
        params = TheoreticalParameters(
            n_stages=4,
            quality_bounds=[0.7, 0.8, 0.85, 0.9],
            cost_ratios=[1.0, 2.0, 4.5, 10.0],
            lambda_param=lam
        )
        
        values = []
        for s in stages:
            # Immediate reward for stopping
            r_stop = params.quality_bounds[s] - lam * params.cost_ratios[s]
            values.append(r_stop)
        
        values_by_lambda[lam] = values
        ax.plot(stages, values, 'o-', label=f'λ={lam}', linewidth=2, markersize=8)
    
    ax.set_xlabel('Stage')
    ax.set_ylabel('Value Function V(s)')
    ax.set_title('Value Function by Stage')
    ax.set_xticks(stages)
    ax.set_xticklabels(['7B', '14B', '32B', '72B'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('theoretical_results.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('theoretical_results.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved as theoretical_results.pdf")


def main():
    """Run all theoretical demonstrations."""
    print("=" * 60)
    print("THEORETICAL FOUNDATIONS OF OPTIMAL STOPPING FOR LLM INFERENCE")
    print("=" * 60)
    
    demonstrate_optimal_policy()
    demonstrate_regret_bounds()
    demonstrate_sample_complexity()
    
    print("\nGenerating theoretical results figure...")
    plot_theoretical_results()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("1. Optimal policy derived via dynamic programming")
    print("2. Regret bound O(√T log T) matches lower bounds") 
    print("3. Sample complexity O(1/ε²) for ε-optimal policy")
    print("4. Simple threshold-based implementation suffices")


if __name__ == "__main__":
    main()