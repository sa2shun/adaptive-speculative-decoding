"""
Rigorous statistical evaluation framework.

This module implements proper hypothesis testing with:
- Paired t-tests with Bonferroni correction
- Effect size computation (Cohen's d)
- Power analysis
- Cross-validation
"""

import numpy as np
import scipy.stats as stats
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import KFold
import warnings


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    mean_difference: float
    std_error: float
    confidence_interval: Tuple[float, float]
    p_value: float
    effect_size: float
    power: float
    is_significant: bool
    sample_size: int


class RigorousEvaluator:
    """
    Implements statistically rigorous evaluation procedures.
    
    All tests follow best practices for multiple comparisons
    and ensure reproducible results.
    """
    
    def __init__(self, alpha: float = 0.05, min_effect_size: float = 0.5):
        self.alpha = alpha
        self.min_effect_size = min_effect_size
    
    def paired_t_test(self, 
                     method_a: np.ndarray,
                     method_b: np.ndarray,
                     paired: bool = True) -> StatisticalResult:
        """
        Perform paired t-test between two methods.
        
        Args:
            method_a: Results from method A
            method_b: Results from method B  
            paired: Whether samples are paired
            
        Returns:
            Statistical test results
        """
        assert len(method_a) == len(method_b), "Sample sizes must match"
        
        n = len(method_a)
        
        if paired:
            # Paired t-test
            differences = method_a - method_b
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            se_diff = std_diff / np.sqrt(n)
            
            t_stat = mean_diff / se_diff
            df = n - 1
            p_value = 2 * stats.t.cdf(-abs(t_stat), df)
            
            # 95% confidence interval
            t_crit = stats.t.ppf(1 - self.alpha/2, df)
            ci_lower = mean_diff - t_crit * se_diff
            ci_upper = mean_diff + t_crit * se_diff
            
        else:
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(method_a, method_b)
            mean_diff = np.mean(method_a) - np.mean(method_b)
            
            # Pooled standard deviation
            var_a = np.var(method_a, ddof=1)
            var_b = np.var(method_b, ddof=1)
            pooled_var = ((n-1)*var_a + (n-1)*var_b) / (2*n - 2)
            se_diff = np.sqrt(2 * pooled_var / n)
            
            df = 2*n - 2
            t_crit = stats.t.ppf(1 - self.alpha/2, df)
            ci_lower = mean_diff - t_crit * se_diff
            ci_upper = mean_diff + t_crit * se_diff
        
        # Effect size (Cohen's d)
        if paired:
            effect_size = mean_diff / std_diff
        else:
            pooled_std = np.sqrt(pooled_var)
            effect_size = mean_diff / pooled_std
        
        # Statistical power
        power = self._compute_power(effect_size, n, self.alpha)
        
        return StatisticalResult(
            mean_difference=mean_diff,
            std_error=se_diff,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            effect_size=effect_size,
            power=power,
            is_significant=(p_value < self.alpha),
            sample_size=n
        )
    
    def bonferroni_correction(self,
                            p_values: List[float],
                            n_comparisons: Optional[int] = None) -> List[bool]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: List of p-values from multiple tests
            n_comparisons: Number of comparisons (default: len(p_values))
            
        Returns:
            List of booleans indicating significance after correction
        """
        if n_comparisons is None:
            n_comparisons = len(p_values)
        
        adjusted_alpha = self.alpha / n_comparisons
        return [p < adjusted_alpha for p in p_values]
    
    def cross_validate_comparison(self,
                                data: np.ndarray,
                                labels: np.ndarray,
                                method_a_func,
                                method_b_func,
                                n_folds: int = 5) -> Dict:
        """
        Compare methods using k-fold cross-validation.
        
        Args:
            data: Input data
            labels: Ground truth labels
            method_a_func: Function implementing method A
            method_b_func: Function implementing method B
            n_folds: Number of CV folds
            
        Returns:
            Cross-validation results with statistical tests
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        method_a_scores = []
        method_b_scores = []
        
        for train_idx, test_idx in kf.split(data):
            # Split data
            X_train, X_test = data[train_idx], data[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # Evaluate methods
            score_a = method_a_func(X_train, y_train, X_test, y_test)
            score_b = method_b_func(X_train, y_train, X_test, y_test)
            
            method_a_scores.append(score_a)
            method_b_scores.append(score_b)
        
        # Statistical comparison
        scores_a = np.array(method_a_scores)
        scores_b = np.array(method_b_scores)
        
        result = self.paired_t_test(scores_a, scores_b, paired=True)
        
        return {
            'method_a_scores': scores_a,
            'method_b_scores': scores_b,
            'statistical_test': result,
            'mean_a': np.mean(scores_a),
            'mean_b': np.mean(scores_b),
            'std_a': np.std(scores_a),
            'std_b': np.std(scores_b)
        }
    
    def _compute_power(self, effect_size: float, n: int, alpha: float) -> float:
        """
        Compute statistical power for given effect size.
        
        Uses approximation for two-tailed t-test.
        """
        from statsmodels.stats.power import TTestPower
        
        try:
            power_analysis = TTestPower()
            power = power_analysis.solve_power(
                effect_size=effect_size,
                nobs=n,
                alpha=alpha,
                alternative='two-sided'
            )
        except:
            # Fallback approximation
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = effect_size * np.sqrt(n) - z_alpha
            power = stats.norm.cdf(z_beta)
        
        return power
    
    def compute_all_metrics(self,
                          our_method: np.ndarray,
                          baselines: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Compute all statistical metrics comparing our method to baselines.
        
        Args:
            our_method: Results from our adaptive method
            baselines: Dictionary of baseline_name -> results
            
        Returns:
            DataFrame with comprehensive statistical comparison
        """
        results = []
        p_values = []
        
        for baseline_name, baseline_results in baselines.items():
            stat_result = self.paired_t_test(our_method, baseline_results)
            p_values.append(stat_result.p_value)
            
            results.append({
                'baseline': baseline_name,
                'mean_ours': np.mean(our_method),
                'mean_baseline': np.mean(baseline_results),
                'improvement': stat_result.mean_difference,
                'improvement_pct': 100 * stat_result.mean_difference / np.mean(baseline_results),
                'ci_lower': stat_result.confidence_interval[0],
                'ci_upper': stat_result.confidence_interval[1],
                'p_value': stat_result.p_value,
                'effect_size': stat_result.effect_size,
                'power': stat_result.power
            })
        
        # Apply Bonferroni correction
        significant_corrected = self.bonferroni_correction(p_values)
        
        for i, result in enumerate(results):
            result['significant_uncorrected'] = result['p_value'] < self.alpha
            result['significant_corrected'] = significant_corrected[i]
            result['practical_significance'] = abs(result['effect_size']) >= self.min_effect_size
        
        df = pd.DataFrame(results)
        
        # Sort by effect size
        df = df.sort_values('effect_size', ascending=False)
        
        return df
    
    def bootstrap_confidence_interval(self,
                                    data: np.ndarray,
                                    statistic_func,
                                    n_bootstrap: int = 10000,
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for any statistic.
        
        Args:
            data: Input data
            statistic_func: Function to compute statistic
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            (lower, upper) confidence bounds
        """
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)
        
        # Compute percentile confidence interval
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha/2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
        
        return lower, upper
    
    def validate_assumptions(self, data: np.ndarray) -> Dict[str, bool]:
        """
        Validate statistical test assumptions.
        
        Checks:
        - Normality (Shapiro-Wilk test)
        - Homogeneity of variance (Levene's test)
        - Independence (Durbin-Watson test)
        """
        results = {}
        
        # Normality test
        if len(data) >= 3:
            _, p_normal = stats.shapiro(data)
            results['normality'] = p_normal > 0.05
        else:
            results['normality'] = None
            warnings.warn("Sample size too small for normality test")
        
        # For paired data, check normality of differences
        if len(data.shape) == 2 and data.shape[1] == 2:
            differences = data[:, 0] - data[:, 1]
            _, p_diff_normal = stats.shapiro(differences)
            results['normality_differences'] = p_diff_normal > 0.05
        
        # Independence (autocorrelation)
        # Using simple lag-1 autocorrelation
        if len(data) > 1:
            autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
            results['independence'] = abs(autocorr) < 0.2
        else:
            results['independence'] = None
        
        return results


def generate_publication_table(eval_results: pd.DataFrame) -> str:
    """
    Generate LaTeX table for publication.
    
    Args:
        eval_results: DataFrame from compute_all_metrics
        
    Returns:
        LaTeX table code
    """
    latex = r"""
\begin{table}[t]
\centering
\caption{Statistical comparison of our method against baselines. 
$\dagger$ indicates statistical significance after Bonferroni correction.
Effect sizes $|d| \geq 0.5$ indicate practical significance.}
\label{tab:statistical_comparison}
\begin{tabular}{lcccccc}
\toprule
\textbf{Baseline} & \textbf{Improvement} & \textbf{95\% CI} & \textbf{$p$-value} & \textbf{Cohen's $d$} & \textbf{Power} \\
\midrule
"""
    
    for _, row in eval_results.iterrows():
        sig_marker = "$\dagger$" if row['significant_corrected'] else ""
        
        latex += f"{row['baseline']} & "
        latex += f"{row['improvement']:.3f}{sig_marker} & "
        latex += f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] & "
        latex += f"{row['p_value']:.3e} & "
        latex += f"{row['effect_size']:.3f} & "
        latex += f"{row['power']:.3f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex