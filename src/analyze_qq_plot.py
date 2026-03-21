"""
Q-Q Plot Analysis Script
Automatically interpret Q-Q plot patterns and provide transformation recommendations
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def analyze_qq_plot(data, alpha=0.05):
    """
    Analyze Q-Q plot pattern and provide interpretation.
    
    Parameters:
    -----------
    data : array-like
        The data to analyze
    alpha : float
        Significance level for normality test (default: 0.05)
        
    Returns:
    --------
    dict : Dictionary with analysis results
    """
    # Remove missing values
    data_clean = np.array(data).flatten()
    data_clean = data_clean[~np.isnan(data_clean)]
    
    if len(data_clean) < 3:
        return {
            'pattern': 'Insufficient Data',
            'interpretation': 'Not enough data points for analysis',
            'action': 'Need at least 3 data points',
            'confidence': 'N/A'
        }
    
    # Get Q-Q plot data
    (osm, osr), (slope, intercept, r) = stats.probplot(data_clean, dist="norm")
    
    # Calculate residuals from the theoretical line
    theoretical_line = slope * osm + intercept
    residuals = osr - theoretical_line
    
    # Calculate skewness and kurtosis
    skewness = stats.skew(data_clean)
    kurtosis_val = stats.kurtosis(data_clean)
    
    # Shapiro-Wilk test
    _, shapiro_p = stats.shapiro(data_clean)
    
    # Analyze the residual pattern
    n = len(residuals)
    first_quarter = residuals[:n//4]
    last_quarter = residuals[-n//4:]
    
    # Mean residuals in tails
    left_tail_mean = np.mean(first_quarter)
    right_tail_mean = np.mean(last_quarter)
    
    # Standard deviation of residuals
    residual_std = np.std(residuals)
    
    # Correlation coefficient (R²)
    r_squared = r ** 2
    
    # Decision logic
    pattern = ""
    interpretation = ""
    action = ""
    symbol = ""
    
    # Check if approximately normal (high R² and passes Shapiro-Wilk)
    if r_squared > 0.98 and shapiro_p > alpha:
        pattern = "Points on Red Line"
        symbol = "✓"
        interpretation = "Data follows normal distribution closely"
        action = "No transformation needed"
    
    # Right-skewed (S-curve up at right)
    elif skewness > 0.5 and right_tail_mean > residual_std * 0.5:
        pattern = "S-Curve (Up at Right)"
        symbol = "⚠️"
        interpretation = "Right-skewed distribution - long tail on right, most values on left"
        if skewness > 1.5:
            action = "Strongly recommend: log(x) or log(x+1) transformation"
        elif skewness > 1.0:
            action = "Recommended: log(x) or sqrt(x) transformation"
        else:
            action = "Consider: sqrt(x) transformation"
    
    # Left-skewed (S-curve down at right)
    elif skewness < -0.5 and left_tail_mean < -residual_std * 0.5:
        pattern = "S-Curve (Down at Right)"
        symbol = "⚠️"
        interpretation = "Left-skewed distribution - long tail on left, most values on right"
        if skewness < -1.0:
            action = "Recommended: Square (x²) or reflect-and-log transformation"
        else:
            action = "Consider: Square (x²) or cube (x³) transformation"
    
    # Heavy tails (banana shape)
    elif kurtosis_val > 1.0 or (abs(left_tail_mean) > residual_std and abs(right_tail_mean) > residual_std):
        pattern = "Banana Shape (Heavy Tails)"
        symbol = "⚠️"
        interpretation = "Heavy-tailed distribution - more extreme values than normal (high kurtosis)"
        action = "Check for outliers, consider: Winsorization, robust methods, or Box-Cox transformation"
    
    # Light tails
    elif kurtosis_val < -1.0:
        pattern = "Flat Distribution (Light Tails)"
        symbol = "ℹ️"
        interpretation = "Light-tailed distribution - fewer extreme values than normal"
        action = "Distribution is flatter than normal, generally acceptable for most models"
    
    # Points spread away from line (generally not normal)
    elif r_squared < 0.95 or shapiro_p < alpha:
        pattern = "Points Spread Away from Line"
        symbol = "⚠️"
        interpretation = "Data does not follow normal distribution"
        if abs(skewness) > 0.5:
            action = f"Consider transformation (skewness={skewness:.2f}). Try: Box-Cox or Yeo-Johnson"
        else:
            action = "Consider: Transformation, check for multimodality, or use non-parametric methods"
    
    # Borderline normal
    else:
        pattern = "Approximately Normal"
        symbol = "✓"
        interpretation = "Data is roughly normal with minor deviations"
        action = "Acceptable for most linear models, monitor residuals"
    
    return {
        'pattern': pattern,
        'symbol': symbol,
        'interpretation': interpretation,
        'action': action,
        'statistics': {
            'r_squared': round(r_squared, 4),
            'shapiro_p_value': round(shapiro_p, 4),
            'skewness': round(skewness, 3),
            'kurtosis': round(kurtosis_val, 3),
            'is_normal': shapiro_p > alpha
        }
    }


def plot_qq_with_analysis(data, title="Q-Q Plot Analysis", figsize=(14, 5)):
    """
    Create Q-Q plot with automatic analysis and interpretation.
    
    Parameters:
    -----------
    data : array-like
        The data to analyze
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    """
    # Clean data
    data_clean = np.array(data).flatten()
    data_clean = data_clean[~np.isnan(data_clean)]
    
    # Analyze
    analysis = analyze_qq_plot(data_clean)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Q-Q Plot
    stats.probplot(data_clean, dist="norm", plot=axes[0])
    axes[0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add pattern annotation
    axes[0].text(0.05, 0.95, f"{analysis['symbol']} {analysis['pattern']}", 
                 transform=axes[0].transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 fontsize=10, fontweight='bold')
    
    # Distribution histogram
    axes[1].hist(data_clean, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].axvline(np.mean(data_clean), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].axvline(np.median(data_clean), color='green', linestyle='--', linewidth=2, label='Median')
    axes[1].set_title('Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("="*80)
    print("Q-Q PLOT ANALYSIS")
    print("="*80)
    print(f"\n{analysis['symbol']} Pattern Detected: {analysis['pattern']}")
    print(f"\nInterpretation:")
    print(f"  {analysis['interpretation']}")
    print(f"\nRecommended Action:")
    print(f"  {analysis['action']}")
    print(f"\nStatistics:")
    for key, value in analysis['statistics'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print("="*80)
    
    return analysis


def compare_transformations(data, transformations=None):
    """
    Compare Q-Q plots of original data and common transformations.
    
    Parameters:
    -----------
    data : array-like
        The data to analyze
    transformations : dict, optional
        Dictionary of transformation functions. If None, uses defaults.
    """
    # Clean data
    data_clean = np.array(data).flatten()
    data_clean = data_clean[~np.isnan(data_clean)]
    
    # Shift data to positive if needed for log/sqrt
    min_val = np.min(data_clean)
    shift = 0 if min_val > 0 else abs(min_val) + 1
    data_positive = data_clean + shift
    
    # Default transformations
    if transformations is None:
        transformations = {
            'Original': data_clean,
            'Log': np.log(data_positive),
            'Square Root': np.sqrt(data_positive),
            'Square': data_clean ** 2,
            'Cube Root': np.cbrt(data_clean)
        }
    
    n_transforms = len(transformations)
    n_cols = 3
    n_rows = (n_transforms + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_transforms > 1 else [axes]
    
    results = []
    
    for idx, (name, transformed_data) in enumerate(transformations.items()):
        # Q-Q plot
        stats.probplot(transformed_data, dist="norm", plot=axes[idx])
        axes[idx].set_title(f'{name}', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        
        # Analyze
        analysis = analyze_qq_plot(transformed_data)
        
        # Add R² annotation
        r_squared = analysis['statistics']['r_squared']
        shapiro_p = analysis['statistics']['shapiro_p_value']
        
        axes[idx].text(0.05, 0.95, 
                      f"R²={r_squared:.3f}\np={shapiro_p:.3f}\n{analysis['symbol']}", 
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                      fontsize=9)
        
        results.append({
            'Transformation': name,
            'R²': r_squared,
            'Shapiro-Wilk p': shapiro_p,
            'Pattern': analysis['pattern'],
            'Is Normal': '✓' if analysis['statistics']['is_normal'] else '✗'
        })
    
    # Hide empty subplots
    for idx in range(n_transforms, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Q-Q Plot Comparison - Find Best Transformation', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    
    # Print results table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R²', ascending=False)
    
    print("\n" + "="*80)
    print("TRANSFORMATION COMPARISON RESULTS")
    print("="*80)
    print("\nRanked by R² (closeness to normal distribution):\n")
    print(results_df.to_string(index=False))
    print("\n" + "="*80)
    print("💡 TIP: Choose the transformation with highest R² and p > 0.05")
    print("="*80)
    
    return results_df


# Example usage
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv('../data/raw/medical_insurance_cost.csv')
    
    print("\n1️⃣ SINGLE VARIABLE ANALYSIS")
    print("-" * 80)
    analysis = plot_qq_with_analysis(df['charges'], title="Q-Q Plot Analysis: Insurance Charges")
    
    print("\n\n2️⃣ TRANSFORMATION COMPARISON")
    print("-" * 80)
    comparison = compare_transformations(df['charges'])
