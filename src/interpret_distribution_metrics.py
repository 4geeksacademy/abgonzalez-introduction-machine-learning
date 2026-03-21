"""
Script to interpret Skewness and Kurtosis values for data distribution analysis
"""

import pandas as pd
import numpy as np


def interpret_skewness(skew_value):
    """
    Interpret skewness value and provide recommendations.
    
    Parameters:
    -----------
    skew_value : float
        The skewness value
        
    Returns:
    --------
    dict : Dictionary with interpretation details
    """
    if -0.5 <= skew_value <= 0.5:
        return {
            'category': 'Fairly Symmetric',
            'symbol': '✓',
            'description': 'Distribution is approximately symmetric (normal-like)',
            'action': 'No transformation needed',
            'severity': 'good'
        }
    elif 0.5 < skew_value <= 1.0:
        return {
            'category': 'Moderately Right-Skewed',
            'symbol': '⚠️',
            'description': 'Long tail on the right, most values on the left',
            'action': 'Consider: sqrt or log transformation',
            'severity': 'moderate'
        }
    elif skew_value > 1.0:
        return {
            'category': 'Highly Right-Skewed',
            'symbol': '⚠️⚠️',
            'description': 'Severe right tail, values heavily concentrated on left',
            'action': 'Recommended: log or Box-Cox transformation',
            'severity': 'high'
        }
    elif -1.0 <= skew_value < -0.5:
        return {
            'category': 'Moderately Left-Skewed',
            'symbol': '⚠️',
            'description': 'Long tail on the left, most values on the right',
            'action': 'Consider: square or cube transformation',
            'severity': 'moderate'
        }
    else:  # skew_value < -1.0
        return {
            'category': 'Highly Left-Skewed',
            'symbol': '⚠️⚠️',
            'description': 'Severe left tail, values heavily concentrated on right',
            'action': 'Recommended: reflect and transform (e.g., max-value transformation)',
            'severity': 'high'
        }


def interpret_kurtosis(kurt_value):
    """
    Interpret kurtosis value and provide recommendations.
    
    Parameters:
    -----------
    kurt_value : float
        The kurtosis value
        
    Returns:
    --------
    dict : Dictionary with interpretation details
    """
    if -2.0 <= kurt_value <= 2.0:
        return {
            'category': 'Mesokurtic (Normal-like)',
            'symbol': '✓',
            'description': 'Normal tail behavior, typical peak',
            'action': 'No action needed',
            'severity': 'good'
        }
    elif kurt_value > 2.0:
        return {
            'category': 'Leptokurtic (Heavy Tails)',
            'symbol': '⚠️',
            'description': 'Heavy tails with sharp peak, prone to outliers',
            'action': 'Check for outliers, consider robust methods or winsorization',
            'severity': 'moderate'
        }
    else:  # kurt_value < -2.0
        return {
            'category': 'Platykurtic (Light Tails)',
            'symbol': 'ℹ️',
            'description': 'Light tails with flat peak, uniform-like distribution',
            'action': 'Distribution is flatter than normal, generally safe',
            'severity': 'info'
        }


def interpret_distribution_stats(df, numeric_features=None):
    """
    Calculate and interpret skewness and kurtosis for numeric features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    numeric_features : list, optional
        List of numeric feature names. If None, auto-detect.
        
    Returns:
    --------
    pd.DataFrame : DataFrame with interpretations
    """
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    results = []
    
    for feature in numeric_features:
        skew = df[feature].skew()
        kurt = df[feature].kurtosis()
        
        skew_interp = interpret_skewness(skew)
        kurt_interp = interpret_kurtosis(kurt)
        
        results.append({
            'Feature': feature,
            'Skewness': round(skew, 3),
            'Skew_Interpretation': f"{skew_interp['symbol']} {skew_interp['category']}",
            'Skew_Action': skew_interp['action'],
            'Kurtosis': round(kurt, 3),
            'Kurt_Interpretation': f"{kurt_interp['symbol']} {kurt_interp['category']}",
            'Kurt_Action': kurt_interp['action']
        })
    
    return pd.DataFrame(results)


def print_detailed_interpretation(df, numeric_features=None):
    """
    Print a detailed, formatted interpretation of distribution statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    numeric_features : list, optional
        List of numeric feature names. If None, auto-detect.
    """
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("="*100)
    print("DISTRIBUTION ANALYSIS - SKEWNESS & KURTOSIS INTERPRETATION")
    print("="*100)
    
    for feature in numeric_features:
        skew = df[feature].skew()
        kurt = df[feature].kurtosis()
        
        skew_interp = interpret_skewness(skew)
        kurt_interp = interpret_kurtosis(kurt)
        
        print(f"\n📊 Feature: {feature}")
        print("-" * 100)
        
        # Skewness
        print(f"   SKEWNESS: {skew:.3f}")
        print(f"   {skew_interp['symbol']} Category: {skew_interp['category']}")
        print(f"   Description: {skew_interp['description']}")
        print(f"   Action: {skew_interp['action']}")
        
        print()
        
        # Kurtosis
        print(f"   KURTOSIS: {kurt:.3f}")
        print(f"   {kurt_interp['symbol']} Category: {kurt_interp['category']}")
        print(f"   Description: {kurt_interp['description']}")
        print(f"   Action: {kurt_interp['action']}")
    
    print("\n" + "="*100)
    print("LEGEND:")
    print("="*100)
    print("✓ = Good (no action needed)")
    print("⚠️ = Moderate concern (consider transformation)")
    print("⚠️⚠️ = High concern (transformation recommended)")
    print("ℹ️ = Informational (generally acceptable)")
    print("\n" + "="*100)


def get_transformation_recommendations(df, numeric_features=None):
    """
    Get a summary of features that need transformation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    numeric_features : list, optional
        List of numeric feature names. If None, auto-detect.
        
    Returns:
    --------
    dict : Dictionary with transformation recommendations
    """
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    recommendations = {
        'no_action': [],
        'consider_transform': [],
        'highly_recommend_transform': []
    }
    
    for feature in numeric_features:
        skew = df[feature].skew()
        kurt = df[feature].kurtosis()
        
        skew_interp = interpret_skewness(skew)
        
        if skew_interp['severity'] == 'good':
            recommendations['no_action'].append(feature)
        elif skew_interp['severity'] == 'moderate':
            recommendations['consider_transform'].append({
                'feature': feature,
                'skewness': skew,
                'action': skew_interp['action']
            })
        elif skew_interp['severity'] == 'high':
            recommendations['highly_recommend_transform'].append({
                'feature': feature,
                'skewness': skew,
                'kurtosis': kurt,
                'action': skew_interp['action']
            })
    
    return recommendations


def print_transformation_summary(df, numeric_features=None):
    """
    Print a summary of transformation recommendations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    numeric_features : list, optional
        List of numeric feature names. If None, auto-detect.
    """
    recommendations = get_transformation_recommendations(df, numeric_features)
    
    print("="*100)
    print("TRANSFORMATION RECOMMENDATIONS SUMMARY")
    print("="*100)
    
    print(f"\n✓ Features that are GOOD as-is ({len(recommendations['no_action'])}):")
    if recommendations['no_action']:
        for feature in recommendations['no_action']:
            print(f"   • {feature}")
    else:
        print("   None")
    
    print(f"\n⚠️ Features to CONSIDER transforming ({len(recommendations['consider_transform'])}):")
    if recommendations['consider_transform']:
        for item in recommendations['consider_transform']:
            print(f"   • {item['feature']} (skew: {item['skewness']:.3f}) → {item['action']}")
    else:
        print("   None")
    
    print(f"\n⚠️⚠️ Features that HIGHLY NEED transformation ({len(recommendations['highly_recommend_transform'])}):")
    if recommendations['highly_recommend_transform']:
        for item in recommendations['highly_recommend_transform']:
            print(f"   • {item['feature']} (skew: {item['skewness']:.3f}, kurt: {item['kurtosis']:.3f})")
            print(f"     → {item['action']}")
    else:
        print("   None")
    
    print("\n" + "="*100)


# Example usage
if __name__ == "__main__":
    # Example with sample data
    import pandas as pd
    
    # Load your dataset
    df = pd.read_csv('../data/raw/medical_insurance_cost.csv')
    
    # Option 1: Get DataFrame with interpretations
    print("\n OPTION 1: Tabular Interpretation")
    interpretation_df = interpret_distribution_stats(df)
    print(interpretation_df.to_string(index=False))
    
    # Option 2: Print detailed interpretation
    print("\n\n OPTION 2: Detailed Interpretation")
    print_detailed_interpretation(df)
    
    # Option 3: Get transformation summary
    print("\n\n OPTION 3: Transformation Summary")
    print_transformation_summary(df)
