import json
import os
from pathlib import Path
from datetime import datetime

def save_encoding_rules(df_original, df_processed, notebook_name, 
                       transformations_applied=None, features_removed=None):
    """
    Generic function to save encoding rules from any preprocessing pipeline.
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original dataframe before processing
    df_processed : pd.DataFrame
        Processed dataframe after encoding
    notebook_name : str
        Name of the notebook (without .ipynb extension)
    transformations_applied : dict, optional
        Dictionary of transformations applied (e.g., {'target': 'log', 'features': ['sqrt(age)']})
    features_removed : dict, optional
        Dictionary of removed features with reasons
        
    Returns:
    --------
    str : Path to saved encoding rules file
    """
    
    # Create output directory
    output_dir = Path(f'../data/processed/{notebook_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize encoding rules dictionary
    encoding_rules = {
        "project_info": {
            "notebook": f"{notebook_name}.ipynb",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Automatically generated encoding rules"
        },
        "categorical_encoding": {},
        "feature_engineering": {
            "interaction_features": {},
            "polynomial_features": {},
            "other_features": {}
        },
        "target_transformation": {},
        "features_removed": features_removed or {},
        "final_features": {
            "numeric": [],
            "encoded_binary": [],
            "encoded_one_hot": [],
            "target": None,
            "total_features": len(df_processed.columns)
        },
        "usage_notes": {}
    }
    
    # Detect original categorical columns
    original_categorical = df_original.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Auto-detect binary encoded columns (columns ending with _encoded)
    for col in df_processed.columns:
        if col.endswith('_encoded'):
            original_col = col.replace('_encoded', '')
            
            if original_col in original_categorical:
                unique_vals = df_original[original_col].unique()
                
                # Create mapping based on the encoded values
                mapping = {}
                for val in unique_vals:
                    # Check what the encoded value is for this original value
                    sample_encoded = df_processed.loc[df_original[original_col] == val, col].iloc[0]
                    mapping[str(val)] = int(sample_encoded)
                
                encoding_rules["categorical_encoding"][original_col] = {
                    "type": "binary",
                    "mapping": mapping,
                    "encoded_column": col,
                    "original_values": [str(v) for v in unique_vals],
                    "description": f"Binary encoding for {original_col}"
                }
                
                encoding_rules["final_features"]["encoded_binary"].append(col)
    
    # Auto-detect one-hot encoded columns (columns with prefix_value pattern)
    one_hot_groups = {}
    processed_cols = set()
    
    for col in df_processed.columns:
        if '_' in col and col not in processed_cols:
            # Check if this is a one-hot encoded column
            prefix = col.rsplit('_', 1)[0]
            
            # Find all columns with same prefix
            prefix_cols = [c for c in df_processed.columns if c.startswith(f"{prefix}_")]
            
            if len(prefix_cols) > 1 and prefix in original_categorical:
                if prefix not in one_hot_groups:
                    original_values = df_original[prefix].unique().tolist()
                    
                    # Determine which category was dropped
                    suffix_values = [c.replace(f"{prefix}_", '') for c in prefix_cols]
                    dropped_category = [v for v in original_values if v not in suffix_values]
                    
                    one_hot_groups[prefix] = {
                        "type": "one-hot",
                        "method": "pd.get_dummies with drop_first=True",
                        "original_values": [str(v) for v in original_values],
                        "encoded_columns": prefix_cols,
                        "dropped_category": str(dropped_category[0]) if dropped_category else None,
                        "description": f"One-hot encoding for {prefix}"
                    }
                    
                    processed_cols.update(prefix_cols)
                    encoding_rules["final_features"]["encoded_one_hot"].extend(prefix_cols)
    
    encoding_rules["categorical_encoding"].update(one_hot_groups)
    
    # Detect interaction features (columns with _x_ or pattern like feature1_feature2)
    for col in df_processed.columns:
        if col not in df_original.columns and not col.endswith('_encoded'):
            # Check for common interaction patterns
            if any(base_col in col for base_col in df_original.columns):
                parts = col.split('_')
                if len(parts) >= 2:
                    encoding_rules["feature_engineering"]["interaction_features"][col] = {
                        "description": f"Interaction or derived feature: {col}",
                        "formula": "To be documented manually"
                    }
    
    # Identify numeric features
    numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
    original_numeric = df_original.select_dtypes(include=['number']).columns.tolist()
    
    # Pure numeric features (not encoded, interaction, or target transformations)
    pure_numeric = [c for c in numeric_cols 
                    if c in original_numeric 
                    and not c.endswith('_log') 
                    and not c.endswith('_sqrt')
                    and not c.endswith('_original')]
    
    encoding_rules["final_features"]["numeric"] = pure_numeric
    
    # Add transformations if provided
    if transformations_applied:
        encoding_rules["target_transformation"] = transformations_applied
        if "transformed_target" in transformations_applied:
            encoding_rules["final_features"]["target"] = transformations_applied["transformed_target"]
    
    # Add usage notes
    encoding_rules["usage_notes"] = {
        "prediction": "When making predictions, ensure new data follows the same encoding scheme",
        "categorical_handling": "For binary encoding, use the mapping provided. For one-hot encoding, set all dummies to 0 if the category is the dropped one",
        "missing_values": "Handle missing values before applying encodings",
        "feature_order": "Maintain the same feature order as in final_features for model compatibility"
    }
    
    # Save to JSON
    output_path = output_dir / 'encoding_rules.json'
    with open(output_path, 'w') as f:
        json.dump(encoding_rules, f, indent=4)
    
    print("="*80)
    print("ENCODING RULES SAVED SUCCESSFULLY")
    print("="*80)
    print(f"\n✓ File saved to: {output_path}")
    print(f"✓ File size: {output_path.stat().st_size} bytes")
    print(f"\n📊 Summary:")
    print(f"   • Binary encoded features: {len(encoding_rules['final_features']['encoded_binary'])}")
    print(f"   • One-hot encoded features: {len(encoding_rules['final_features']['encoded_one_hot'])}")
    print(f"   • Interaction features: {len(encoding_rules['feature_engineering']['interaction_features'])}")
    print(f"   • Numeric features: {len(encoding_rules['final_features']['numeric'])}")
    print(f"   • Total features: {encoding_rules['final_features']['total_features']}")
    print("="*80)
    
    return str(output_path)

