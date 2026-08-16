import os
import shutil
import pandas as pd

def load_and_normalize_data(raw_path="data/raw/spam_data.csv", output_path="data/processed/processed_data.csv"):
    """
    Loads dataset from raw_path (copies from default location if needed),
    identifies text and label columns dynamically, normalizes classes,
    and writes to output_path.
    """
    # Ensure directories exist
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if raw file exists, copy from dataset/Spam_Data.csv if available
    if not os.path.exists(raw_path):
        default_source = os.path.join("dataset", "Spam_Data.csv")
        if os.path.exists(default_source):
            print(f"Copying dataset from {default_source} to {raw_path}")
            shutil.copy(default_source, raw_path)
        else:
            raise FileNotFoundError(f"Dataset not found at '{raw_path}' or '{default_source}'. Please provide the data.")
            
    print(f"Loading raw dataset from {raw_path}...")
    try:
        df = pd.read_csv(raw_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(raw_path, encoding='latin-1')
        
    print(f"Loaded dataset shape: {df.shape}")
    print(f"Columns in dataset: {df.columns.tolist()}")
    
    # 1. Identify text column dynamically
    text_cols = ['message', 'text', 'email', 'v2', 'body', 'content', 'sms']
    text_col = None
    for col in df.columns:
        if col.lower() in text_cols:
            text_col = col
            break
            
    if text_col is None:
        # Fallback 1: Look for object columns with long average string length
        for col in df.columns:
            if df[col].dtype == object and df[col].astype(str).str.len().mean() > 15:
                text_col = col
                break
                
    if text_col is None:
        # Fallback 2: default to the second column
        text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
    # 2. Identify label column dynamically
    label_cols = ['label', 'category', 'v1', 'class', 'spam', 'target', 'type']
    label_col = None
    for col in df.columns:
        if col.lower() in label_cols:
            label_col = col
            break
            
    if label_col is None:
        # Fallback: default to the first column (excluding the identified text column)
        for col in df.columns:
            if col != text_col:
                label_col = col
                break
        if label_col is None:
            label_col = df.columns[0]
            
    print(f"Identified text column: '{text_col}'")
    print(f"Identified label column: '{label_col}'")
    
    # Keep only the target columns
    processed_df = df[[text_col, label_col]].copy()
    processed_df.columns = ['text', 'label']
    
    # 3. Normalize Labels
    processed_df['label'] = processed_df['label'].astype(str).str.lower().str.strip()
    
    label_map = {
        'ham': 0, '0': 0, '0.0': 0, 'legit': 0, 'legitimate': 0,
        'spam': 1, '1': 1, '1.0': 1, 'unwanted': 1
    }
    
    processed_df['label'] = processed_df['label'].map(label_map)
    
    if processed_df['label'].isnull().any():
        print("Warning: Some labels could not be mapped automatically. Dropping null label rows.")
        processed_df = processed_df.dropna(subset=['label'])
        
    processed_df['label'] = processed_df['label'].astype(int)
    
    # Save processed dataframe
    processed_df.to_csv(output_path, index=False)
    print(f"Normalized dataset written successfully to '{output_path}' ({len(processed_df)} rows).")
    print(processed_df['label'].value_counts())
    return processed_df

if __name__ == "__main__":
    load_and_normalize_data()
