import os
import sys
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure the 'src' directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_and_normalize_data
from preprocessing import TextPreprocessor
from feature_extraction import get_tfidf_features

def train_and_evaluate_ml():
    """
    Trains traditional machine learning models (Logistic Regression, Multinomial NB, SVM)
    on the preprocessed spam dataset using TF-IDF features and saves the metrics.
    """
    # 1. Check/Load processed dataset
    processed_path = "data/processed/processed_data.csv"
    if not os.path.exists(processed_path):
        print("Processed dataset not found. Running data loader first...")
        load_and_normalize_data()
        
    df = pd.read_csv(processed_path)
    
    # 2. Run preprocessing
    print("Preprocessing text dataset...")
    preprocessor = TextPreprocessor()
    df_preprocessed = preprocessor.preprocess_df(df)
    
    X = df_preprocessed['cleaned_text']
    y = df_preprocessed['label']
    
    # 3. Stratified Train-Test Split (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Training split: {X_train.shape[0]} samples")
    print(f"Testing split: {X_test.shape[0]} samples")
    
    # 4. Feature Extraction (TF-IDF Vectorizer fitted on Training only)
    X_train_vec, X_test_vec, vectorizer = get_tfidf_features(X_train, X_test)
    
    # 5. Define ML Models
    # SVM: kernel='linear' and probability=True to get prediction confidences
    models = {
        'Logistic Regression': LogisticRegression(C=1.0, class_weight='balanced', random_state=42),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    }
    
    results = []
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 6. Train & Evaluate each model
    for name, model in models.items():
        print(f"\nTraining model: {name}...")
        
        # Measure training time
        start_train = time.time()
        model.fit(X_train_vec, y_train)
        train_time = time.time() - start_train
        
        # Measure prediction time
        start_pred = time.time()
        y_pred = model.predict(X_test_vec)
        pred_time = time.time() - start_pred
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"[{name}] Results:")
        print(f"  - Accuracy:  {acc:.4f}")
        print(f"  - Precision: {prec:.4f}")
        print(f"  - Recall:    {rec:.4f}")
        print(f"  - F1 Score:  {f1:.4f}")
        print(f"  - Confusion Matrix:\n{cm}")
        print(f"  - Train Time: {train_time:.4f}s | Pred Time: {pred_time:.4f}s")
        
        # Save model pickle
        model_name_clean = name.lower().replace(" ", "_") + ".pkl"
        model_save_path = os.path.join("models", model_name_clean)
        joblib.dump(model, model_save_path)
        print(f"Model saved to '{model_save_path}'")
        
        # Extract confusion matrix values (TN, FP, FN, TP)
        tn, fp, fn, tp = cm.ravel()
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'Train_Time_Sec': train_time,
            'Pred_Time_Sec': pred_time,
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp)
        })
        
    # 7. Save results to comparison CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/model_comparison.csv", index=False)
    print("\nTraditional ML training complete. Metrics saved to 'results/model_comparison.csv'.")

if __name__ == "__main__":
    train_and_evaluate_ml()
