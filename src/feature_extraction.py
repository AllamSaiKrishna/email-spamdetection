import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(train_texts, test_texts, save_path="models/tfidf_vectorizer.pkl", max_features=5000):
    """
    Fits a TF-IDF Vectorizer on train_texts and transforms both train_texts and test_texts.
    Saves the trained vectorizer to save_path.
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Initializing TF-IDF Vectorizer with ngram_range=(1, 2) and max_features={max_features}...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), # unigrams and bigrams
        max_features=max_features,
        min_df=2,           # ignore words that appear in only 1 email
        max_df=0.95         # ignore words that appear in more than 95% of emails
    )
    
    print("Fitting TF-IDF Vectorizer on training texts only (to avoid data leakage)...")
    X_train_vec = vectorizer.fit_transform(train_texts)
    
    print("Transforming test texts...")
    X_test_vec = vectorizer.transform(test_texts)
    
    # Save vectorizer
    joblib.dump(vectorizer, save_path)
    print(f"Trained TF-IDF Vectorizer saved to '{save_path}'")
    
    return X_train_vec, X_test_vec, vectorizer

def load_tfidf_vectorizer(load_path="models/tfidf_vectorizer.pkl"):
    """
    Loads a saved TF-IDF Vectorizer from load_path.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"TF-IDF Vectorizer file not found at '{load_path}'")
    return joblib.load(load_path)
