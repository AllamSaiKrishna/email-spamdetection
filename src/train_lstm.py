import os
import sys
import time
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure the 'src' directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_and_normalize_data
from preprocessing import TextPreprocessor

def train_and_evaluate_lstm():
    """
    Trains a Deep Learning LSTM model using TensorFlow/Keras on preprocessed texts.
    Pads sequences, fits and saves a Tokenizer, and saves the trained LSTM model.
    Appends the model's metrics to 'results/model_comparison.csv'.
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
    
    # Deferred TensorFlow imports to ensure they run only when script executes
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    # Set TensorFlow random seeds for reproducibility
    tf.random.set_seed(42)
    
    # 4. Tokenization & Sequence padding
    vocab_size = 10000
    max_len = 150
    embedding_dim = 100
    
    print(f"Fitting Keras Tokenizer on training text with vocab_size={vocab_size}...")
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    print("Converting texts to sequences and padding...")
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='pre', truncating='pre')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='pre', truncating='pre')
    
    # Save tokenizer object
    os.makedirs("models", exist_ok=True)
    tokenizer_path = "models/tokenizer.pkl"
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved successfully to '{tokenizer_path}'")
    
    # 5. Build LSTM Architecture
    print("Building TensorFlow/Keras LSTM model...")
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        SpatialDropout1D(0.2),
        LSTM(64, dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid') # Sigmoid for binary classification
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 6. Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint_path = "models/lstm_model.keras"
    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Compute class weights to address imbalance
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"Calculated class weights: {class_weight_dict}")
    
    # 7. Train Model
    print("Training LSTM model...")
    start_train = time.time()
    
    # Train using 10% validation split and class weights
    model.fit(
        X_train_pad, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    train_time = time.time() - start_train
    print(f"Training finished in {train_time:.2f} seconds.")
    
    # 8. Load best checkpoint for final evaluation
    if os.path.exists(checkpoint_path):
        print(f"Loading best weights from '{checkpoint_path}'...")
        model = tf.keras.models.load_model(checkpoint_path)
        
    # 9. Evaluate model
    print("Evaluating LSTM on test set...")
    start_pred = time.time()
    y_pred_prob = model.predict(X_test_pad)
    pred_time = time.time() - start_pred
    
    # Apply 0.5 decision threshold
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n[LSTM] Results:")
    print(f"  - Accuracy:  {acc:.4f}")
    print(f"  - Precision: {prec:.4f}")
    print(f"  - Recall:    {rec:.4f}")
    print(f"  - F1 Score:  {f1:.4f}")
    print(f"  - Confusion Matrix:\n{cm}")
    print(f"  - Pred Time: {pred_time:.4f}s")
    
    tn, fp, fn, tp = cm.ravel()
    
    lstm_results = {
        'Model': 'LSTM',
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
    }
    
    # 10. Update/Save results to model comparison CSV
    comparison_path = "results/model_comparison.csv"
    if os.path.exists(comparison_path):
        df_comp = pd.read_csv(comparison_path)
        # Drop previous LSTM row if it exists
        df_comp = df_comp[df_comp['Model'] != 'LSTM']
        df_comp = pd.concat([df_comp, pd.DataFrame([lstm_results])], ignore_index=True)
    else:
        df_comp = pd.DataFrame([lstm_results])
        
    df_comp.to_csv(comparison_path, index=False)
    print(f"LSTM metrics written/updated in '{comparison_path}' successfully.")

if __name__ == "__main__":
    train_and_evaluate_lstm()
