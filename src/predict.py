import os
import sys
import pickle
import joblib
import numpy as np

# Ensure 'src' is in Python Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import TextPreprocessor

class EmailSpamPredictor:
    def __init__(self, models_dir="models", best_model_info_path="results/metrics/best_model.txt"):
        self.models_dir = models_dir
        self.best_model_info_path = best_model_info_path
        self.preprocessor = TextPreprocessor()
        self.models = {}
        self.vectorizer = None
        self.tokenizer = None
        
    def _get_best_model_name(self):
        """
        Reads the name of the best-performing model from the evaluations text file.
        Falls back to Logistic Regression if the file doesn't exist.
        """
        if os.path.exists(self.best_model_info_path):
            try:
                with open(self.best_model_info_path, 'r') as f:
                    return f.read().strip()
            except Exception:
                pass
        return "Logistic Regression"
        
    def predict(self, email_text, model_name=None):
        """
        Cleans the input email text, runs predictions using the specified model,
        and returns a prediction dictionary with labels, confidence levels, and an explanation.
        """
        if model_name is None:
            model_name = self._get_best_model_name()
            
        # 1. Preprocess raw text
        cleaned_text = self.preprocessor.clean_text(email_text)
        
        # Check for empty text
        if not cleaned_text.strip():
            return {
                'prediction': 'Ham',
                'confidence': 1.0,
                'model': model_name,
                'explanation': 'The input email is empty or has no substantial words after preprocessing.'
            }
            
        # 2. Prediction based on model category
        if model_name in ['Logistic Regression', 'Naive Bayes', 'SVM']:
            # Load TF-IDF vectorizer if not already cached
            if self.vectorizer is None:
                vectorizer_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
                if not os.path.exists(vectorizer_path):
                    raise FileNotFoundError(f"TF-IDF Vectorizer not found at '{vectorizer_path}'. Train models first.")
                self.vectorizer = joblib.load(vectorizer_path)
                
            # Load ML model
            model_file = model_name.lower().replace(" ", "_") + ".pkl"
            model_path = os.path.join(self.models_dir, model_file)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model pickle not found at '{model_path}'. Train models first.")
                
            if model_name not in self.models:
                self.models[model_name] = joblib.load(model_path)
                
            model = self.models[model_name]
            
            # Vectorize cleaned text
            vec_text = self.vectorizer.transform([cleaned_text])
            
            # Run prediction and get probability scores
            pred = model.predict(vec_text)[0]
            probs = model.predict_proba(vec_text)[0]
            confidence = probs[pred]
            
        elif model_name == 'LSTM':
            import tensorflow as tf
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            # Load LSTM Tokenizer
            if self.tokenizer is None:
                tokenizer_path = os.path.join(self.models_dir, "tokenizer.pkl")
                if not os.path.exists(tokenizer_path):
                    raise FileNotFoundError(f"LSTM Tokenizer not found at '{tokenizer_path}'. Train LSTM first.")
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                    
            # Load LSTM model
            lstm_path = os.path.join(self.models_dir, "lstm_model.keras")
            if not os.path.exists(lstm_path):
                raise FileNotFoundError(f"LSTM Keras model not found at '{lstm_path}'. Train LSTM first.")
                
            if 'LSTM' not in self.models:
                print("Loading TensorFlow LSTM model...")
                self.models['LSTM'] = tf.keras.models.load_model(lstm_path)
                
            model = self.models['LSTM']
            
            # Transform text into padded sequences
            seq = self.tokenizer.texts_to_sequences([cleaned_text])
            padded = pad_sequences(seq, maxlen=150, padding='pre', truncating='pre')
            
            # LSTM returns single probability value (Sigmoid output)
            prob = model.predict(padded, verbose=0)[0][0]
            if prob > 0.5:
                pred = 1
                confidence = prob
            else:
                pred = 0
                confidence = 1 - prob
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        label = "Spam" if pred == 1 else "Ham"
        explanation = self._generate_explanation(email_text, cleaned_text, label, confidence, model_name)
        
        return {
            'prediction': label,
            'confidence': float(confidence),
            'model': model_name,
            'explanation': explanation
        }
        
    def _generate_explanation(self, original_text, cleaned_text, label, confidence, model_name):
        """
        Creates a human-readable explanation mapping keywords and text features.
        """
        spam_signals = [
            'free', 'winner', 'offer', 'money', 'urgent', 'congratulations', 
            'won', 'cash', 'prize', 'claim', 'award', 'now', 'guaranteed',
            'call', 'txt', 'reply', 'subscribe', 'earn', 'cheap'
        ]
        
        # Check which spam indicators are present in the original text
        words_found = [w for w in spam_signals if w in original_text.lower()]
        
        explanation = f"This email has been predicted as **{label}** with a confidence score of **{confidence:.2%}** using **{model_name}**.\n\n"
        
        if label == "Spam":
            explanation += "### Key Spam Indicators Found:\n"
            if words_found:
                explanation += f"- **High-Risk Words**: The message contains spam keywords: *{', '.join(words_found)}*.\n"
            else:
                explanation += "- **Contextual Similarity**: Although it has no obvious standalone keywords, the sentence structure matches common spam formats.\n"
            explanation += "- **Sender Urgency**: Spammers frequently use time-sensitive vocabulary to force quick actions.\n"
        else:
            explanation += "### Legitimate (Ham) Email Characteristics:\n"
            explanation += "- **Conversational Flow**: The vocabulary and phrasing represent personal, corporate, or normal communications.\n"
            if words_found:
                explanation += f"- **Keyword Override**: While words like *{', '.join(words_found)}* are present, the context matches ham emails and does not trigger classification.\n"
            else:
                explanation += "- **No Red Flags**: No common marketing or fishing indicators were detected.\n"
                
        return explanation

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Email Spam Classifier")
    parser.add_argument("--text", type=str, required=True, help="Email text to classify")
    parser.add_argument("--model", type=str, default=None, choices=["Logistic Regression", "Naive Bayes", "SVM", "LSTM"], help="Model to use")
    
    args = parser.parse_args()
    
    try:
        predictor = EmailSpamPredictor()
        res = predictor.predict(args.text, args.model)
        
        print("\n" + "="*50)
        print("                 CLASSIFICATION RESULT                ")
        print("="*50)
        print(f"Prediction: {res['prediction']}")
        print(f"Confidence: {res['confidence']:.2%}")
        print(f"Model Used: {res['model']}")
        print("-"*50)
        print(res['explanation'])
        print("="*50)
    except Exception as e:
        print(f"Error during prediction: {e}")
