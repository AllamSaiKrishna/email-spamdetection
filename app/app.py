import os
import sys
import pandas as pd
import streamlit as st

# Add 'src' to system path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

try:
    from predict import EmailSpamPredictor
except ImportError:
    # Fallback to local import if run differently
    from src.predict import EmailSpamPredictor

# Set page configuration with professional themes
st.set_page_config(
    page_title="Email Spam Classification Hub",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize predictor helper class
@st.cache_resource
def get_predictor():
    models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'metrics', 'best_model.txt')
    return EmailSpamPredictor(models_dir=models_path, best_model_info_path=best_model_path)

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading prediction pipeline: {e}. Please ensure models are trained.")

# Sidebar navigation
st.sidebar.title("📚 Spam Classification")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation Menu:", ["Home Page", "Prediction Page", "Model Comparison Page", "About Project Page"])

st.sidebar.markdown("---")
st.sidebar.info("💡 **Academic Minor Project**\nB.Tech CSE (AI & ML)\nTopic: Email Spam Detection")

# ----------------- HOME PAGE -----------------
if page == "Home Page":
    st.title("✉️ Email Spam Detection using Binary Classification")
    st.markdown("### Machine Learning & Deep Learning Based Spam Detection System")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        #### Project Overview
        Welcome to the **Email Spam Detection System**. This B.Tech minor academic project implements a state-of-the-art text classification hub designed to automatically filter unwanted or malicious messages (**Spam**) from legitimate communications (**Ham**).
        
        The project evaluates and compares two distinct algorithmic branches:
        1. **Traditional Machine Learning Models**: Leveraging TF-IDF Vectorization paired with:
            - **Logistic Regression** (L2 Regularized)
            - **Multinomial Naive Bayes** (Probabilistic classification)
            - **Support Vector Machine (SVM)** (Hyperplane maximization)
        2. **Deep Learning Model**: An **LSTM (Long Short-Term Memory)** recurrent neural network implemented via TensorFlow/Keras to capture sequential text relationships.
        
        #### How it works:
        - Copy and paste any suspicious text or actual email body inside the **Prediction Page**.
        - The classifier runs it through a standardized text preprocessing engine (lowercase conversion, HTML/URL stripping, punctuation filtering, stop-word removal, and WordNet Lemmatization).
        - Feature extractors map the cleaned text to embeddings/TF-IDF matrices.
        - The classifier predicts the probability score and explains the decision.
        """)
    
    with col2:
        st.success("🤖 **System Status**: Online")
        best_model_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'metrics', 'best_model.txt')
        if os.path.exists(best_model_txt):
            with open(best_model_txt, 'r') as f:
                best_model_name = f.read().strip()
            st.metric(label="Current Champion Model", value=best_model_name)
        else:
            st.warning("Run training to identify the best model.")
            
    st.markdown("---")
    st.markdown("#### System Architecture Flowchart")
    st.code("""
                 ┌─────────────────────┐
                 │   Email Dataset     │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Data Preprocessing  │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Feature Extraction  │
                 └──────────┬──────────┘
                            ↓
             ┌──────────────┴──────────────┐
             ↓                             ↓
    ┌─────────────────┐          ┌─────────────────┐
    │ Traditional ML  │          │ Deep Learning   │
    │                 │          │                 │
    │ Logistic Reg.   │          │ LSTM            │
    │ Naive Bayes     │          │                 │
    │ SVM             │          │                 │
    └────────┬────────┘          └────────┬────────┘
             ↓                            ↓
             └──────────────┬─────────────┘
                            ↓
                  ┌────────────────────┐
                  │ Model Evaluation   │
                  └─────────┬──────────┘
                            ↓
                  ┌────────────────────┐
                  │ Best Model         │
                  └─────────┬──────────┘
                            ↓
                  ┌────────────────────┐
                  │ Streamlit Web App  │
                  └────────────────────┘
    """, language="text")

# ----------------- PREDICTION PAGE -----------------
elif page == "Prediction Page":
    st.title("🔍 Real-time Email Spam Predictor")
    st.markdown("Input any message below to assess whether it represents legitimate communication or potential spam.")
    st.markdown("---")
    
    # Text input area
    email_text = st.text_area("Enter/Paste Email Content:", height=200, placeholder="Type or paste your message here (e.g., 'Congratulations! You won...')...")
    
    # Model Selection
    model_choices = ["Logistic Regression", "Naive Bayes", "SVM", "LSTM"]
    
    best_model_name = "Logistic Regression"
    best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'metrics', 'best_model.txt')
    if os.path.exists(best_model_path):
        with open(best_model_path, 'r') as f:
            best_model_name = f.read().strip()
            
    default_idx = model_choices.index(best_model_name) if best_model_name in model_choices else 0
    selected_model = st.selectbox("Choose Classification Engine:", model_choices, index=default_idx)
    
    if st.button("Check Email", type="primary"):
        if not email_text.strip():
            st.warning("Please enter email content before classification.")
        else:
            try:
                with st.spinner("Analyzing message context and vocabulary..."):
                    result = predictor.predict(email_text, selected_model)
                
                st.markdown("---")
                col_res1, col_res2 = st.columns([1, 1.2])
                
                with col_res1:
                    prediction = result['prediction']
                    confidence = result['confidence']
                    model_used = result['model']
                    
                    if prediction == "Spam":
                        st.error("🚨 **Prediction: SPAM (High Risk)**")
                    else:
                        st.success("✅ **Prediction: HAM (Legitimate / Safe)**")
                        
                    st.metric(label="Prediction Confidence Score", value=f"{confidence:.2%}")
                    st.info(f"🧠 **Model Employed**: {model_used}")
                    
                with col_res2:
                    st.markdown("#### Decision Breakdown")
                    st.markdown(result['explanation'])
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}. Make sure the trained models exist in the `models/` directory.")

# ----------------- MODEL COMPARISON PAGE -----------------
elif page == "Model Comparison Page":
    st.title("📊 Model Performance Comparison")
    st.markdown("Compare the quantitative evaluation metrics (computed on the stratified 20% test split) across all models.")
    st.markdown("---")
    
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'model_comparison.csv')
    
    if os.path.exists(csv_path):
        df_metrics = pd.read_csv(csv_path)
        
        # Display Table
        st.subheader("Model Performance Summary Table")
        st.dataframe(
            df_metrics[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Train_Time_Sec', 'Pred_Time_Sec']]
            .style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], color='#90EE90')
            .format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1 Score': '{:.4f}', 'Train_Time_Sec': '{:.4f}s', 'Pred_Time_Sec': '{:.4f}s'})
        )
        
        st.markdown("""
        *Insight*: The **F1-Score** is the primary metric used to declare the best model. 
        - **Precision** measures the ratio of true spam to all predicted spam. A higher precision is vital because a **False Positive** (classifying a critical business or personal email as spam) can have severe consequences.
        - **Recall** measures the ratio of spam detected to all actual spam. High recall keeps the inbox clutter-free of **False Negatives** (missed spam).
        """)
        
        # Graphs
        st.markdown("---")
        st.subheader("Comparative Visualizations")
        col_g1, col_g2 = st.columns(2)
        plots_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'plots')
        
        with col_g1:
            metrics_img = os.path.join(plots_path, 'model_comparison_metrics.png')
            if os.path.exists(metrics_img):
                st.image(metrics_img, caption="Performance Metrics Comparison", use_container_width=True)
                
        with col_g2:
            time_img = os.path.join(plots_path, 'model_time_comparison.png')
            if os.path.exists(time_img):
                st.image(time_img, caption="Training and Inference Execution Speeds", use_container_width=True)
                
        # Confusion Matrices
        st.markdown("---")
        st.subheader("Confusion Matrices Breakdown")
        st.markdown("Visualize the precise counts of True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP) for each classifier.")
        
        col_cm1, col_cm2 = st.columns(2)
        
        with col_cm1:
            lr_cm_img = os.path.join(plots_path, 'confusion_matrix_logistic_regression.png')
            if os.path.exists(lr_cm_img):
                st.image(lr_cm_img, caption="Logistic Regression Confusion Matrix", use_container_width=True)
                
            svm_cm_img = os.path.join(plots_path, 'confusion_matrix_svm.png')
            if os.path.exists(svm_cm_img):
                st.image(svm_cm_img, caption="SVM Confusion Matrix", use_container_width=True)
                
        with col_cm2:
            nb_cm_img = os.path.join(plots_path, 'confusion_matrix_naive_bayes.png')
            if os.path.exists(nb_cm_img):
                st.image(nb_cm_img, caption="Naive Bayes Confusion Matrix", use_container_width=True)
                
            lstm_cm_img = os.path.join(plots_path, 'confusion_matrix_lstm.png')
            if os.path.exists(lstm_cm_img):
                st.image(lstm_cm_img, caption="LSTM Confusion Matrix", use_container_width=True)
    else:
        st.warning("Performance results not found. Please complete the model training scripts to view metrics.")

# ----------------- ABOUT PROJECT PAGE -----------------
elif page == "About Project Page":
    st.title("ℹ️ Academic Project Details")
    st.markdown("---")
    
    st.markdown("""
    #### Problem Statement
    Electronic spam continues to be an active vector for malware propagation, phishing scams, and digital waste. Standard rules-based filtering struggles to keep pace with dynamic evasion techniques used by spammers. This project builds and analyzes classification models capable of identifying spam characteristics through Natural Language Processing.
    
    #### Project Methodology
    ```text
    Raw Dataset ──> Duplicate Removal ──> Cleaning ──> Tokenization & Lemmatization ──> Vectorization (TF-IDF / Embedding) ──> Classifier Training ──> Evaluation
    ```
    
    #### NLP Preprocessing Details:
    1. **Duplicate Check**: Standardizes and drops identical emails (403 duplicates removed from raw dataset).
    2. **HTML & URL Removal**: Removes raw tags (`<b>`, `<a>`) and hyperlinks which are highly prevalent in promotional spam.
    3. **Case Normalization**: Converts all text to lowercase to ensure consistency (e.g., 'WINNER' and 'winner' map to the same token).
    4. **Stop-word Filtering**: Removes common English stop-words (prepositions, conjunctions) but explicitly protects high-importance spam signals (e.g., *free, money, urgent, cash, claim, won*).
    5. **WordNet Lemmatization**: Resolves inflected words back to their root form (e.g., 'wins', 'winning', 'won' map to 'win').
    
    #### Description of Trained Models:
    - **Logistic Regression**: A linear model that estimates the probability of classification using a sigmoid function. Good baseline with fast inference.
    - **Multinomial Naive Bayes**: A probabilistic classifier based on Bayes' Theorem, calculating the joint probability of terms appearing in spam vs. ham. Highly efficient for text.
    - **Support Vector Machine (SVM)**: Learns a maximum-margin separating hyperplane in the high-dimensional TF-IDF space. Extremely accurate.
    - **LSTM (Long Short-Term Memory)**: A recurrent neural network (RNN) capable of capturing sequential word dependencies. Uses:
      - 10,000 vocab tokenizer mapping word indexes.
      - Embedding Layer mapping inputs into 100-dimensional continuous vectors.
      - LSTM layer with dropout to retain temporal sequence structure.
      - Dense output layer with Sigmoid mapping probability.
      
    #### Academic Viva / Presentation Tips:
    - If asked why **SVM** or **LSTM** performs better: SVM excels in high-dimensional sparse TF-IDF text features; LSTM excels by processing words in sequence, recognizing context and order.
    - Why is **F1-score** used over Accuracy? The dataset is highly imbalanced (only ~13% spam). A model predicting 'Ham' on every email achieves ~87% accuracy but an F1-score of 0. It is critical to use F1-score which harmonizes Precision and Recall.
    """)
