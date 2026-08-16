# Email Spam Detection Using Binary Classification

An end-to-end Machine Learning and Deep Learning project built as a B.Tech CSE (AI & ML) academic minor project. 

This repository implements a complete pipeline that classifies email and text messages as **Ham (Legitimate)** or **Spam (Malicious/Unsolicited)**. The project trains and evaluates four classifiers, provides interactive analysis notebooks, and hosts a professional user interface using Streamlit.

---

## 📌 Project Overview
Spam classification is a fundamental Natural Language Processing (NLP) task. With the growing frequency of phishing attempts and marketing clutter, relying on static rules is insufficient. This project compares traditional statistical ML methods with deep learning sequences to construct the optimal model based on the **F1-Score**.

### Key Features
- **Dynamic Dataset Loading**: Automatically identifies text and label column names in CSV files.
- **Robust Preprocessing Pipeline**: Strips HTML tags/URLs, normalizes characters, removes NLTK stopwords (protecting critical keywords), and applies WordNet Lemmatization.
- **Model Diversity**: Evaluates Logistic Regression, Multinomial Naive Bayes, Linear Support Vector Machines (SVM), and Recurrent Neural Networks (LSTM).
- **Interactive UI**: Streamlit web application showcasing live inference, explanations, performance charts, and confusion matrices.
- **Vibrant Presentation Notebook**: Jupyter Notebook demonstrating EDA and predictions.

---

## ⚙️ Technology Stack
- **Programming Language**: Python 3.12+
- **Machine Learning**: Scikit-Learn
- **Deep Learning**: TensorFlow / Keras
- **Data Engineering**: Pandas, NumPy
- **Natural Language Processing**: NLTK (Tokenizer, Stopwords, WordNet Lemmatizer)
- **Visualizations**: Matplotlib, Seaborn
- **Web Application**: Streamlit

---

## 📊 Dataset Specifications
The pipeline runs on the **SMS/Email Spam Collection** dataset, mapping values to:
- **Ham** (Legitimate) $\rightarrow$ `0`
- **Spam** (Unwanted/Phishing) $\rightarrow$ `1`

**Class Balance:**
- Ham: 4,825 instances (~87.0%)
- Spam: 747 instances (~13.0%)

---

## 🏛️ System Architecture
```text
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
```

---

## 📥 Installation and Setup

### 1. Clone the Project & Create Virtual Environment
Open your terminal/command prompt and run:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/MacOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

Follow these commands in sequence to run the entire pipeline:

### Step 1: Dataset Preparation
Prepares the folder structure, copies dataset from raw resources, and normalizes column headers.
```bash
python src/data_loader.py
```

### Step 2: Train Machine Learning Models
Trains Logistic Regression, Naive Bayes, and SVM on TF-IDF features and saves metrics.
```bash
python src/train_ml_models.py
```

### Step 3: Train LSTM Deep Learning Model
Trains the sequential LSTM network using TensorFlow and saves weights.
```bash
python src/train_lstm.py
```

### Step 4: Evaluate Models & Generate Visuals
Computes comparison charts and saves confusion matrix heatmaps under `results/plots/`.
```bash
python src/evaluate_models.py
```

### Step 5: Launch Streamlit Web UI
```bash
streamlit run app/app.py
```

---

## 🏆 Model Performance Comparison (Actual Trained Results)

Computed on the **stratified 20% test split** (1,033 samples):

| Model | Accuracy | Precision | Recall | F1 Score | Train Time | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM** (Champion) | **98.06%** | 91.11% | **93.89%** | **92.48%** | 1.42s | 0.048s |
| **LSTM** | 97.97% | 92.31% | 91.60% | 91.95% | 20.83s | 0.441s |
| **Logistic Regression** | 97.10% | 84.83% | **93.89%** | 89.13% | 0.014s | 0.000s |
| **Naive Bayes** | 97.19% | **99.04%** | 78.63% | 87.66% | 0.001s | 0.000s |

### Key Takeaways
- **Support Vector Machine (SVM)** achieved the highest F1-Score of **92.48%**, followed closely by **LSTM** at **91.95%**.
- **Naive Bayes** achieved the highest Precision (**99.04%**), indicating it rarely flags legitimate emails as spam (1 False Positive), but at the cost of lower Recall (**78.63%**).
- **LSTM** provides a highly balanced performance, capturing sequential context, but has higher training and inference time.

---

## 🎯 Sample Predictions (Demonstration)

### Test Case 1: Spam Example
**Input:** `"Congratulations! You have won a $1000 prize. Call 0800 now to claim your reward."`
- **Prediction:** `Spam`
- **Confidence:** `100.00%` (using SVM)
- **Detected Flags:** *congratulations, won, prize, claim, now, call*

### Test Case 2: Ham Example
**Input:** `"Hey, are we still meeting today at 5 PM for coffee?"`
- **Prediction:** `Ham`
- **Confidence:** `99.88%` (using SVM)
- **Detected Flags:** None (natural conversational structure)

---

## ⚠️ Limitations
- **Obfuscated Words**: Words written like `F-R-E-E` or `w1nner` can sometimes bypass the TF-IDF vocabulary.
- **Short Texts**: Extremely short messages (e.g. "Ok", "Call me") lack context and may default to Ham.
- **Static Retraining**: The model must be retrained to adapt to modern spam/phishing campaign formats.

---

## 🔮 Future Scope
1. **Transformer Ensembles**: Transition to `DistilBERT` or `RoBERTa` for state-of-the-art context mapping.
2. **Obfuscation Robustness**: Add character-level regex filters to standardize spaced or substituted spam words.
3. **API Integrations**: Construct a browser extension or server webhook mapping classifications directly into inbox folders.
4. **Explainable AI (XAI)**: Integrate LIME or SHAP to highlight exactly which words contributed to the classification.
