import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure 'src' is in Python Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def generate_evaluation_plots():
    """
    Loads comparison metrics from results/model_comparison.csv,
    saves visualization plots (grouped metrics, train/pred times),
    generates styled confusion matrices, and writes the best model selection.
    """
    csv_path = "results/model_comparison.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Model comparison file '{csv_path}' not found. "
            "Please run the model training scripts first."
        )
        
    df = pd.read_csv(csv_path)
    
    print("\n=======================================================")
    print("                MODEL PERFORMANCE COMPARISON           ")
    print("=======================================================")
    print(df.to_string(index=False))
    print("=======================================================\n")
    
    # 1. Identify the best model based on F1 Score (standard for spam detection)
    best_idx = df['F1 Score'].idxmax()
    best_model = df.loc[best_idx, 'Model']
    best_f1 = df.loc[best_idx, 'F1 Score']
    best_acc = df.loc[best_idx, 'Accuracy']
    
    print(f"*** Best Performing Model: {best_model} ***")
    print(f"   - F1-Score: {best_f1:.4f}")
    print(f"   - Accuracy: {best_acc:.4f}")
    print("   (Selection is based on F1-Score since both false positives and false negatives are critical.)\n")
    
    # Save the best model name to a text file for the app / predict modules to read
    os.makedirs("results/metrics", exist_ok=True)
    best_model_path = "results/metrics/best_model.txt"
    with open(best_model_path, "w") as f:
        f.write(best_model)
    print(f"Saved best model name to '{best_model_path}'")
    
    # Set directory for plots
    os.makedirs("results/plots", exist_ok=True)
    
    # Use seaborn style
    sns.set_theme(style="whitegrid")
    
    # 2. Performance Comparison Bar Chart (Accuracy, Precision, Recall, F1)
    plt.figure(figsize=(11, 7))
    df_melted = df.melt(
        id_vars=['Model'], 
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'], 
        var_name='Metric', 
        value_name='Score'
    )
    
    ax = sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='viridis')
    plt.title('Model Performance Comparison (ML vs Deep Learning)', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 1.05)
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
    plt.xlabel('Spam Detection Model', fontsize=12)
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Put text labels on the bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f'{height:.3f}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='bottom', 
                fontsize=8, fontweight='semibold',
                xytext=(0, 2), 
                textcoords='offset points'
            )
            
    plt.tight_layout()
    plots_metrics_path = "results/plots/model_comparison_metrics.png"
    plt.savefig(plots_metrics_path, dpi=300)
    plt.close()
    print(f"Saved performance metrics bar chart to '{plots_metrics_path}'")
    
    # 3. Training & Prediction time comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Train time plot
    sns.barplot(data=df.sort_values(by='Train_Time_Sec'), x='Model', y='Train_Time_Sec', ax=axes[0], palette='Blues_d')
    axes[0].set_title('Training Time (Lower is Better)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Time (Seconds)', fontsize=11)
    axes[0].set_xlabel('Model', fontsize=11)
    
    # Predict time plot
    sns.barplot(data=df.sort_values(by='Pred_Time_Sec'), x='Model', y='Pred_Time_Sec', ax=axes[1], palette='Reds_d')
    axes[1].set_title('Inference/Prediction Time (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Time (Seconds)', fontsize=11)
    axes[1].set_xlabel('Model', fontsize=11)
    
    plt.tight_layout()
    plots_time_path = "results/plots/model_time_comparison.png"
    plt.savefig(plots_time_path, dpi=300)
    plt.close()
    print(f"Saved time comparison chart to '{plots_time_path}'")
    
    # 4. Generate Confusion Matrices
    for _, row in df.iterrows():
        model_name = row['Model']
        tn = int(row['TN'])
        fp = int(row['FP'])
        fn = int(row['FN'])
        tp = int(row['TP'])
        
        cm = np.array([[tn, fp], [fn, tp]])
        total_samples = np.sum(cm)
        
        plt.figure(figsize=(6, 5))
        
        # Build cell labels
        names = ['True Negatives\n(Ham)', 'False Positives\n(Spam Warning)', 'False Negatives\n(Spam Missed)', 'True Positives\n(Spam)']
        counts = [f"{val}" for val in cm.flatten()]
        percentages = [f"{val/total_samples:.2%}" for val in cm.flatten()]
        
        labels = [f"{n}\nCount: {c}\n({p})" for n, c, p in zip(names, counts, percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        
        # Custom color map depending on class mapping
        sns.heatmap(
            cm, annot=labels, fmt='', cmap='Blues', cbar=False,
            xticklabels=['Predicted Ham', 'Predicted Spam'],
            yticklabels=['Actual Ham', 'Actual Spam'],
            annot_kws={"fontsize": 10, "fontweight": "semibold"}
        )
        
        plt.title(f'{model_name} Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
        plt.ylabel('Actual Label', fontsize=11)
        plt.xlabel('Predicted Label', fontsize=11)
        plt.tight_layout()
        
        cm_path = f"results/plots/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix plot for '{model_name}' to '{cm_path}'")
        
    print("\nAll evaluations completed successfully!")

if __name__ == "__main__":
    generate_evaluation_plots()
