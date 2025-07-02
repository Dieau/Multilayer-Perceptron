import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlp.utils import Colors

def plot_seed_search(results):
    if not results:
        print(f"{Colors.YELLOW}No seed search results to plot.{Colors.NC}")
        return

    seeds = [r['seed'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    best_result = max(results, key=lambda x: x['accuracy'])
    
    fig, axs = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Training Analysis Across Different Seeds', fontsize=16)

    # 1. Bar plot for final seed accuracies
    bars = axs[0].bar(seeds, accuracies, color='lightblue', label='Accuracy')
    best_bar = bars[seeds.index(best_result['seed'])]
    best_bar.set_color('salmon')
    axs[0].set_xlabel('Initialization Seed')
    axs[0].set_ylabel('Final Validation Accuracy')
    axs[0].set_title('Seed Performance Comparison')
    axs[0].set_xticks(seeds)
    axs[0].set_ylim(bottom=max(0, min(accuracies) - 0.05 if accuracies else 0))
    axs[0].axhline(y=best_result['accuracy'], color='r', linestyle='--', label=f"Best Acc: {best_result['accuracy']:.2%}")
    axs[0].legend()

    # 2. Loss curves for all seeds
    for result in results:
        style = 'r-' if result['seed'] == best_result['seed'] else 'gray'
        alpha = 1.0 if result['seed'] == best_result['seed'] else 0.4
        label = f"Best (Seed {result['seed']})" if result['seed'] == best_result['seed'] else None
        axs[1].plot(result['history']['val_loss'], style, alpha=alpha, label=label)
    axs[1].set_title('Validation Loss Curves per Seed')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    if any(r['seed'] == best_result['seed'] for r in results): axs[1].legend()

    # 3. Accuracy curves for all seeds
    for result in results:
        style = 'r-' if result['seed'] == best_result['seed'] else 'gray'
        alpha = 1.0 if result['seed'] == best_result['seed'] else 0.4
        label = f"Best (Seed {result['seed']})" if result['seed'] == best_result['seed'] else None
        axs[2].plot(result['history']['val_acc'], style, alpha=alpha, label=label)
    axs[2].set_title('Validation Accuracy Curves per Seed')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Accuracy')
    axs[2].grid(True, linestyle='--', alpha=0.6)
    if any(r['seed'] == best_result['seed'] for r in results): axs[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_prediction_results(history, y_true_labels, y_pred_labels, y_pred_probs):
    """
    Creates a single figure with three subplots for a comprehensive prediction analysis.
    1. Learning curves from the model's training.
    2. A confusion matrix of the prediction results.
    3. Detailed probability bar charts for a few sample predictions.
    """
    print(f"\n{Colors.CYAN}--- Generating Prediction Analysis Report ---{Colors.NC}")
    
    # --- Create Figure with Subplots ---
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_cm = fig.add_subplot(gs[:, 1])
    ax_detail = fig.add_subplot(gs[:, 2])
    
    fig.suptitle('Model Performance and Prediction Analysis', fontsize=16)

    # --- 1. Plot Learning Curves (Historic Metric) ---
    epochs_range = range(1, len(history['train_loss']) + 1)
    ax_loss.plot(epochs_range, history['train_loss'], label='Training Loss')
    ax_loss.plot(epochs_range, history['val_loss'], label='Validation Loss')
    ax_loss.set_title('Loss Curves')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True)

    ax_acc.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    ax_acc.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    ax_acc.set_title('Accuracy Curves')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()
    ax_acc.grid(True)

    # --- 2. Plot Confusion Matrix ---
    labels = ['Malignant', 'Benign']
    cm = np.zeros((2, 2), dtype=int)
    for i in range(len(y_true_labels)):
        cm[y_true_labels[i], y_pred_labels[i]] += 1
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax_cm)
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('Actual Label')
    ax_cm.set_title('Prediction Confusion Matrix')

    # --- 3. Plot Detailed Prediction Samples ---
    num_samples_to_show = 5
    sample_indices = np.random.choice(len(y_true_labels), size=min(num_samples_to_show, len(y_true_labels)), replace=False)
    
    bar_data = []
    bar_labels = []
    for idx in sample_indices:
        true_label_str = 'M' if y_true_labels[idx] == 0 else 'B'
        bar_data.append(y_pred_probs[:, idx])
        bar_labels.append(f'Sample {idx}\n(True: {true_label_str})')

    bar_df = pd.DataFrame(np.array(bar_data).T, index=['P(Malignant)', 'P(Benign)'], columns=bar_labels)
    bar_df.plot(kind='bar', ax=ax_detail, colormap='viridis', alpha=0.7)
    ax_detail.set_title('Sample Prediction Probabilities')
    ax_detail.set_ylabel('Probability')
    ax_detail.set_xticklabels(ax_detail.get_xticklabels(), rotation=0)
    ax_detail.set_ylim(0, 1.1)
    ax_detail.legend(title='Samples')
    ax_detail.grid(axis='y', linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()