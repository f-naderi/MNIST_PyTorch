# evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def evaluate_model(model, test_loader, criterion, device):
    # Evaluate the model on the test dataset
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def get_predictions(model, test_loader, device):
    # Get all predictions and ground truth labels
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())
    
    return np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix(model, test_loader, device, class_names=None):
    # Plot the confusion matrix
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    y_pred, y_true = get_predictions(model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('images/confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return cm


def print_classification_report(model, test_loader, device):
    # Display the classification report
    y_pred, y_true = get_predictions(model, test_loader, device)
    class_names = [str(i) for i in range(10)]
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print("="*60)
    print(report)


def visualize_predictions(model, test_loader, device, num_images=10):
    # Display a number of model predictions
    model.eval()
    
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images[:num_images].to(device)
    labels = labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_images):
        img = images[i].cpu().squeeze()
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        
        axes[i].imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('images/predictions.png', dpi=100, bbox_inches='tight')
    plt.show()


def plot_training_history(history):
    # Plot the training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('images/training_history.png', dpi=100, bbox_inches='tight')
    plt.show()





def plot_roc_curves(model, test_loader, device, class_names=None):
    # Plot ROC curves for all classes (One-vs-Rest)
    
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.numpy())
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Binarize labels (One-vs-Rest)
    y_test_binarized = label_binarize(all_targets, classes=range(10))
    n_classes = y_test_binarized.shape[1]
    
    # Compute ROC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot the figure 
    plt.figure(figsize=(12, 10))
    
    # Plot micro-average
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC (AUC = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot for each class   
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Digit {class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Random classifier reference line  
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves - One vs Rest (MNIST)', fontsize=16)
    plt.legend(loc='lower right', fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Display AUC scores numerically     
    print("\nAUC Scores (One-vs-Rest):")
    print("-" * 40)
    for i in range(n_classes):
        print(f"  Digit {class_names[i]}: {roc_auc[i]:.4f}")
    print(f"  Micro-average: {roc_auc['micro']:.4f}")
    
    return roc_auc


def plot_precision_recall_curves(model, test_loader, device, class_names=None):
    # Plot Precision-Recall curves for all classes
    
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.numpy())
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Binarize labels 
    y_test_binarized = label_binarize(all_targets, classes=range(10))
    n_classes = y_test_binarized.shape[1]
    
    # Compute Precision-Recall for each class 
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_test_binarized[:, i], all_probs[:, i]
        )
        average_precision[i] = average_precision_score(
            y_test_binarized[:, i], all_probs[:, i]
        )
    
    # Compute micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test_binarized.ravel(), all_probs.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_test_binarized, all_probs, average="micro"
    )
    
    # Plot figure 
    plt.figure(figsize=(12, 10))
    
    # Plot micro-average
    plt.plot(recall["micro"], precision["micro"],
             label=f'micro-average PR (AP = {average_precision["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot for each class   
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'Digit {class_names[i]} (AP = {average_precision[i]:.3f})')
    
    # Baseline
    class_counts = np.bincount(all_targets)
    baseline = class_counts / len(all_targets)
    plt.axhline(y=np.mean(baseline), color='gray', linestyle='--', lw=2, 
                label=f'Baseline (random)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves - One vs Rest (MNIST)', fontsize=16)
    plt.legend(loc='lower left', fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/precision_recall_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
      
    # Display Average Precision numerically 
    print("\nAverage Precision Scores (One-vs-Rest):")
    print("-" * 40)
    for i in range(n_classes):
        print(f"  Digit {class_names[i]}: {average_precision[i]:.4f}")
    print(f"  Micro-average: {average_precision['micro']:.4f}")
    
    return average_precision