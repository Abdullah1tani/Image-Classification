from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Function to evaluate and print metrics for a model
def evaluate_model(name, true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")
    return accuracy, precision, recall, f1, conf_matrix

# Function to evaluate and print metrics for a model
def mlp_evaluate_model(model, test_loader, name):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for features, _ in test_loader:
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
    accuracy = accuracy_score(test_labels_np, all_preds)
    conf_matrix = confusion_matrix(test_labels_np, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels_np, all_preds, average='weighted')
    print(f"{name} - Accuracy: {accuracy:.4f}")
    print(f"{name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")
    return accuracy, precision, recall, f1, conf_matrix

# Plot confusion matrices
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()
