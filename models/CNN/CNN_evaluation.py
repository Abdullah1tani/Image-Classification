import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from CNN_implementation import cnn_test, LargeCNN, CNN, ShallowCNN, DeepCNN
from features.features_utils import select_subset

# define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the CIFAR-10 datasets with the specified transform
test_data = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)

# Select subsets: first 100 test samples per class
test_subset = select_subset(test_data, 100)

# Define DataLoader for batch processing
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

# initialize device and set it to GPU. if no GPU available use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize loss function
criterion = nn.CrossEntropyLoss()

# initialize default cnn model
cnn = CNN("VGG11", 3).to(device)

# initialize cnn models for depth evaluation
shallow_cnn = ShallowCNN().to(device)
deep_cnn = DeepCNN().to(device)

# initialize cnn models for size evaluation
large_cnn = LargeCNN("large_cnn", 5).to(device)
very_large_cnn = LargeCNN("very_large_cnn", 7).to(device)

# Load saved weights
cnn.load_state_dict(torch.load("VGG11_weights.pth"))

shallow_cnn.load_state_dict(torch.load("shallow_cnn_weights.pth"))
deep_cnn.load_state_dict(torch.load("deep_cnn_weights.pth"))

large_cnn.load_state_dict(torch.load("large_cnn_weights.pth"))
very_large_cnn.load_state_dict(torch.load("very_large_cnn_weights.pth"))

# Test the models
print("\nTesting VGG11:")
vgg11_metrics = cnn_test(cnn, test_loader, criterion, device, "VGG11")

print("\nTesting Shallow CNN:")
shallow_metrics = cnn_test(shallow_cnn, test_loader, criterion, device, "Shallow CNN")

print("\nTesting Deep CNN:")
deep_metrics = cnn_test(deep_cnn, test_loader, criterion, device, "Deep CNN")

print("\nTesting Large CNN:")
large_metrics = cnn_test(large_cnn, test_loader, criterion, device, "Large Kernel CNN")

print("\nTesting Very Large CNN:")
very_large_metrics = cnn_test(very_large_cnn, test_loader, criterion, device, "Very Large Kernel CNN")

# Save evaluation summary to display depth results
depth_evaluation_results = pd.DataFrame({
    'Model': ['VGG11', 'Shallow CNN', 'Deep CNN'],
    'Accuracy': [vgg11_metrics[0], shallow_metrics[0], deep_metrics[0]],
    'Precision': [vgg11_metrics[1], shallow_metrics[1], deep_metrics[1]],
    'Recall': [vgg11_metrics[2], shallow_metrics[2], deep_metrics[2]],
    'F1 Score': [vgg11_metrics[3], shallow_metrics[3], deep_metrics[3]]
})

print("\nDepth Evaluation Summary Table:")
print(depth_evaluation_results)

# Save evaluation summary to display different size results
size_evaluation_results = pd.DataFrame({
    'Model': ['VGG11', 'Large Kernel CNN', 'Very Large Kernel CNN'],
    'Accuracy': [vgg11_metrics[0], large_metrics[0], very_large_metrics[0]],
    'Precision': [vgg11_metrics[1], large_metrics[1],  very_large_metrics[1]],
    'Recall': [vgg11_metrics[2], large_metrics[2], very_large_metrics[2]],
    'F1 Score': [vgg11_metrics[3], large_metrics[3], very_large_metrics[3]]
})

print("\nKernel Size Evaluation Summary Table:")
print(size_evaluation_results)

# Save evaluation summary to display all results
evaluation_results = pd.DataFrame({
    'Model': ['VGG11', 'Shallow CNN', 'Deep CNN', 'Large Kernel CNN', 'Very Large Kernel CNN'],
    'Accuracy': [vgg11_metrics[0], shallow_metrics[0], deep_metrics[0], large_metrics[0], very_large_metrics[0]],
    'Precision': [vgg11_metrics[1], shallow_metrics[1], deep_metrics[1], large_metrics[1], very_large_metrics[1]],
    'Recall': [vgg11_metrics[2], shallow_metrics[2], deep_metrics[2], large_metrics[2], very_large_metrics[2]],
    'F1 Score': [vgg11_metrics[3], shallow_metrics[3], deep_metrics[3], large_metrics[3], very_large_metrics[3]]
})

print("\nAll Models Evaluation Summary Table:")
print(evaluation_results)

# Test the models
print("\nTesting VGG11:")
vgg11_metrics = cnn_test(cnn, test_loader, criterion, device, "VGG11 (Subset)")

print("\nTesting Shallow CNN:")
shallow_metrics = cnn_test(shallow_cnn, test_loader, criterion, device, "Shallow CNN (4 blocks)")

print("\nTesting Deep CNN:")
deep_metrics = cnn_test(deep_cnn, test_loader, criterion, device, "Deep CNN (12 blocks)")

print("\nTesting Large CNN:")
large_metrics = cnn_test(large_cnn, test_loader, criterion, device, "Large Kernel CNN (7x7)")

print("\nTesting Very Large CNN:")
very_large_metrics = cnn_test(very_large_cnn, test_loader, criterion, device, "Very Large Kernel CNN (9x9)")

# Save evaluation summary to display depth results
depth_evaluation_results = pd.DataFrame({
    'Model': ['VGG11 (Subset)', 'Shallow CNN', 'Deep CNN'],
    'Accuracy': [vgg11_metrics[0], shallow_metrics[0], deep_metrics[0]],
    'Precision': [vgg11_metrics[1], shallow_metrics[1], deep_metrics[1]],
    'Recall': [vgg11_metrics[2], shallow_metrics[2], deep_metrics[2]],
    'F1 Score': [vgg11_metrics[3], shallow_metrics[3], deep_metrics[3]]
})

print("\nDepth Evaluation Summary Table:")
print(depth_evaluation_results)

# Save evaluation summary to display different size results
size_evaluation_results = pd.DataFrame({
    'Model': ['VGG11 (Subset)', 'Large Kernel CNN', 'Very Large Kernel CNN'],
    'Accuracy': [vgg11_metrics[0], large_metrics[0], very_large_metrics[0]],
    'Precision': [vgg11_metrics[1], large_metrics[1],  very_large_metrics[1]],
    'Recall': [vgg11_metrics[2], large_metrics[2], very_large_metrics[2]],
    'F1 Score': [vgg11_metrics[3], large_metrics[3], very_large_metrics[3]]
})

print("\nKernel Size Evaluation Summary Table:")
print(size_evaluation_results)

# Save evaluation summary to display all results
evaluation_results = pd.DataFrame({
    'Model': ['VGG11 (Subset)', 'Shallow CNN', 'Deep CNN', 'Large Kernel CNN', 'Very Large Kernel CNN'],
    'Accuracy': [vgg11_metrics[0], shallow_metrics[0], deep_metrics[0], large_metrics[0], very_large_metrics[0]],
    'Precision': [vgg11_metrics[1], shallow_metrics[1], deep_metrics[1], large_metrics[1], very_large_metrics[1]],
    'Recall': [vgg11_metrics[2], shallow_metrics[2], deep_metrics[2], large_metrics[2], very_large_metrics[2]],
    'F1 Score': [vgg11_metrics[3], shallow_metrics[3], deep_metrics[3], large_metrics[3], very_large_metrics[3]]
})

print("\nAll Models Evaluation Summary Table:")
print(evaluation_results)

# Model comparison graph
results_df = pd.DataFrame(evaluation_results)
x = np.arange(len(results_df["Model"]))  # label locations
width = 0.2  # width of the bars

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - 1.5 * width, results_df["Accuracy"], width, label='Accuracy', color='skyblue')
rects2 = ax.bar(x - 0.5 * width, results_df["Precision"], width, label='Precision', color='orange')
rects3 = ax.bar(x + 0.5 * width, results_df["Recall"], width, label='Recall', color='green')
rects4 = ax.bar(x + 1.5 * width, results_df["F1 Score"], width, label='F1 Score', color='red')

# Adding labels, title, and customizing x-axis
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Metric Value', fontsize=12)
ax.set_title('CNN Model Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_df["Model"], rotation=30, ha="right", fontsize=10)
ax.legend()

# Finalizing the layout
fig.tight_layout()

# Display the graph
plt.show()
