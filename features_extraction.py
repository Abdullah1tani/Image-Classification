import torchvision.models as models
import numpy as np
import os
import torch

from torch.utils.data import DataLoader
from features_utils import select_subset, extract_features
from torchvision import datasets, transforms

# Define transformation for resizing and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the CIFAR-10 datasets with the specified transform
train_data = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

# Select subsets: 500 training samples and 100 test samples per class
train_subset = select_subset(train_data, 500)
test_subset = select_subset(test_data, 100)

# Display the sizes of the training and testing subsets
print(f"Selected training subset size: {len(train_subset)}")
print(f"Selected testing subset size: {len(test_subset)}")

# Define DataLoader for batch processing
train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

# Just to confirm correct loading
for images, labels in train_loader:
    print(f"Training batch - images shape: {images.shape}, labels shape: {labels.shape}")
    break

for images, labels in test_loader:
    print(f"Testing batch - images shape: {images.shape}, labels shape: {labels.shape}")
    break

# Load the pre-trained ResNet-18 model
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
feature_extractor.eval()

# Extract features for training subset
print("Begin feature extraction for training subset:")
train_features, train_labels = extract_features(train_loader, feature_extractor)

# Extract features for testing subset
print("\nBegin feature extraction for testing subset:")
test_features, test_labels = extract_features(test_loader, feature_extractor)

# Save the extracted features and labels
np.savez('cifar10_features.npz', train_features=train_features, train_labels=train_labels,
         test_features=test_features, test_labels=test_labels)

# verify the shape of extracted features
print("Feature extraction complete. Saved features to 'cifar10_features.npz'.")
print(f"Train subset: samples: {train_features.shape[0]}, features: {train_features.shape[1]}")
print(f"Test subset: samples: {test_features.shape[0]}, features: {test_features.shape[1]}")

# Check if the file was saved correctly
print("File exists:", os.path.exists('cifar10_features.npz'))

