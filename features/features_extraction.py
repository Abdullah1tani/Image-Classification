import torch
import torchvision.models as models
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from features_utils import select_subset, extract_features

# Define transformation for resizing to 224x224, normalizing for ResNet-18
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet-18
])

# Load the CIFAR-10 training dataset with the specified transformation
# `root='./data'` specifies where to store the data, `train=True` loads the training set
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Load the CIFAR-10 test dataset similarly, but with `train=False` to get the test data
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create the training subset with the first 500 samples for each class
train_subset = select_subset(train_data, 500)

# Create the testing subset with the first 100 samples for each class
test_subset = select_subset(test_data, 100)

# Define DataLoader for batch processing
train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

# Load pre-trained ResNet-18 model using the new weights parameter
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Remove the last fully connected layer to get feature extraction model
feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
feature_extractor.eval()  # Set to evaluation mode to avoid training behavior

# Extract features for training subset
print("Begin feature extraction for training subset:")
train_features, train_labels = extract_features(train_loader, feature_extractor)

# Extract features for testing subset
print("\nBegin feature extraction for testing subset:")
test_features, test_labels = extract_features(test_loader, feature_extractor)

# Check the initial dimensionality to ensure it's 512
assert train_features.shape[1] == 512, "Training features should be 512-dimensional."
assert test_features.shape[1] == 512, "Test features should be 512-dimensional."

# Apply PCA to reduce to 50 dimensions
pca = PCA(n_components=50)                  # Initialize PCA to keep 50 principal components
train_features_50d = pca.fit_transform(train_features)  # Fit PCA on training data and transform
test_features_50d = pca.transform(test_features)        # Apply the same transformation to test data

# Save the 50-dimensional features for training and testing to a new file
np.savez('cifar10_features_50d.npz', train_features=train_features_50d, train_labels=train_labels,
         test_features=test_features_50d, test_labels=test_labels)

print("PCA applied successfully. Features reduced to 50 dimensions and saved to 'cifar10_features_50d.npz'.")
