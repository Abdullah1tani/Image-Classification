import torch
import torchvision.models as models
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from features_utils import select_subset, extract_features

# Define a transformation to apply to each image in the CIFAR-10 dataset
# Here, we only convert the images to tensors for now
transform = transforms.Compose([
    transforms.ToTensor()  # Convert images to PyTorch tensors
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

# Display the sizes of the training and testing subsets just to confirm correct loading
print(f"Selected training subset size: {len(train_subset)}")  #(500 samples * 10 classes)
print(f"Selected testing subset size: {len(test_subset)}")    #(100 samples * 10 classes)
