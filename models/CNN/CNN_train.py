import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from features.features_utils import select_subset
from CNN_implementation import CNN, LargeCNN, ShallowCNN, DeepCNN, cnn_train

# define transform
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally to better train the model
    transforms.RandomCrop(32, padding=4),  # Add padding and crop to better train the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the CIFAR-10 datasets with the specified transform
train_data = datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)

# Select subsets: first 500 training samples per class
train_subset = select_subset(train_data, 500)

# Define DataLoader for batch processing
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)

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

# train all models
cnn_train(cnn, train_loader, criterion, device)

cnn_train(shallow_cnn, train_loader, criterion, device)
cnn_train(deep_cnn, train_loader, criterion, device)

cnn_train(large_cnn, train_loader, criterion, device)
cnn_train(very_large_cnn, train_loader, criterion, device)