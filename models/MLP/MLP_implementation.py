import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Load CIFAR-10 feature vectors (from PCA-reduced file with 50 dimensions)
data = np.load('../../features/cifar10_features_50d.npz')
train_features, train_labels = data['train_features'], data['train_labels']
test_features, test_labels = data['test_features'], data['test_labels']

# Convert data to PyTorch tensors
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# Define DataLoader for batching
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(50, 512)        # Input layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)       # Hidden layer
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 10)        # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn2(self.fc2(x))
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Initialize the MLP model, loss function, and optimizer
mlp_model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)

# Training loop
epochs = 10  # Number of training epochs
for epoch in range(epochs):
    mlp_model.train()
    total_loss = 0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = mlp_model(features)       # compute predictions
        loss = criterion(outputs, labels)   # measure difference between predictions and true labels
        loss.backward()
        optimizer.step()         # update weights
        total_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Save the trained model
torch.save(mlp_model.state_dict(), 'mlp_model.pth')
print("MLP model training complete and saved.")

#Experimenting by varying the depth of the network

class MLP_Deeper(nn.Module):
    def __init__(self):
        super(MLP_Deeper, self).__init__()
        self.fc1 = nn.Linear(50, 512)     #example of variation made: self.fc1 = nn.Linear(50, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512,512)    #example of variation made: self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 10)   # Additional hidden layer  /  #example of variation made: self.fc3 = nn.Linear(256, 10)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 10)    # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn2(self.fc2(x))
        x = self.relu2(x)
        x = self.bn3(self.fc3(x))
        x = self.relu3(x)
        x = self.fc4(x)
        return x