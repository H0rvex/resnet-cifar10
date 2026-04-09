"""ResNet for CIFAR-10 - 84.6% accuracy
Simplified ResNet with skip connections, stride downsampling,
batch normalization, and data augmentation."""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda")

# transform compose
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# train transform compose
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # data augmentation
    transforms.RandomCrop(32, padding=4), # shifted by 4 pixels
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# class
class ResidualBlock(nn.Module):
    """Residual block with 2 conv layers and a skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = None
        if in_channels != out_channels or stride != 1:
            # only changes number of channels
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
                )
                
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.layer1(x)))
        x = self.bn2(self.layer2(x))
        if self.skip:
            residual = self.skip(residual)
        x += residual
        x = torch.relu(x)
        return x

class ResNet(nn.Module):
    """Main ResNet architecture with 6 blocks,
    an initial conv layer, linear layer"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.block1 = ResidualBlock(16, 16)
        self.block2 = ResidualBlock(16, 16)
        self.block3 = ResidualBlock(16, 32, stride=2) # channel change
        self.block4 = ResidualBlock(32, 32)
        self.block5 = ResidualBlock(32, 64, stride=2) # channel change
        self.block6 = ResidualBlock(64, 64)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    # dataset
    train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # data loading
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=128, num_workers=0)

    # instance
    model = ResNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    # learning rate scheduling
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # training loop
    for epoch in range(30):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # forward
            predictions = model(images)
            # loss
            loss = loss_fn(predictions, labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            # update
            optimizer.step()
        
        scheduler.step()

        print(f"Epoch {epoch}, loss = {loss.item():.4f}")
    
    
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            predicted = predictions.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print(f"Accuracy: {correct / total * 100:.1f}%")









    