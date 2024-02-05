import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from data.data_augment import train_transformations, test_transformations
from data.dataset import CIFAR10C
from models.loss import NTXentLoss
from models.model import resnet18

train_dataset = CIFAR10C("./data", train=True, download=True, transform=train_transformations())
loss_func = NTXentLoss()
model = resnet18()
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.05)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


for epoch in range(5):
   for idx, (xi, xj, _) in enumerate(train_loader):
      print(idx, xi, xj)