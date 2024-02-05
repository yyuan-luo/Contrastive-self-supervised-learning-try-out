import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from data.data_augment import train_transformations, test_transformations
from data.dataset import CIFAR10C
from models.loss import NTXentLoss
from models.model import resnet18
from tqdm import tqdm

train_dataset = CIFAR10C("./data", train=True, download=True, transform=train_transformations())
loss_func = NTXentLoss()
model = resnet18()
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.05)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


def train(epoch):
   total_loss = 0.0
   tqdm_bar = tqdm(train_loader)
   for idx, (xi, xj, _) in enumerate(tqdm_bar):
      model.zero_grad()
      rep1 = model(xi)
      rep2 = model(xj)
      loss = loss_func(rep1, rep2)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      tqdm_bar.set_description('{} Epoch: [{}] Loss: {:.4f}'.format("Training", epoch, loss.item()))
         
         
if __name__ == "__main__":
   for epoch in range(5):
      train(epoch)