import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from data.data_augment import train_transformations, test_transformations
from data.dataset import CIFAR10C
from models.loss import NTXentLoss
from models.model import resnet18, simnet

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

def train(epoch):
   total_loss = 0.0
   tqdm_bar = tqdm(train_loader)
   for idx, (xi, xj, _) in enumerate(tqdm_bar):
      xi = xi.to(device)
      xj = xj.to(device)
      model.zero_grad()
      rep1 = model(xi)
      rep2 = model(xj)
      loss = loss_func(rep1, rep2)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      tqdm_bar.set_description('{} Epoch: [{}] Loss: {:.4f}'.format("Training", epoch, loss.item()))

def train_classifier(epoch):
   total_loss = 0.0
   model.eval()
   tqdm_bar = tqdm(trainc_loader)
   for idx, (img, label) in enumerate(tqdm_bar):
      classifier.zero_grad()
      img = img.to(device)
      label = label.to(device)
      feature = model(img)
      out = classifier(feature)
      loss = cross_entropy(out, label)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      tqdm_bar.set_description('{} Epoch: [{}] Loss: {:.4f}'.format("Training", epoch, loss.item()))
      
def test():
   total_loss = 0.0
   total_acc = 0.0
   model.eval()
   classifier.eval()
   tqdm_bar = tqdm(test_loader)
   with torch.no_grad():
      for idx, (img, label) in enumerate(tqdm_bar):
         classifier.zero_grad()
         img = img.to(device)
         label = label.to(device)
         feature = model(img)
         out = classifier(feature)
         loss = cross_entropy(out, label)
         pred = out.max(dim=1)[1]
         correct = pred.eq(label).sum().item()
         correct /= label.size(0)
         batch_acc = (correct * 100)
         total_acc += batch_acc
         total_loss += loss.item()
         tqdm_bar.set_description('{} Loss: {:.4f} Avg Acc: {:.4f}'.format("Testing", loss.item(), total_acc/(idx + 1)))
         
if __name__ == "__main__":
   train_dataset = CIFAR10C("./data", train=True, download=True, transform=train_transformations())
   test_dataset = datasets.CIFAR10("./data", train=False, download=False, transform=test_transformations())
   trainc_dataset = datasets.CIFAR10("./data", train=True, download=False, transform=test_transformations())
   loss_func = NTXentLoss()
   cross_entropy = nn.CrossEntropyLoss()
   model = resnet18().to(device)
   classifier = simnet().to(device)
   optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.05)

   train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
   trainc_loader = DataLoader(trainc_dataset, batch_size=512, shuffle=True)
   test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
   if os.path.exists('./model.pth'):
      model.load_state_dict(torch.load("./model.pth"))
   if os.path.exists('./classifier.pth'):
      classifier.load_state_dict(torch.load('./classifier.pth'))
   # train encoder
   for epoch in range(5):
      train(epoch)
   torch.save(model.state_dict(), './model.pth')
   # train classifier
   optimizer = optim.Adam(classifier.parameters(), lr=1, weight_decay=0.05)
   for epoch in range(5):
      train_classifier(epoch)
   torch.save(classifier.state_dict(), './classifier.pth')
   # test
   test()
   