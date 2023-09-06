import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim.lr_scheduler as sc
import numpy as np
from tqdm import tqdm

class TestModel(nn.Module):
    def __init__(self, in_channels, num_classes):
      super(TestModel, self).__init__()
      self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
      self.bn1 = nn.BatchNorm2d(64)
      self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
      self.bn2 = nn.BatchNorm2d(128)
      self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
      self.bn3 = nn.BatchNorm2d(256)
      self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
      self.bn4 = nn.BatchNorm2d(512)
      self.Block1 = nn.Sequential(
          nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(),
          nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(64)
      )
      self.Block2 = nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(),
          nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(128)
      )
      self.Block3 = nn.Sequential(
          nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(),
          nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(256)
      )
      self.Block4 = nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(),
          nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(512)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(512*2*2, 512*2),
          nn.BatchNorm1d(512*2),
          nn.Linear(512*2, 512),
          nn.BatchNorm1d(512),
          nn.Linear(512, 256),
          nn.BatchNorm1d(256),
          nn.Linear(256, num_classes)
      )
    def forward(self, x):
      x = F.relu(self.bn1(self.conv1(x)))
      x = F.relu(self.Block1(x) + x)
      x = F.relu(self.bn2(self.conv2(x)))
      x = F.relu(self.Block2(x) + x)
      x = F.relu(self.bn3(self.conv3(x)))
      x = F.relu(self.Block3(x) + x)
      x = F.relu(self.bn4(self.conv4(x)))
      x = F.relu(self.Block4(x) + x)
      x = self.classifier(x)
      return x

batch_size = 128 # 16
epochs = 50
learning_rate = 0.1

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

train = torchvision.datasets.CIFAR100(root="./", train=True, download=True, transform=train_transform)
test = torchvision.datasets.CIFAR100(root="./", train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, 
                                           shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

model = TestModel(3, 100).cuda()

optimizer =  optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

scheduler = sc.StepLR(optimizer, step_size=20, gamma=0.1)

for epoch in range(epochs):
    model.train()
    train_correct, train_all_data = 0,0
    for img, label in tqdm(train_loader):
        img = img.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_correct += torch.sum(torch.argmax(output, dim=1) == label).item()
        train_all_data += len(label)
    scheduler.step()
    correct, all_data = 0,0
    model.eval()
    for img, label in test_loader:
        with torch.no_grad():
            img = img.cuda()
            label = label.cuda()
            output = model(img)

            correct += torch.sum(torch.argmax(output, dim=1) == label).item()
            all_data += len(label)

x = torch.randn(batch_size, 3, 32, 32).cuda()

torch.onnx.export(model,
                  x,
                  "simple_model.onnx",
                  verbose = True,
                  input_names=['input'],
                  output_names=['output'])