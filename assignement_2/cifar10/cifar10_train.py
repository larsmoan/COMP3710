import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import ResNet18
from torch.utils.data import DataLoader
import wandb

# ---------- Device configuration ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# ---------- Logging WANDB ----------
wandb.init(project="cluster_CIFAR10", name="initial_testing")
wandb.config.update({"architecture": "ResNet18_no_downsample", "dataset": "CIFAR-10", "epochs": 5, 
                     "batch_size": 128, "weight_decay": 5e-4, "max_lr": 0.1, "grad_clip": 1.5})


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.465), (0.2023, 0.01994, 0.2010)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32,padding=4, padding_mode='reflect')
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, wandb.config.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, wandb.config.batch_size*2, shuffle=False, num_workers=2)
                      


# ---------- Model ----------
model = ResNet18()
model.to(device)


#----------- Training --------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  #Swap out this one for higher acc

total_steps = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=wandb.config.max_lr, epochs=wandb.config.epochs, total_steps=total_steps)


model.train()
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
      print("Epoch [{}/{}], Step [{}/{}] Loss {:.5f}".format(epoch+1, num_epochs, i+1, total_steps, loss.item()))
    scheduler.step()



#----------- Testing --------------
model.eval()
with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

  print("Accuracy of the model on the test images: {}%".format(100*correct/total))
