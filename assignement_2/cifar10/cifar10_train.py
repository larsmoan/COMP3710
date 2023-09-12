import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import ResNet18, ResNet34
from torch.utils.data import DataLoader
import wandb

# --------- Hyperparameters ----------
wandb.init(project="cluster_CIFAR10", name="RESNET_18")
wandb.config.update({"architecture": "cifar10model", "dataset": "CIFAR-10", "epochs": 1, 
                     "batch_size": 128, "weight_decay": 5e-4, "max_lr": 0.1, "grad_clip": 1.5})


# ---------- Device configuration ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


train_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.465), (0.2023, 0.01994, 0.2010)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32,padding=4, padding_mode='reflect')
])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.465), (0.2023, 0.01994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=wandb.config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=wandb.config.batch_size*2, shuffle=False)
                      


# ---------- Model ----------
model = ResNet18()
model.to(device)


#----------- Training --------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.max_lr, momentum=0.9, weight_decay=wandb.config.weight_decay)  #Swap out this one for higher acc


scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=wandb.config.max_lr, epochs=wandb.config.epochs, steps_per_epoch=len(train_loader))



best_acc = 0
for epoch in range(wandb.config.epochs):
  model.train()
  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (i+1) % 100 == 0:
      learning_rate = optimizer.param_groups[0]['lr']
      wandb.log({"epoch": epoch, "learning_rate": learning_rate, "loss": loss.item()})  
  

  #Testing each epoch
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
      acc = 100 * correct / total
      if acc > best_acc and acc > 90:
        best_acc = acc
        torch.save(model.state_dict(), 'cifar10_resnet18.pth')
    wandb.log({"accuracy": 100*correct/total})


