from shakes_resnet18 import ResNet18
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.465), (0.2023, 0.01994, 0.2010)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32,padding=4, padding_mode='reflect')
])

#Note: Chakes creates two distinct transforms for training and testing, but use the same
# values for mean and std. Seems waste

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False)




model = ResNet18()
print(model)

model.to(device)


#----------- Training --------------
learning_rate = 0.1
num_epochs = 5



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  #Swap out this one for higher acc

total_steps = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)


model.train()
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    print(images.shape, i)
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