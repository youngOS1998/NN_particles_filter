import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np

class Model(nn.Module):
    
    def __init__(self, num_classes=10, init_weights=False):
        super(Model, self).__init__()
        # define your neural network here

        self.net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        
        return self.net(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):   # 若是卷积层
                nn.init.normal_(m.weight, mean=0, std=1)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
  

use_cuda = True
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize([224, 224]),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=10, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=10, shuffle=True, **kwargs)
#         return train_loader, test_loader

import torch.optim as optim

net = Model(init_weights=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002)

save_path = './AlexNet.pth'
beat_acc = 0.0

i = 0

loss_list = []

for epoch in range(10):
    net.train()
    running_loss = 0.0
    
    for step, data in enumerate(train_loader):

        images, labels = data
        optimizer.zero_grad()   # 清除历史梯度
        
        outputs = net.forward(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()        # 优化器更新参数

        if i % 10 == 0:
            loss_list.append(loss)
        
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        i = i + 1

torch.save(net.state_dict(), 'parameter.pkl')

loss_save = np.array(loss_list)
np.save('./loss_save.npy', loss_save)