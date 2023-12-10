

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
import networks
import utils
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from networks import ConvNet
from utils import get_dataset
from torchprofile import profile_macs
import os
from networks import ConvNet
from utils import get_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image
from torchprofile import profile_macs
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import os
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label in range(10):  # 对于0到9的文件夹
            folder_path = os.path.join(self.root_dir, str(label))
            for img_file in os.listdir(folder_path):
                if img_file.endswith('.png'):  # 确保文件是PNG格式
                    self.images.append(os.path.join(folder_path, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('L')  
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  
])




channel, im_size, num_classes, _, _, _, train_dataset_1, test_dataset, _ = get_dataset('MNIST', './data')
# change dataset dir
train_dataset = CustomDataset(root_dir='./result/synthesized_images', transform=transform)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=im_size)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
scheduler = CosineAnnealingLR(optimizer, T_max=20)

test_accuracies = []

# 在训练循环外定义测试函数
def test_model(net, test_loader, criterion):
    net.eval()  # 将模型设置为评估模式
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 在评估过程中不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / total
    print(f'Test Loss: {test_loss:.3f}, Accuracy: {test_accuracy:.2f}%')
    
    return test_accuracy

# 训练模型
for epoch in range(60):
    net.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    scheduler.step()
    # 在每个训练周期结束时进行测试
    print(f'Epoch {epoch+1} - Train Loss: {train_loss:.3f}')
    test_model(net, test_loader, criterion)
    test_accuracy = test_model(net, test_loader, criterion)
    test_accuracies.append(test_accuracy) 

flops = profile_macs(net, data[0].unsqueeze(0))  # 使用一个batch中的第一个图像来计算
print(f"FLOPs after epoch {epoch}: {flops}")


# 绘制并保存测试准确度图像
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.savefig('test_accuracy_over_epochs.png')  # 保存图像
plt.show()  # 显示图像
