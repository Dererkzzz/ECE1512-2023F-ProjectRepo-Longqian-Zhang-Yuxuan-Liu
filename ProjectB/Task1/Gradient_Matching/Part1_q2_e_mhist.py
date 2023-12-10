
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchprofile import profile_macs
from networks import ConvNet
from utils import get_dataset
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MHIST 数据集类
class MHISTDataset(Dataset):
    def __init__(self, csv_file, root_dir, partition, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.partition = partition
        self.transform = transform
        # 仅保留所需分区的数据
        self.annotations = self.annotations[self.annotations['Partition'] == self.partition]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.annotations.iloc[idx, 1]

        # 将标签转换为数值
        label_to_idx = {'SSA': 0, 'HP': 1}  # 示例，根据实际情况调整
        label = label_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)


class MHISTDataset_train(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 读取HP类别的图片
        hp_path = os.path.join(self.root_dir, 'HP')
        for i in range(49):  
            image_path = os.path.join(hp_path, f"image_{i+50}.png")
            self.image_paths.append(image_path)
            self.labels.append('HP')

        # 读取ssa类别的图片
        ssa_path = os.path.join(self.root_dir, 'ssa')
        for i in range(49): 
            image_path = os.path.join(ssa_path, f"image_{i}.png")
            self.image_paths.append(image_path)
            self.labels.append('ssa')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name)
        label = self.labels[idx]

        # 将标签转换为数值
        label_to_idx = {'ssa': 0, 'HP': 1}
        label = label_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载训练和测试数据集
#train_dataset = MHISTDataset(csv_file='./data/mhist_dataset/annotations.csv', root_dir='./data/mhist_dataset/images/images', partition='train', transform=transform)
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

train_dataset = MHISTDataset_train(root_dir='./result/MHIST_result_noise_conv6_it200', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


test_dataset = MHISTDataset(csv_file='./data/mhist_dataset/annotations.csv', root_dir='./data/mhist_dataset/images/images', partition='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 初始化模型、损失函数和优化器
model = ConvNet(channel=3, num_classes=2, net_width=128, net_depth=6, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=(224, 224))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = CosineAnnealingLR(optimizer, T_max=20)


from sklearn.metrics import f1_score, precision_score, recall_score
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    return precision, recall, f1

f1_scores = []


# Training and Evaluation Process
num_epochs = 60  # Set the number of epochs
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
    	
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    scheduler.step()
    flops = profile_macs(model, images[0].unsqueeze(0).to(device))  # FLOPs calculation
    
    print(f"FLOPs after epoch {epoch+1}: {flops}")

    # Evaluate model after each epoch
    precision, recall, f1 = evaluate_model(model, test_loader)
    print(f"Epoch {epoch+1} -  F1 Score: {f1}")
    f1_scores.append(f1)
    
    
    # 在最后一个epoch结束时保存F1分数图表
    if epoch == num_epochs - 1:
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(1, num_epochs+1), f1_scores, marker='o', color='b', label='F1 Score')
        plt.title('F1 Score per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.xticks(np.arange(1, num_epochs+1))
        plt.legend()
        plt.grid(True)
        plt.savefig('f1_scores_per_epoch.png')
        plt.show()






