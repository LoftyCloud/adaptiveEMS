import numpy as np

from skfuzzy import control as ctrl
import random
## 数据准备过程
from sklearn.model_selection import train_test_split

def getData(file_path):
    # 打开txt文件并读取数据
    with open(file_path, 'r') as file:
        data = file.readlines()  # 读取文件的所有行
        # print(len(data))
        # print(data[0])

    numeric_values = []
    for line in data:
        # 假设每行数据是以空格分隔的数字，你可以进一步处理这些数据
        values = line.strip().split(' ')  # 以空格分隔数据并去除行末的换行符
        values = [float(value) for value in values]
        # 将每行数据添加到 numeric_values
        numeric_values.append(values)

        # print(len(numeric_values))
        # print(numeric_values[0])
    return numeric_values

random_seed = 42
# 读取CSV文件并加载数据
data1 = getData('../Data/suburban.txt')
data2 = getData('../Data/urban.txt')
print(len(data1))
print(len(data1[0]))

# 随机抽取1000个样本，设置标签并合并
sample_num = 1000
data1 = random.sample(data1, sample_num)
data2 = random.sample(data2, sample_num)
label1 = np.zeros(sample_num, dtype=int)
label2 = np.ones(sample_num, dtype=int)
# print(data1.shape)

data = np.concatenate([data1, data2])  # (2000,90)
labels = np.concatenate([label1, label2])  #(2000,)
# print(data.shape)
# print(labels.shape)

# 打乱数据和标签的顺序，以确保数据的随机性
permutation = np.random.permutation(len(data))
data = data[permutation]
labels = labels[permutation]

# 准备训练数据
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# 转换数据为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# 创建数据集和数据加载器
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# 创建模型、损失函数和优化器
input_size = X_train.shape[1]
model = NeuralNetwork(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()

# 评估模型
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_binary = torch.round(y_pred)
    test_accuracy = torch.sum(y_pred_binary.squeeze() == y_test_tensor).item() / len(y_test_tensor)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'neural_network_model.pth')