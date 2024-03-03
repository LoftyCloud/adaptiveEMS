import time

# 定义神经网络模型
import numpy as np
import torch
from torch import nn

import joblib
from DCload import dcload
data = dcload()

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

# 加载模型
loaded_model = NeuralNetwork(90)
loaded_model.load_state_dict(torch.load('neural_network_model.pth'))
loaded_model.eval()
label = []
step = 0
while step < (len(data) - 2):
    ##########################驾驶条件识别###########################################
    data_split = []
    if step < 45:
        data_split = data[0:90]
    elif step >= len(data) - 45:
        data_split = data[len(data) - 90:len(data)]
    else:
        data_split = data[step - 45:step + 45]
    sample_data_tensor = torch.FloatTensor(data_split).view(1, -1)
    prediction = loaded_model(sample_data_tensor)

    step += 20
    label.append(1 if prediction.item()>0.5 else 0)
    # print(prediction.item())
print(label)