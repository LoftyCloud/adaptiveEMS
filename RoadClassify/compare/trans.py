import locale
import pickle

import numpy as np
# 设置小数点格式为点号
from keras.src.callbacks import History
from matplotlib import pyplot, pyplot as plt
import random

locale.setlocale(locale.LC_ALL, 'C')


## 数据准备过程
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

data = np.concatenate([data1, data2])
labels = np.concatenate([label1, label2])
print(data.shape)
print(labels.shape)

# 打乱数据和标签的顺序，以确保数据的随机性
permutation = np.random.permutation(len(data))
data = data[permutation]
labels = labels[permutation]

from sklearn.preprocessing import StandardScaler
# 对数据进行标准化
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)
for i in range(10):
    pyplot.plot(data_normalized[i], label=labels[i])
pyplot.legend()
pyplot.xlabel("time")
pyplot.ylabel("data")
pyplot.show()


from sklearn.model_selection import train_test_split

# 准备训练数据
X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.1, random_state=42)
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 读取BERT模型和tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# 数据预处理
def preprocess_data(data):
    # 将数据转换为BERT输入格式
    inputs = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
    return inputs

# 转换数据
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# 创建自定义的Transformer模型
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        # 使用BERT模型的隐藏状态作为特征
        self.bert = bert_model
        # 添加一个线性层进行二元分类
        self.fc = nn.Linear(bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # 获取BERT模型的隐藏状态
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # 取BERT模型的CLS token的输出作为特征
        cls_output = outputs.last_hidden_state[:, 0, :]
        # 使用线性层进行分类
        output = self.fc(cls_output)
        return output

# 创建模型实例
transformer_model = TransformerModel()

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=1e-5)
criterion = nn.BCEWithLogitsLoss()

# 训练模型
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer_model.to(device)

for epoch in range(num_epochs):
    transformer_model.train()
    optimizer.zero_grad()
    inputs = {key: value.to(device) for key, value in X_train.items()}
    outputs = transformer_model(**inputs)
    loss = criterion(outputs.squeeze(), torch.Tensor(y_train).to(device))
    loss.backward()
    optimizer.step()

# 测试模型
transformer_model.eval()
with torch.no_grad():
    inputs = {key: value.to(device) for key, value in X_test.items()}
    outputs = transformer_model(**inputs)
    predicted_labels = torch.round(torch.sigmoid(outputs)).cpu().numpy()

# 计算准确度
accuracy = accuracy_score(y_test, predicted_labels)
print(f'Accuracy on test data: {accuracy:.2f}')

