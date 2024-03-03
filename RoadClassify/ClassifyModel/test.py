import locale

import numpy as np
# 设置小数点格式为点号
from matplotlib import pyplot, pyplot as plt
import random
locale.setlocale(locale.LC_ALL, 'C')
## 数据准备过程
def getData(file_path):
    # 打开txt文件并读取数据
    with open(file_path, 'r') as file:
        data = file.readlines()  # 读取文件的所有行

    numeric_values = []
    for line in data:
        values = line.strip().split(' ')  # 以空格分隔数据并去除行末的换行符
        values = [float(value) for value in values]
        # 将每行数据添加到 numeric_values
        numeric_values.append(values)
    return numeric_values


random_seed = 42
# 读取CSV文件并加载数据
data1 = getData('../Data/suburban.txt')
data2 = getData('../Data/urban.txt')
print(len(data1))
print(len(data1[0]))

# 随机抽取1000个样本，设置标签并合并
sample_num = 5
data1 = random.sample(data1, sample_num)
data2 = random.sample(data2, sample_num)
label1 = np.zeros(sample_num, dtype=int)
label2 = np.ones(sample_num, dtype=int)

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

# 模型加载
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

from keras.initializers import glorot_normal
# 创建一个Sequential模型
model = Sequential()
# 添加一个LSTM层
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(len(data[0]), 1), kernel_initializer=glorot_normal()))
model.add(Dropout(0.1))
model.add(LSTM(units=30, activation='relu', return_sequences=False, kernel_initializer=glorot_normal()))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# 加载预训练权重
model.load_weights('../Data/lstm_model.h5')
X_val = data_normalized.reshape((data_normalized.shape[0], data_normalized.shape[1], 1))
yVal = model.predict(X_val)
print(X_val.shape)
for l in yVal:
    print(0 if l<0.5 else 1)

#
# pyplot.figure(figsize=(7, 4))
#
# for i in range(10):
#     label = f"label:{labels[i]}, pre:{yVal[i][0]:.2f}"
#     pyplot.plot(X_val[i], label=label)
#
# for i in range(len(X_val)):
#     print('[', end='')
#     for j in range(len(X_val[0])):
#         print(X_val[i][j][0],end=' ')
#     print(']')
# print(labels)
# for i in range(10):
#     print(yVal[i][0],end=' ')
#
#
# # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# pyplot.legend()
# pyplot.show()
# # print(yVal)
# # print(labels)
