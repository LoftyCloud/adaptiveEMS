import numpy as np
import pandas as pd
# 指定CSV文件的路径
from keras import Sequential
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler

filepath = 'E:/documents/DRLResult/data/wltp.csv'
# filepath = 'E:/documents/DRLResult2/data/road.csv'

# 使用pandas的read_csv函数读取CSV文件
data = pd.read_csv(filepath)
# 访问表中的'cycmps'列数据
cycmps_data = np.array(data['cycMps'])
# print(cycmps_data)

# 数据处理，将驾驶循环拆分成验证数据格式
data_split = []
step_len = 5
for i in range(1, len(cycmps_data), step_len):
    if i <= 45:
        data = cycmps_data[i:i + 90]
    elif i >= len(cycmps_data)-45:
        data = cycmps_data[len(cycmps_data) - 90:len(cycmps_data)]
    else:
        data = cycmps_data[i - 45:i + 45]
    # print(f'i = {i}')
    data_split.append(data)
# print(len(data_split))
# for i in range(len(data_split)):
#     print(len(data_split[i]))
# print(data_split[0])

# 对数据进行标准化
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_split)

# 加载模型
from keras.initializers import glorot_normal
from keras.layers import Dense, Dropout, LSTM
# 创建模型
model = Sequential()
model.add(
    LSTM(units=50, activation='relu', return_sequences=True, input_shape=(90, 1), kernel_initializer=glorot_normal()))
model.add(Dropout(0.1))
model.add(LSTM(units=30, activation='relu', return_sequences=False, kernel_initializer=glorot_normal()))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
# 加载预训练权重
model.load_weights('../Data/lstm_model.h5')

print(data_normalized.shape)
X_val = data_normalized.reshape((data_normalized.shape[0], data_normalized.shape[1], 1))
print(X_val.shape)
yVal = model.predict(X_val)
# print(yVal)
pyplot.figure(figsize=(5, 3))
pyplot.plot([], [], color='blue', label='suburban')
pyplot.plot([], [], color='red', label='urban')

# pyplot.plot(cycmps_data,color = 'blue')
step = 10
for i in range(len(X_val)):
    if yVal[i][0] >= 0.5:
        pyplot.plot(range(max(0,i*step_len-step),min(len(cycmps_data),i*step_len+step)),cycmps_data[max(0,i*step_len-step): min(len(cycmps_data),i*step_len+step)], color='red')
    else:
        pyplot.plot(range(max(0,i*step_len-step), min(len(cycmps_data),i*step_len+step)),cycmps_data[max(0,i*step_len-step): min(len(cycmps_data),i*step_len+step)], color='blue')
print(yVal)
# print(yVal)
# for i in range(len(X_val)):
#     if yVal[i][0] < 0.5:
#         pyplot.plot(list(range(0 + i * step_len, i * step_len + 90)), data_split[i], color='blue')
#         pass
#     # else:
#     #     pyplot.plot(list(range(0 + i * step_len, i * step_len + 90)), data_split[i], color='red')
#     #     pass
# for i in range(len(X_val)):
#     if yVal[i][0] >= 0.5:
#         pyplot.plot(list(range(0 + i * step_len, i * step_len + 90)), data_split[i], color='red')
pyplot.legend()
pyplot.grid(True)
pyplot.show()

# for i in range(1,len(cycmps_data),10):
#     # 对数据进行标准化
#     scaler = StandardScaler()
#     data = scaler.fit_transform(cycmps_data[i: i + 89])
#     data = data.reshape(data.shape[0], data.shape[1], 1)
#     label = model.predict(data,verbose=0)
#     # print(label)
#     print(label[-1])
