import locale
# 设置小数点格式为点号
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
# 模型加载
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.initializers import glorot_normal
# 创建一个Sequential模型
model = Sequential()
# 添加一个LSTM层
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(90, 1), kernel_initializer=glorot_normal()))
model.add(Dropout(0.1))
model.add(LSTM(units=30, activation='relu', return_sequences=False, kernel_initializer=glorot_normal()))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
# 加载预训练权重
model.load_weights('lstm_model.h5')

from DCload import dcload
data = dcload()

label = []
step = 10
print(data.shape)
# plt.plot(data)
# plt.show()
data_splits = []
while step < (len(data) - 2):
    ##########################驾驶条件识别###########################################
    if step < 45:
        data_split = data[0:90]
    elif step >= len(data) - 45:
        data_split = data[len(data) - 90:len(data)]
    else:
        data_split = data[step - 45:step + 45]
    # 对数据进行标准化
    data_splits.append(data_split)
    step += 20

scaler = StandardScaler()
# data_normalized = scaler.fit_transform(data_split.reshape(-1, 1))
data_normalized = scaler.fit_transform(data_splits)
X_val = data_normalized.reshape((data_normalized.shape[0], data_normalized.shape[1], 1))
#
# X_val = data_normalized[np.newaxis, :]

print(X_val.shape)  # (1,90,1)
# plt.plot(X_val[0])
yVal = model.predict(X_val)

# a = np.mean(yVal)
# print(a)
# print(yVal)
label = []
for l in yVal:
    label.append(1 if l > 0.5 else 0)
print(label)
    #
    # scaler = StandardScaler()
    # # print(np.array(data_split).shape)  #(90,)
    # data_normalized = scaler.fit_transform(np.array(data_split).reshape(-1, 1))
    #
    # # print(data_normalized.shape)
    # # X_val = data_normalized[np.newaxis, :]
    # yVal = model.predict(data_normalized.reshape(1, 90, 1))


    # print(yVal)
    # label.append(1 if yVal[0][0] > 0.5 else 0)
    # if step>200:
    #     plt.show()
    #     break
    # print(prediction.item())

# print(label)
