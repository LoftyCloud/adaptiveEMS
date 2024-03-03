import locale
import pickle
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

# print(labels[0],data[0,:])
## 模型训练过程
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 准备训练数据
X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.1, random_state=42)
# 创建一个Sequential模型
model = Sequential()
from keras.initializers import glorot_normal
# 添加一个LSTM层
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(len(data[0]), 1), kernel_initializer=glorot_normal()))
model.add(Dropout(0.1))
model.add(LSTM(units=30, activation='relu', return_sequences=False, kernel_initializer=glorot_normal()))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
# 编译模型
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001, clipvalue=1.0), metrics=['accuracy'])
# 训练模型
hist = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))
model.save('Data/lstm_model.h5')
# 保存history对象到文件
with open('../Data/history.pkl', 'wb') as file:
    pickle.dump(hist.history, file)
# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy on test data: {accuracy:.2f}%')

plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

