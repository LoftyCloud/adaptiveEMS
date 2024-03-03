from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.utils import plot_model

# 创建一个简单的LSTM模型
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(10, 1)))
model.add(Dropout(0.1))
model.add(LSTM(units=30, activation='relu', return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# 绘制模型结构图并保存为文件
plot_model(model, to_file='lstm_network.png', show_shapes=True, show_layer_names=True)
