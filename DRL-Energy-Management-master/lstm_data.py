import pickle

# 读取history.pkl文件
with open('history.pkl', 'rb') as file:
    history = pickle.load(file)
import matplotlib.pyplot as plt
print(history)
# 提取训练和验证准确度
train_accuracy = history['accuracy']
val_accuracy = history['val_accuracy']

# 绘制准确度变化图
epochs = range(1, len(train_accuracy) + 1)

# print(train_accuracy)
# print(val_accuracy)

plt.plot(epochs, train_accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
# plt.title('Training and validation accuracy')
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
