import numpy as np
import random
## 数据准备过程
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

data = np.concatenate([data1, data2])  # (2000,90)
labels = np.concatenate([label1, label2])  #(2000,)

# 打乱数据和标签的顺序，以确保数据的随机性
permutation = np.random.permutation(len(data))
data = data[permutation]
labels = labels[permutation]
# 准备训练数据
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

# 计算平均速度
average_speed = np.mean(X_train, axis=1, keepdims=True)
# 计算平均加速度
average_acceleration = np.mean(np.diff(X_train, axis=1), axis=1, keepdims=True)
# 计算波动大小
speed_std = np.std(X_train, axis=1, keepdims=True)
# 计算最大速度
maximum_speed = np.max(X_train, axis=1, keepdims=True)
# 将特征合并成一个特征矩阵
features = np.concatenate([average_speed, average_acceleration, speed_std, maximum_speed], axis=1)
# # 对特征进行标准化
# scaler = StandardScaler()
# features_normalized = scaler.fit_transform(features)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# 使用K均值聚类
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(features)
kmeans_labels = kmeans.labels_
# print(kmeans_labels.shape)
# print(y_test.shape)
test_accuracy = accuracy_score(y_train, kmeans_labels)
print(f'Accuracy on train data: {test_accuracy:.4f}')
# # 根据聚类结果生成预测标签
# predicted_labels = np.zeros_like(cluster_labels)
# predicted_labels[cluster_labels == 0] = 1
#
# # 计算准确度
# accuracy = accuracy_score(y_train, predicted_labels)
# print(f'Accuracy on training data: {accuracy:.4f}')
#
# # 在测试数据上进行预测
# test_cluster_labels = kmeans.predict(X_test)
# test_predicted_labels = np.zeros_like(test_cluster_labels)
# test_predicted_labels[test_cluster_labels == 0] = 1
#
# test_accuracy = accuracy_score(y_test, test_predicted_labels)
# print(f'Accuracy on test data: {test_accuracy:.4f}')

# 保存聚类模型
import joblib
joblib.dump(kmeans, 'classification_cluster.pkl')
