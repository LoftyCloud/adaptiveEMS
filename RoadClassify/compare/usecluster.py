import joblib
# 加载聚类模型
import numpy as np
from sklearn.preprocessing import StandardScaler
kmeans_model = joblib.load('classification_cluster.pkl')
from DCload import dcload
data = dcload()
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

    average_speed = np.mean(data_split)
    speed_std = np.std(data_split)
    average_acceleration = np.mean(np.diff(data_split))
    maximum_speed = np.max(data_split)

    features = np.array([average_speed, average_acceleration, speed_std, maximum_speed])
    # 对特征进行标准化
    # scaler = StandardScaler()
    # features_normalized = scaler.fit_transform(features.reshape(-1, 1))

    labels_predicted = kmeans_model.predict([features])
    step += 20
    label.append(labels_predicted[0])
    # print(labels_predicted)
print(label)
