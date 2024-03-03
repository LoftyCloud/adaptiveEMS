import numpy as np

from skfuzzy import control as ctrl
import random
## 数据准备过程
from sklearn.model_selection import train_test_split

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
# print(data.shape)
# print(labels.shape)
# 打乱数据和标签的顺序，以确保数据的随机性
permutation = np.random.permutation(len(data))
data = data[permutation]
labels = labels[permutation]
# 准备训练数据
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

# 计算速度序列的平均值和波动性
average_speed = np.mean(X_train, axis=1)
speed_fluctuation = np.std(X_train, axis=1)
# 创建输入和输出的模糊变量
feature1 = ctrl.Antecedent(np.arange(min(average_speed), max(average_speed), 0.1), 'average_speed')
feature2 = ctrl.Antecedent(np.arange(min(speed_fluctuation), max(speed_fluctuation), 0.1), 'speed_fluctuation')
output_class = ctrl.Consequent(np.arange(0, 2, 1), 'output_class')  # 修改输出范围为离散的两个类别[0,1]
# 定义模糊集合
feature1.automf(3)  # 自动划分三个模糊集合
feature2.automf(3)
output_class.automf(names=['class_0', 'class_1'])

# 规则1：速度平均值较低且波动较大，属于城市驾驶条件
rule1 = ctrl.Rule(feature1['poor'] & feature2['good'], output_class['class_1'])
rule2 = ctrl.Rule(feature1['average'] & feature2['good'], output_class['class_1'])
rule3 = ctrl.Rule(feature1['good'] & feature2['good'], output_class['class_0'])

rule4 = ctrl.Rule(feature1['good'] & feature2['poor'], output_class['class_0'])
rule5 = ctrl.Rule(feature1['average'] & feature2['poor'], output_class['class_0'])
rule6 = ctrl.Rule(feature1['poor'] & feature2['poor'], output_class['class_1'])

rule7 = ctrl.Rule(feature1['poor'] & feature2['average'], output_class['class_1'])
rule8 = ctrl.Rule(feature1['average'] & feature2['average'], output_class['class_0'])
rule9 = ctrl.Rule(feature1['good'] & feature2['average'], output_class['class_0'])

# 创建模糊系统
fuzzy_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
classification = ctrl.ControlSystemSimulation(fuzzy_system)
# 使用模糊系统进行分类
classified_labels = []
for sample in X_train:
    average_speed = np.mean(sample)
    speed_fluctuation = np.std(sample)
    classification.input['average_speed'] = average_speed  # 使用正确的输入变量名称
    classification.input['speed_fluctuation'] = speed_fluctuation
    classification.compute()
    classified_labels.append(classification.output['output_class'])

import pickle
# 保存 classification 模型
with open('classification_fuzzy1.pkl', 'wb') as file:
    pickle.dump(classification, file)

# 将输出映射到类别
classified_labels = np.array(classified_labels)
predicted_labels = np.where(classified_labels > 0.5, 1, 0)

# 计算准确度
accuracy = np.mean(predicted_labels == y_train)
print(f'Accuracy on test data: {accuracy:.4f}')