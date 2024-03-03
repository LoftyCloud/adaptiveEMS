import pickle
import numpy as np

from DCload import dcload

data = dcload()


def predict_driving_condition(classification, sample):
    # 计算输出
    classification.input['average_speed'] = sample[0]
    classification.input['speed_fluctuation'] = sample[1]
    classification.compute()
    # 获取输出
    predicted_class = classification.output['output_class']
    # 计算单步运行时间
    # 将输出映射到类别

    if predicted_class > 0.5:
        return 1  # 城市驾驶条件
    else:
        return 0  # 郊区驾驶条件,可以根据实际情况调整阈值


# 加载 classification 模型
with open('classification_fuzzy1.pkl', 'rb') as file:
    classification = pickle.load(file)

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
    # 对数据进行标准化

    average_speed = np.mean(data_split)
    speed_fluctuation = np.std(data_split)
    sample_to_predict = [average_speed, speed_fluctuation]  # 以示例数据为例，根据实际情况调整

    # 使用模型进行预测
    predicted_condition = predict_driving_condition(classification, sample_to_predict)
    step += 20
    label.append(predicted_condition)
    # print(predicted_condition)
    # print(f"{1 if predicted_condition == 1 else 0}")
print(label)