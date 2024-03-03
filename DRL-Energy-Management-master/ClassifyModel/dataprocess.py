import pandas as pd
from matplotlib import pyplot
import os

# 将数据拆分并写入txt文件中
if os.path.exists('../Data/suburban.txt'):
    # 如果文件存在，删除文件
    os.remove('../Data/suburban.txt')
    os.remove('../Data/urban.txt')

filepath = '../Data/drivecycle.xlsx'
excel_file = pd.ExcelFile(filepath)
# 获取工作表列表
sheet_names = excel_file.sheet_names

# 读取选定的工作表数据
df_list = []
for i in range(len(sheet_names)):
    df = excel_file.parse(sheet_names[i])
    df_list.append(df)

# 数据截断
data_len = 90  # 定义截断窗口大小
with open('../Data/suburban.txt', 'a') as f:  # 创建或追加文本文件
    for i in range(7):
        df = df_list[i].iloc[:, 1]  # 获取速度信息
        # print(df)
        for j in range(1, len(df) - data_len, 2):
            data = df[j: j + data_len]  # 截断
            data_str = ' '.join(map(str, data))  # 以空格分隔并转换为字符串
            f.write(data_str + '\n')  # 写入文本文件，每个data占一行
        pyplot.plot(range(1, data_len+1), data, label='suburban')
    pyplot.legend()
    pyplot.show()

with open('../Data/urban.txt', 'a') as f:  # 创建或追加文本文件
    for i in range(8, len(df_list)):
        df = df_list[i].iloc[:, 1]  # 获取速度信息
        for j in range(1, len(df) - data_len, 2):
            data = df[j: j + data_len]  # 截断
            data_str = ' '.join(map(str, data))  # 以空格分隔并转换为字符串
            f.write(data_str + '\n')  # 写入文本文件，每个data占一行
        pyplot.plot(range(1, data_len+1),data, label='urban')
    pyplot.legend()
    pyplot.show()
