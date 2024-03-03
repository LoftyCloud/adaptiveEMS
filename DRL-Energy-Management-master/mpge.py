import scipy.io as scio
import numpy as np
import pandas as pd

initSOC = 0.6

# # path = 'DDPGmodel/HWFET/output.xlsx'
# # path = 'DDPGmodel/UDDS/output.xlsx'
# path = 'DDPGmodel/Proposed/output3.xlsx'
# # 加载综合驾驶循环
# data_path = 'E:\documents\DRLResult2\data\DCdata2.mat'
# # path = 'DDPGmodel/UDDS/'
# data = scio.loadmat(data_path)
# car_spd_one = data['data']  # 速度数据
# total_milage = np.sum(car_spd_one) / 1000

total_milage = 78.71062049999999

# 读取xlsx文件
# finalSOC = 0.562
# fuel = 2493.28

# finalSOC = 0.891
# fuel = 2416.67
#
# finalSOC = 0.568
# fuel = 2541.22

finalSOC = 0.657
fuel = 2335.18
P_Bat = (initSOC - finalSOC) * 1.54  # kWh
mpge = (total_milage * 0.621371) / (264.172 * fuel / (0.72 * 1000000) + P_Bat / 33.7)
print(mpge)
