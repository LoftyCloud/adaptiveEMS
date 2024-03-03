# -*- coding: utf-8 -*-
import os

import pandas as pd

from Prius_model import Prius_model
import scipy.io as scio
import torch
import torch.nn as nn
import numpy as np


# 加载Agent
class Actor(nn.Module):
    def __init__(self, state_dim=3, action_dim=1, action_high=1):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 150)
        self.fc2 = nn.Linear(150, 75)
        self.fc3 = nn.Linear(75, 50)
        self.fc4 = nn.Linear(50, action_dim)
        self.action_high = action_high

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x)) * self.action_high
        return x


def np_to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


# AgentName = 'UDDS150'

AgentName = 'UDDS200'
DC = 'UDDS'

actor = Actor()
actor.load_state_dict(torch.load('../model/'+AgentName + '.pth'))
# 加载工况数据
data_path = '../Data_Standard Driving Cycles/Standard_' + DC + '.mat'
data = scio.loadmat(data_path)
car_spd_one = data['speed_vector']  # [[0,0]]

# car_spd_one = np.tile(car_spd_one, 2)
# ####################  仿真过程  ####################
# 状态空间与动作空间维度
s_dim = 3
a_dim = 1
a_bound = 1
# 初始化变量
total_milage = np.sum(car_spd_one) / 1000  # 11

total_step = 0
step_episode = 0
mean_reward_all = 0

# 车辆模型
Prius = Prius_model()
# 仿真设置
SOC = 0.6
SOC_origin = SOC
ep_reward = 0
ep_reward_all = 0
step_episode += 1
SOC_data = []
# 记录仿真过程数据
V_list = []
I_list = []
P_Batt_list = []
P_req_list = []
P_out_list = []
Eng_spd_list = []
Eng_trq_list = []
Eng_pwr_list = []
Eng_pwr_opt_list = []
Gen_spd_list = []
Gen_trq_list = []
Gen_pwr_list = []
Mot_spd_list = []
Mot_trq_list = []
Mot_pwr_list = []
Batt_pwr_list = []
inf_batt_list = []
inf_batt_one_list = []
Reward_list = []
Reward_list_all = []
T_list = []
Mot_eta_list = []
Gen_eta_list = []
fuel_list = []
# 初始速度与初始加速度
car_spd = car_spd_one[:, 0]
car_a = car_spd_one[:, 0] - 0
# 初始状态
s = np.zeros(s_dim)
s[0] = car_spd / 33.4
s[1] = (car_a - (-1.5)) / (1.5 - (-1.5))
s[2] = SOC

import time
T = []
for j in range(len(car_spd_one[0]) - 1):
    starttime = time.time()
    a = actor(np_to_tensor(s)).detach().numpy()
    Eng_pwr_opt = (a[0]) * 56000
    # print(a)

    # 仿真一步
    out, cost, I = Prius.run(car_spd, car_a, Eng_pwr_opt, SOC)
    # 记录变量变化
    V_list.append(float(out['Batt_V']))
    I_list.append(float(out['Batt_I']))
    P_Batt_list.append(float(out['P_batt']))
    P_req_list.append(float(out['P_req']))
    P_out_list.append(float(out['P_out']))
    Eng_spd_list.append(float(out['Eng_spd']))
    Eng_trq_list.append(float(out['Eng_trq']))
    Eng_pwr_list.append(float(out['Eng_pwr']))
    Eng_pwr_opt_list.append(float(out['Eng_pwr_opt']))
    Mot_spd_list.append(float(out['Mot_spd']))
    Mot_trq_list.append(float(out['Mot_trq']))
    Mot_pwr_list.append(float(out['Mot_pwr']))
    Gen_spd_list.append(float(out['Gen_spd']))
    Gen_trq_list.append(float(out['Gen_trq']))
    Gen_pwr_list.append(float(out['Gen_pwr']))
    Batt_pwr_list.append(float(out['Batt_pwr']))
    inf_batt_list.append(int(out['inf_batt']))
    inf_batt_one_list.append(int(out['inf_batt_one']))
    Mot_eta_list.append(float(out['Mot_eta']))
    Gen_eta_list.append(float(out['Gen_eta']))
    fuel_list.append(float(out['fuel']))
    T_list.append(float(out['T']))
    SOC_new = float(out['SOC'])
    SOC_data.append(SOC_new)
    # 燃油消耗量记录
    cost = float(cost)
    r = cost
    ep_reward += r  # 累计奖励

    # 状态更新
    car_spd = car_spd_one[:, j + 1]
    car_a = car_spd_one[:, j + 1] - car_spd_one[:, j]
    s_ = np.zeros(s_dim)
    s_[0] = car_spd / 33.4
    s_[1] = (car_a - (-1.5)) / (1.5 - (-1.5))
    s_[2] = SOC_new
    s = s_
    SOC = SOC_new
    endtime = time.time()
    T.append(1000*(endtime-starttime))
# print(sum(T)/(len(car_spd_one[0]) - 1))
# print(max(T))

print(starttime)
print(endtime)
print((endtime-starttime)/(len(car_spd_one[0]) - 1)*1000)

data = {
    'Batt_V': V_list,
    'Batt_I': I_list,
    'P_batt': P_Batt_list,
    'P_req': P_req_list,
    'P_out': P_out_list,
    'Eng_spd': Eng_spd_list,
    'Eng_trq': Eng_trq_list,
    'Eng_pwr': Eng_pwr_list,
    'Eng_pwr_opt': Eng_pwr_opt_list,
    'Mot_spd': Mot_spd_list,
    'Mot_trq': Mot_trq_list,
    'Mot_pwr': Mot_pwr_list,
    'Gen_spd': Gen_spd_list,
    'Gen_trq': Gen_trq_list,
    'Gen_pwr': Gen_pwr_list,
    'Batt_pwr': Batt_pwr_list,
    'inf_batt': inf_batt_list,
    'inf_batt_one': inf_batt_one_list,
    'Mot_eta': Mot_eta_list,
    'Gen_eta': Gen_eta_list,
    'T': T_list,
    'SOC_data': SOC_data,
    'fuel': fuel_list,
}
# print(ep_reward)
cost_Engine = (-ep_reward / 0.72 / 1000)  # g，密度，单位转换  -> 燃油消耗
SOC_final = SOC_new
data2 = {
    'cost_Engine': [cost_Engine],
    'Fuel_100Km': [(cost_Engine * (100 / total_milage))],  # 每100英里燃油消耗
    'SOC-final': [SOC_final]
}

df = pd.DataFrame(data)
df2 = pd.DataFrame(data2)

file_path = 'Result/' + AgentName + '_' + DC + '.xlsx'

# 如果文件不存在，则创建文件
if not os.path.exists(file_path):
    with open(file_path, 'w'):
        pass
    print(f"文件 {file_path} 创建成功。")

# 尝试读取文件，如果文件不存在，创建一个空的Excel文件
try:
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet2', index=False)
except FileNotFoundError:
    # 创建一个新的 Excel 文件，并写入数据
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet2', index=False)
    print(f"文件 {file_path} 不存在，已创建并写入数据。")
