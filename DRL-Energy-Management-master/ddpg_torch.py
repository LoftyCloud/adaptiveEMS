from Prius_model import Prius_model
import scipy.io as scio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 设置随机种子
torch.manual_seed(1)
np.random.seed(1)

# 超参数
MAX_EPISODES = 300
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64

# 转换为PyTorch的张量
def np_to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_high):
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

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 150)
        self.fc2 = nn.Linear(150, 75)
        self.fc3 = nn.Linear(75, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DDPG:
    def __init__(self, state_dim = 3, action_dim = 1, action_high = 1, action_low = 0):
        self.actor = Actor(state_dim, action_dim, action_high)
        self.target_actor = Actor(state_dim, action_dim, action_high)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_A)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_C)

        self.memory = np.zeros(MEMORY_CAPACITY, dtype=object)
        self.pointer = 0

    def choose_action(self, state):
        state = np_to_tensor(state)
        return self.actor(state).detach().numpy()

    def learn(self):
        if self.pointer < BATCH_SIZE:
            return

        indices = np.random.choice(min(self.pointer, MEMORY_CAPACITY), size=BATCH_SIZE)
        batch = [self.memory[i] for i in indices]
        state_batch = np_to_tensor(np.vstack([item[0] for item in batch]))
        action_batch = np_to_tensor(np.vstack([item[1] for item in batch]))
        reward_batch = np_to_tensor(np.vstack([item[2] for item in batch]))
        next_state_batch = np_to_tensor(np.vstack([item[3] for item in batch]))

        target_actions = self.target_actor(next_state_batch).detach()
        target_q_values = self.target_critic(next_state_batch, target_actions).detach()
        target_q_values *= GAMMA

        # reward_batch = reward_batch.unsqueeze(1)
        target_q_values += reward_batch

        critic_loss = nn.functional.mse_loss(self.critic(state_batch, action_batch), target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update()

    def store_transition(self, state, action, reward, next_state):
        self.memory[self.pointer] = (state, action, reward, next_state)
        self.pointer += 1
        if self.pointer >= MEMORY_CAPACITY:
            self.pointer = 0

    # 软更新目标网络
    def soft_update(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_((1 - TAU) * target_param.data + TAU * param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_((1 - TAU) * target_param.data + TAU * param.data)

    def savemodel(self,dirpath):
        torch.save(self.actor.state_dict(), dirpath + 'ddpg_actor1107.pth')

DDPG = DDPG()
# control exploration
var = 2

# 加载驾驶循环数据
data_path = 'Data_Standard Driving Cycles/Standard_HWFET.mat'
path = 'DDPGmodel/UDDS/'

data = scio.loadmat(data_path)
car_spd_one = data['speed_vector']  # 速度向量
total_milage = np.sum(car_spd_one) / 1000
total_step = 0
step_episode = 0

mean_reward_all = 0
cost_Engine_list = []
cost_all_list = []
cost_Engine_100Km_list = []
mean_reward_list = []
list_even = []
list_odd = []
mean_discrepancy_list = []
SOC_final_list = []
# 车辆模型
Prius = Prius_model()

# for i in range(MAX_EPISODES):
for i in range(100):
    # 定义初值
    SOC = 0.65
    SOC_origin = SOC
    ep_reward = 0
    ep_reward_all = 0
    step_episode += 1
    SOC_data = []
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
    car_spd = car_spd_one[:, 0]
    car_a = car_spd_one[:, 0] - 0

    s = np.zeros(3)  # 初始化环境状态，（速度，加速度，SOC）
    s[0] = car_spd / 33.4
    s[1] = (car_a - (-1.5)) / (1.5 - (-1.5))  # 加速度
    s[2] = SOC
    # 运行驾驶循环
    for j in range(car_spd_one.shape[1] - 1):
        # 初始动作
        action = DDPG.choose_action(s)
        # 拉普拉斯分布为动作添加扰动
        a = np.clip(np.random.laplace(action, var), 0, 1)
        # 发动机输出功率
        Eng_pwr_opt = (a[0]) * 56000

        out, cost, I = Prius.run(car_spd, car_a, Eng_pwr_opt, SOC)
        # 记录变量变化
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
        T_list.append(float(out['T']))
        SOC_new = float(out['SOC'])
        SOC_data.append(SOC_new)
        cost = float(cost)  # cost = (Eng_pwr / 42600)

        # 奖励函数（r<0）
        r = cost
        ep_reward += r
        Reward_list.append(r)
        if SOC_new < 0.6 or SOC_new > 0.85:
            r =  cost - (350 * ((0.6 - SOC_new) ** 2))

        # Obtained from the wheel speed sensor
        car_spd = car_spd_one[:, j + 1]
        car_a = car_spd_one[:, j + 1] - car_spd_one[:, j]
        s_ = np.zeros(3)
        s_[0] = car_spd / 33.4
        s_[1] = (car_a - (-1.5)) / (1.5 - (-1.5))
        s_[2] = SOC_new
        DDPG.store_transition(s, a, r, s_)

        if total_step > MEMORY_CAPACITY:
            var *= 0.99993
            DDPG.learn()

        s = s_
        ep_reward_all += r
        Reward_list_all.append(r)
        total_step += 1
        SOC = SOC_new
        cost_Engine = (-ep_reward / 0.72 / 1000)
        cost_all = (-ep_reward_all / 0.72 / 1000)

        # 车辆驾驶循环结束，打印训练结果
        if j == (car_spd_one.shape[1] - 2):
            SOC_final_list.append(SOC)
            mean_reward = ep_reward_all / car_spd_one.shape[1]
            mean_reward_list.append(mean_reward)
            cost_Engine += (SOC < SOC_origin) * (SOC_origin - SOC) * (201.6 * 6.5) * 3600 / (42600000) / 0.72
            cost_Engine_list.append(cost_Engine)
            cost_Engine_100Km_list.append(cost_Engine * (100 / total_milage))
            cost_all += (SOC < SOC_origin) * (SOC_origin - SOC) * (201.6 * 6.5) * 3600 / (42600000) / 0.72
            cost_all_list.append(cost_all)

            print('Episode:', i,'mean_reward: %.3f'%mean_reward, ' cost_Engine: %.3f' % cost_Engine,
                  ' Fuel_100Km: %.3f' % (cost_Engine * (100 / total_milage)), ' SOC-final: %.3f' % SOC,
                  ' Explore: %.2f' % var)

    x = np.arange(0, len(SOC_data), 1)
    y = SOC_data
    plt.plot(x, y)
    plt.xlabel('Time(s)')
    plt.ylabel('SOC')
    plt.savefig(path + 'ddpg.png')

DDPG.savemodel(path)

