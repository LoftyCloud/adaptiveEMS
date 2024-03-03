import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from DCload import dcload
data = dcload()


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, input_size, output_size, learning_rate=0.001, discount_factor=0.9, exploration_prob=1.0, exploration_decay=0.995):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay

        # 初始化神经网络和优化器
        self.q_network = QNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def choose_action(self, state):
        # epsilon-greedy 策略选择动作
        if torch.rand(1).item() < self.exploration_prob:
            return torch.randint(self.output_size, (1,)).item()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item()

    def update_q_network(self, state, action, reward, next_state):
        # 将状态和下一个状态转换为 PyTorch Tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # 预测当前状态的 Q 值
        q_values = self.q_network(state_tensor)
        current_q_value = q_values[0, action]

        # 预测下一个状态的 Q 值
        next_q_values = self.q_network(next_state_tensor)
        max_future_q = torch.max(next_q_values)

        # 计算 TD 误差
        td_error = reward + self.discount_factor * max_future_q - current_q_value

        # 更新神经网络参数
        loss = td_error**2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_exploration_prob(self):
        # 减小探索概率
        self.exploration_prob *= self.exploration_decay

# 创建 DQN 代理
state_space_size = 4
action_space_size = 2
agent = DQNAgent(state_space_size, action_space_size)

loaded_agent = torch.load('dqn_agent.pth')

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
    # 将状态值取整
    state = np.array([average_speed, speed_std, average_acceleration, maximum_speed], dtype=int)
    # 使用模型选择动作
    with torch.no_grad():
        action = agent.choose_action(torch.tensor(state).float())
    step += 20
    # print(action)
    label.append(action)
print(label)