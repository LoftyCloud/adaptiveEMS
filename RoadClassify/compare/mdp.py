import numpy as np
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
# print(data1.shape)

data = np.concatenate([data1, data2])  # (2000,90)
labels = np.concatenate([label1, label2])  #(2000,)
# print(data.shape)
# print(labels.shape)

# 打乱数据和标签的顺序，以确保数据的随机性
permutation = np.random.permutation(len(data))
data = data[permutation]
labels = labels[permutation]
from sklearn.preprocessing import StandardScaler
# 对数据进行标准化
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# 准备训练数据
X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.1, random_state=42)

import gym
class DrivingConditionEnv(gym.Env):
    def __init__(self, data, labels):
        super(DrivingConditionEnv, self).__init__()

        self.data = data
        self.labels = labels
        self.current_step = 0

        # 定义状态空间和动作空间
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)  # 二元分类

    def reset(self):
        # 在每个episode开始时重置状态
        self.current_step = 0
        state = self._get_state()
        return state

    def _get_state(self):
        # 从当前速度序列中提取特征作为状态
        current_speed_sequence = self.data[self.current_step]

        average_speed = np.mean(current_speed_sequence)
        speed_std = np.std(current_speed_sequence)
        average_acceleration = np.mean(np.diff(current_speed_sequence))
        maximum_speed = np.max(current_speed_sequence)

        # 将状态值取整
        state = np.array([average_speed, speed_std, average_acceleration, maximum_speed], dtype=int)

        return state

    def step(self, action):
        # 在环境中执行动作，返回下一个状态、奖励、是否完成等信息
        reward = 1 if action == self.labels[self.current_step] else 0
        done = (self.current_step == len(self.data) - 2)

        self.current_step += 1
        next_state = self._get_state()
        return next_state, reward, done, {}


# 创建强化学习环境
# env = DrivingConditionEnv(data_normalized, labels)
env = DrivingConditionEnv(data, labels)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

# 训练代理
num_episodes = 500
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update_q_network(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    agent.decay_exploration_prob()

    if (episode+1) % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# 保存模型
torch.save(agent, 'dqn_agent.pth')

#
# # 使用代理进行预测
# state = env.reset()
# while True:
#     action = agent.choose_action(state)
#     next_state, reward, done, _ = env.step(action)
#     if done:
#         break
#     state = next_state