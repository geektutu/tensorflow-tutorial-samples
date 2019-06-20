# train.py
# https://geektutu.com
import random
import gym
import numpy as np
from tensorflow.keras import models, layers

env = gym.make("CartPole-v0")  # 加载游戏环境

STATE_DIM, ACTION_DIM = 4, 2  # State 维度 4, Action 维度 2
model = models.Sequential([
    layers.Dense(64, input_dim=STATE_DIM, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(ACTION_DIM, activation='linear')
])
model.summary()  # 打印神经网络信息


def generate_data_one_episode():
    '''生成单次游戏的训练数据'''
    x, y, score = [], [], 0
    state = env.reset()
    while True:
        action = random.randrange(0, 2)
        x.append(state)
        y.append([1, 0] if action == 0 else [0, 1]) # 记录数据
        state, reward, done, _ = env.step(action) # 执行动作
        score += reward
        if done:
            break
    return x, y, score


def generate_training_data(expected_score=100):
    '''# 生成N次游戏的训练数据，并进行筛选，选择 > 100 的数据作为训练集'''
    data_X, data_Y, scores = [], [], []
    for i in range(10000):
        x, y, score = generate_data_one_episode()
        if score > expected_score:
            data_X += x
            data_Y += y
            scores.append(score)
    print('dataset size: {}, max score: {}'.format(len(data_X), max(scores)))
    return np.array(data_X), np.array(data_Y)


data_X, data_Y = generate_training_data()
model.compile(loss='mse', optimizer='adam')
model.fit(data_X, data_Y, epochs=5)
model.save('CartPole-v0-nn.h5')  # 保存模型
