# test_q_learning.py
# https://geektutu.com
import time
import pickle
import gym
import numpy as np


def transform_state(state):
    """将 position, velocity 通过线性转换映射到 [0, 40] 范围内"""
    pos, v = state
    pos_low, v_low = env.observation_space.low
    pos_high, v_high = env.observation_space.high

    a = 40 * (pos - pos_low) / (pos_high - pos_low)
    b = 40 * (v - v_low) / (v_high - v_low)

    return int(a), int(b)


# 加载模型
with open('MountainCar-v0-q-learning.pickle', 'rb') as f:
    Q = pickle.load(f)
    print('model loaded')

env = gym.make('MountainCar-v0')
s = env.reset()
score = 0
while True:
    env.render()
    time.sleep(0.01)
    # transform_state函数 与 训练时的一致
    s = transform_state(s)
    a = np.argmax(Q[s]) if s in Q else 0
    s, reward, done, _ = env.step(a)
    score += reward
    if done:
        print('score:', score)
        break
env.close()
