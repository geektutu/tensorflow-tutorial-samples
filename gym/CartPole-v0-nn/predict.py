# predict.py
# https://geektutu.com
import time
import numpy as np
import gym
from tensorflow.keras import models


saved_model = models.load_model('CartPole-v0-nn.h5')  # 加载模型
env = gym.make("CartPole-v0")  # 加载游戏环境

for i in range(5):
    state = env.reset()
    score = 0
    while True:
        time.sleep(0.01)
        env.render()   # 显示画面
        action = np.argmax(saved_model.predict(np.array([state]))[0])  # 预测动作
        state, reward, done, _ = env.step(action)  # 执行这个动作
        score += reward     # 每回合的得分
        if done:       # 游戏结束
            print('using nn, score: ', score)  # 打印分数
            break
env.close()
