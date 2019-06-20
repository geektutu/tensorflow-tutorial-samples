# try_gym.py
# https://geektutu.com
import gym  # 0.12.5
import random
import time

env = gym.make("CartPole-v0")  # 加载游戏环境

state = env.reset()
score = 0
while True:
    time.sleep(0.1)
    env.render()   # 显示画面
    action = random.randint(0, 1)  # 随机选择一个动作 0 或 1
    state, reward, done, _ = env.step(action)  # 执行这个动作
    score += reward     # 每回合的得分
    if done:       # 游戏结束
        print('score: ', score)  # 打印分数
        break
env.close()
