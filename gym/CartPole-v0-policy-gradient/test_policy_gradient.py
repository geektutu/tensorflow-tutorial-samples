# test_policy_gradient.py
# https://geektutu.com
import time
import numpy as np
import gym
from tensorflow.keras import models


saved_model = models.load_model('CartPole-v0-pg.h5')
env = gym.make("CartPole-v0")

for i in range(5):
    s = env.reset()
    score = 0
    while True:
        time.sleep(0.01)
        env.render()
        prob = saved_model.predict(np.array([s]))[0]
        a = np.random.choice(len(prob), p=prob)
        s, r, done, _ = env.step(a)
        score += r
        if done:
            print('using policy gradient, score: ', score)  # 打印分数
            break
env.close()
