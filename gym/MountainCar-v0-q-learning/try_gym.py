import time
import random
import gym  # 0.12.5

env = gym.make('MountainCar-v0')
env.reset()
score = 0
for i in range(2):
    while True:
        env.render()
        time.sleep(0.01)
        a = random.randint(0, 2)
        s, reward, done, _ = env.step(a)
        score += reward
        if done:
            print('score:', score)
            break
    env.close()