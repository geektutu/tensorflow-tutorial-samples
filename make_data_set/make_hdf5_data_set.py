import os
import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

"""
h5py: 2.7.1
PIL 4.3.0 (pip install pillow)
numpy 1.13.1
sklearn: 0.19.1
"""
if not os.path.exists('./data_set'):
    os.mkdir('./data_set')


def make_hdf5_data_set():
    x, y = [], []

    for i, image_path in enumerate(os.listdir('./images')):
        # label转为独热编码后再保存
        label = int(image_path.split('_')[0])
        label_one_hot = [0 if i != label else 1 for i in range(10)]
        y.append(label_one_hot)

        # 图片像素值映射到 0 - 1之间
        image = Image.open('./images/{}'.format(image_path)).convert('L')
        image_arr = 1 - np.reshape(image, 784) / 255.0
        x.append(image_arr)

    with h5py.File('./data_set/data.h5', 'w') as f:
        f.create_dataset('x_data', data=np.array(x))
        f.create_dataset('y_data', data=np.array(y))


class DataSet:
    def __init__(self):
        with h5py.File('./data_set/data.h5', 'r') as f:
            x, y = f['x_data'].value, f['y_data'].value

        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(x, y, test_size=0.2, random_state=0)

        self.train_size = len(self.train_x)

    def get_train_batch(self, batch_size=64):
        # 随机获取batch_size个训练数据
        choice = np.random.randint(self.train_size, size=batch_size)
        batch_x = self.train_x[choice, :]
        batch_y = self.train_y[choice, :]

        return batch_x, batch_y

    def get_test_set(self):
        return self.test_x, self.test_y


if __name__ == '__main__':
    make_hdf5_data_set()
    import time

    s = time.time()
    for i in range(1000):
        data_set = DataSet()
        train_x, train_y = data_set.get_train_batch()
        test_x, test_y = data_set.get_test_set()

    print(time.time() - s)
