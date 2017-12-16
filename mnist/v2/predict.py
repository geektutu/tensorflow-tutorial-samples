import tensorflow as tf
import numpy as np
from PIL import Image

from model import Network

'''
python 3.6
tensorflow 1.4
pillow(PIL) 4.3.0
使用tensorflow的模型来预测手写数字
输入是28 * 28像素的图片，输出是个具体的数字
'''

CKPT_DIR = 'ckpt'


class Predict:
    def __init__(self):
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # 加载模型到sess中
        self.restore()

    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError("未保存任何模型")

    def predict(self, image_path):
        # 读图片并转为黑白的
        img = Image.open(image_path).convert('L')
        flatten_img = np.reshape(img, 784)
        x = np.array([1 - flatten_img])
        y = self.sess.run(self.net.y, feed_dict={self.net.x: x})

        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得独热编码最大值的下标，即代表的数字
        print(image_path)
        print('        -> Predict digit', np.argmax(y[0]))


if __name__ == "__main__":
    app = Predict()
    app.predict('../test_images/0.png')
    app.predict('../test_images/1.png')
    app.predict('../test_images/4.png')
