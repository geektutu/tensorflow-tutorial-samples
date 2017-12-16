import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import Network

'''
python 3.6
tensorflow 1.4

重点对保存模型的部分添加了注释
如果想看其他代码的注释，请移步 v1
v2 版本比 v1 版本增加了模型的保存和继续训练
'''


class Train:
    def __init__(self):
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.data = input_data.read_data_sets('../data_set', one_hot=True)

        # tf.train.Saver是用来保存训练结果的。
        # max_to_keep 用来设置最多保存多少个模型，默认是5
        # 如果保存的模型超过这个值，最旧的模型将被删除
        self.saver = tf.train.Saver(max_to_keep=10)

    def train(self):
        batch_size = 64
        train_step = 10000
        # 每隔1000步保存模型
        save_interval = 1000

        # step 记录训练次数
        # 读取网络中的global_step的值，即当前已经训练的次数
        step = self.sess.run(self.net.global_step)

        while step < train_step:
            x, label = self.data.train.next_batch(batch_size)
            _, loss = self.sess.run([self.net.train, self.net.loss],
                                          feed_dict={self.net.x: x, self.net.label: label})
            step = self.sess.run(self.net.global_step)
            if step % 10 == 0:
                print('第%5d步，当前loss：%.2f' % (step, loss))

            # 每隔1000步保存一次模型，模型保存在ckpt文件夹下名为model的文件，
            # 模型文件名最后会增加global_step的值，比如1000的模型文件名为 model-1000
            if step % save_interval == 0:
                self.saver.save(self.sess, 'ckpt/model', global_step=step)

    def calculate_accuray(self):
        test_x = self.data.test.images
        test_label = self.data.test.labels
        accuracy = self.sess.run(self.net.accuracy,
                                 feed_dict={self.net.x: test_x, self.net.label: test_label})
        print("准确率: %.2f，共测试了%d张图片 " % (accuracy, len(test_label)))


if __name__ == "__main__":
    app = Train()
    app.train()
    app.calculate_accuray()
