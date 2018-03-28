import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import Network

'''
python 3.6
tensorflow 1.4

重点对训练可视化的部分添加了注释
如果想看其他代码的注释，请移步 v1, v2
v3比v2增加了loss和accuracy的可视化
'''

CKPT_DIR = 'ckpt'


class Train:
    def __init__(self):
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.data = input_data.read_data_sets('../data_set', one_hot=True)

    def train(self):
        batch_size = 64
        train_step = 20000
        step = 0
        save_interval = 1000
        saver = tf.train.Saver(max_to_keep=5)

        # merge所有的summary node
        merged_summary_op = tf.summary.merge_all()
        # 可视化存储目录为当前文件夹下的 log
        merged_writer = tf.summary.FileWriter("./log", self.sess.graph)

        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            # 读取网络中的global_step的值，即当前已经训练的次数
            step = self.sess.run(self.net.global_step)
            print('Continue from')
            print('        -> Minibatch update : ', step)

        while step < train_step:
            x, label = self.data.train.next_batch(batch_size)
            _, loss, merged_summary = self.sess.run(
                [self.net.train, self.net.loss, merged_summary_op],
                feed_dict={self.net.x: x, self.net.label: label}
            )
            step = self.sess.run(self.net.global_step)

            if step % 100 == 0:
                merged_writer.add_summary(merged_summary, step)

            if step % save_interval == 0:
                saver.save(self.sess, CKPT_DIR + '/model', global_step=step)
                print('%s/model-%d saved' % (CKPT_DIR, step))

    def calculate_accuracy(self):
        test_x = self.data.test.images
        test_label = self.data.test.labels
        accuracy = self.sess.run(self.net.accuracy,
                                 feed_dict={self.net.x: test_x, self.net.label: test_label})
        print("准确率: %.2f，共测试了%d张图片 " % (accuracy, len(test_label)))


if __name__ == "__main__":
    app = Train()
    app.train()
    app.calculate_accuracy()
