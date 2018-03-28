import tensorflow as tf


class Network:
    def __init__(self):
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self.x = tf.placeholder(tf.float32, [None, 784], name="x")
        self.label = tf.placeholder(tf.float32, [None, 10], name="label")

        self.w = tf.Variable(tf.zeros([784, 10]), name="fc/weight")
        self.b = tf.Variable(tf.zeros([10]), name="fc/bias")
        self.y = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b, name="y")

        self.loss = -tf.reduce_sum(self.label * tf.log(self.y + 1e-10))
        self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
            self.loss, global_step=self.global_step)

        predict = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(predict, "float"))

        # 创建 summary node
        # w, b 画直方图
        # loss, accuracy画标量图
        tf.summary.histogram('weight', self.w)
        tf.summary.histogram('bias', self.b)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
