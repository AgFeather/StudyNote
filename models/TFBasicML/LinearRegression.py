import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


"""使用TensorFlow实现Linear Regression算法并可视化训练过程中的各个阶段"""


num_samples = 200
weight = 2.0
bias = 0.5
x = np.linspace(-1, 1, 200)
y = weight * x + np.random.standard_normal(x.shape) * 0.3 + bias#生成线性数据
train_x = np.reshape(x, [num_samples, 1])
train_y = np.reshape(y, [num_samples, 1])


def show_figure(w, b):
    plt.scatter(x, y)
    plt.plot(x, w * x + b, color='red')
    plt.show()


def linear_regression():
    input_x = tf.placeholder(tf.float32, [None, 1], name='input_x')
    target_y = tf.placeholder(tf.float32, [None, 1], name='target_y')

    weight = tf.get_variable('weight',[1], initializer=tf.truncated_normal_initializer(stddev=0.1))
    bias = tf.get_variable('bias', [1], initializer=tf.constant_initializer(0.0))

    output = tf.multiply(input_x, weight) + bias

    loss = tf.reduce_mean(tf.square(output - target_y))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
    return input_x, target_y, weight, bias, loss, optimizer

def train(num_epochs=500):
    input_x, target_y, weight, bias, loss, optimizer = linear_regression()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_epochs):
            feed_dict = {input_x: train_x, target_y: train_y}
            show_loss, w, b, _ = sess.run([loss, weight, bias, optimizer], feed_dict)
            if i % 50 == 0:
                print("global step:{} loss:{:.3f}".format(i, show_loss))
                show_figure(w, b)

if __name__ == '__main__':
    train()






