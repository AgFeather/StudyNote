import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""使用TensorFlow实现LR算法并可视化训练过程中的各个阶段"""

group1_x1 = np.random.random(100) * -1
group1_x2 = np.random.random(100) * -1
group1_y = np.ones(group1_x1.shape)
group2_x1 = np.random.random(100) #* 11 + 9
group2_x2=  np.random.random(100) #* 11 + 9
group2_y = np.zeros(group2_x1.shape)

train_x1 = np.concatenate([np.reshape(group1_x1, [-1, 1]), np.reshape(group1_x2, [-1, 1])], axis=1)
train_x2 = np.concatenate([np.reshape(group2_x1, [-1, 1]), np.reshape(group2_x2, [-1, 1])], axis=1)
train_x = np.concatenate([train_x1, train_x2], axis=0)
train_y = np.concatenate([np.reshape(group1_y, [-1, 1]), np.reshape(group2_y, [-1, 1])], axis=0)


def logistic_regression():
    input_x = tf.placeholder(tf.float32, [None, 2], name='input_x')
    label_y = tf.placeholder(tf.float32, [None, 1], name='label_y')

    weights = tf.get_variable('weights', [2], initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(0.0))

    output = tf.sigmoid(tf.multiply(input_x, weights) + biases)

    loss = tf.reduce_mean(tf.square(output - label_y))
    onehot_output = tf.one_hot(tf.cast(output, tf.int32), depth=2)
    onehot_label = tf.one_hot(tf.cast(label_y, tf.int32), depth=2)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(onehot_output, axis=1), tf.argmax(onehot_label, axis=1)), tf.float32))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    return input_x, label_y, weights, biases, loss, accuracy, optimizer


def train(num_epochs=200):
    input_x, label_y, weights, biases, loss, accuracy, optimizer = logistic_regression()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_epochs):
            feed_dict = {input_x:train_x, label_y:train_y}
            w, b, show_loss, show_accuracy, _ = sess.run([weights, biases, loss, accuracy, optimizer], feed_dict)
            if i % 20 == 0:
                print("Epoch:{}, loss:{:.4f}, accuracy:{:.2f}%".format(i, show_loss, show_accuracy*100))
                show_figure(w, b)



def show_figure(w, b):
    print("w:{}, b:{}".format(w, b))
    plt.scatter(group1_x1, group1_x2, color='blue')
    plt.scatter(group2_x1, group2_x2, color='green')
    x1 = np.linspace(-1, 1, 100)
    y = -(w[0] * x1 + b[0]) / w[1]
    plt.plot(x1, y, color='red')
    plt.show()




if __name__ == '__main__':
    train()