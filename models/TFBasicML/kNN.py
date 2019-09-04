import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""使用TensorFlow实现kNN算法"""


group1_x1 = np.random.random(100) * 2
group1_x2 = np.random.random(100) * 2
group1_y = np.ones(group1_x1.shape)
group2_x1 = np.random.random(100) * 2 + 1
group2_x2=  np.random.random(100) * 2 + 1
group2_y = np.zeros(group2_x1.shape)

train_x1 = np.concatenate([np.reshape(group1_x1, [-1, 1]), np.reshape(group1_x2, [-1, 1])], axis=1)
train_x2 = np.concatenate([np.reshape(group2_x1, [-1, 1]), np.reshape(group2_x2, [-1, 1])], axis=1)
train_x = np.concatenate([train_x1, train_x2], axis=0)
train_y = np.concatenate([np.reshape(group1_y, [-1, 1]), np.reshape(group2_y, [-1, 1])], axis=0)




def kNN():
    input_x = tf.placeholder(tf.float32, [None, 2], name='input_x')
    label_y = tf.placeholder(tf.float32, [None, 1], name='label_y')

    test_x = tf.placeholder(tf.float32, [2], name='test_x')

    distance = tf.reduce_mean(tf.square(test_x - input_x), axis=1)
    predict = tf.gather(label_y, tf.argmax(distance, axis=0))

    return input_x, label_y, test_x, predict

def eval(test_sample):
    input_x, label_y, test_x, predict = kNN()
    with tf.Session() as sess:
        feed_dict = {input_x:train_x, label_y:train_y, test_x:test_sample}
        predict_label = sess.run(predict, feed_dict)
        print('Predition:{}'.format(predict_label[0]))
        show_figure(test_sample)
        return predict_label

def show_figure(test_sample):
    plt.scatter(group1_x1, group1_x2, color='blue')
    plt.scatter(group2_x1, group2_x2, color='yellow')
    plt.scatter(test_sample[0], test_sample[1], color='red')
    plt.show()



if __name__ == '__main__':
    test_sample = np.array([2.2, 2.2])
    eval(test_sample)

