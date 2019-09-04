'''
使用TensorFlow构建一个2层encoder和2层decoder的自编码机，对MNIST数据进行分类
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

learning_rate = 0.01
num_steps = 5000
batch_size = 256

display_step = 1000
examples_to_show = 10

num_hidden1 = 256
num_hidden2 = 128
num_input = 784

input_x = tf.placeholder(tf.float32, [None, num_input])

weights = {
    'encoder_h1':tf.Variable(tf.random_normal([num_input, num_hidden1])),
    'encoder_h2':tf.Variable(tf.random_normal([num_hidden1, num_hidden2])),
    'decoder_h1':tf.Variable(tf.random_normal([num_hidden2, num_hidden1])),
    'decoder_h2':tf.Variable(tf.random_normal([num_hidden1, num_input]))
}

biases = {
    'encoder_b1':tf.Variable(tf.random_normal([num_hidden1])),
    'encoder_b2':tf.Variable(tf.random_normal([num_hidden2])),
    'decoder_b1':tf.Variable(tf.random_normal([num_hidden1])),
    'decoder_b2':tf.Variable(tf.random_normal([num_input]))
}

def encoder(input_x):
    layer_1 = tf.add(tf.matmul(input_x, weights['encoder_h1']), biases['encoder_b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    return layer_2

def decoder(input_x):
    layer_1 = tf.add(tf.matmul(input_x, weights['decoder_h1']),biases['decoder_b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    return layer_2

encoder_op = encoder(input_x)
prediction = decoder(encoder_op)

loss = tf.reduce_mean(tf.pow(prediction - input_x, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_steps):
        batch_x, _  = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={input_x:batch_x})
        if i%display_step == 0:
            s_loss = sess.run(loss, feed_dict={input_x:batch_x})
            print('training step: %d, loss: %.2f'%(i, s_loss))


    #test
    #encode and decode images from test set and visualize their reconstruction
    n = 4
    convas_orig = np.empty((28*n, 28*n))
    convas_recon = np.empty((28*n, 28*n))
    for i in range(n):
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
