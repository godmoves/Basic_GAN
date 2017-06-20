from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#matplotlib inline

LENGTH = 1000  

def sample_data(size, length=100):
    data = []
    for _ in range(size):
        data.append(sorted(np.random.normal(4, 1.5, length)))
    return np.array(data)


def random_data(size, length=100):
    data = []
    for _ in range(size):
        x = np.random.random(length)
        data.append(x)
    return np.array(data)


def preprocess_data(x):
    return [[np.mean(data), np.std(data)] for data in x]


def plot(d_loss, g_loss):
    plt.subplot(2, 1, 1)
    plt.cla()
    plt.plot(d_loss, c="r")

    plt.subplot(2, 1, 2)
    plt.cla()
    plt.plot(g_loss, c="b")

    plt.draw()
    plt.pause(0.001)


x = tf.placeholder(tf.float32, shape=[None, 2], name="feature")
y = tf.placeholder(tf.float32, shape=[None, 1], name="label")
in_size = LENGTH
out_size = LENGTH

z = tf.placeholder(tf.float32, shape=[None, LENGTH], name="noise")
Weights = tf.Variable(tf.random_normal([in_size, 32]))
biases = tf.Variable(tf.zeros([1, 32]) + 0.1)
G_output = tf.matmul(z, Weights) + biases
G_output = tf.nn.relu(G_output)

Weights2 = tf.Variable(tf.random_normal([32, 32]))
biases2 = tf.Variable(tf.zeros([1, 32]) + 0.1)
G_output2 = tf.matmul(G_output, Weights2) + biases2
G_output2 = tf.nn.sigmoid(G_output2)

Weights3 = tf.Variable(tf.random_normal([32, out_size]))
biases3 = tf.Variable(tf.zeros([1, out_size]) + 0.1)
G_output3 = tf.matmul(G_output2, Weights3) + biases3

G_PARAMS = [Weights, biases, Weights2, biases2, Weights3, biases3]

dWeights = tf.Variable(tf.random_normal([2, 32]), name="D_W")
dbiases = tf.Variable(tf.zeros([1, 32]) + 0.1, name="D_b")
D_output = tf.matmul(x, dWeights) + dbiases
D_output = tf.nn.relu(D_output)

dWeights2 = tf.Variable(tf.random_normal([32, 32]), name="D_W2")
dbiases2 = tf.Variable(tf.zeros([1, 32]) + 0.1, name="D_b2")
D_output2 = tf.matmul(D_output, dWeights2) + dbiases2
D_output2 = tf.nn.sigmoid(D_output2)

dWeights3 = tf.Variable(tf.random_normal([32, 1]), name="D_W3")
dbiases3 = tf.Variable(tf.zeros([1, 1]) + 0.1, name="D_b3")
D_output3_ = tf.matmul(D_output2, dWeights3) + dbiases3
D_output3 = tf.nn.sigmoid(D_output3_)

D_PARAMS = [dWeights, dbiases,
            dWeights2, dbiases2,
            dWeights3, dbiases3]

MEAN = tf.reduce_mean(G_output3, 1)
MEAN_T = tf.transpose(tf.expand_dims(MEAN, 0))
STD = tf.sqrt(tf.reduce_mean(tf.square(G_output3 - MEAN_T), 1))
DATA = tf.concat([MEAN_T, tf.transpose(tf.expand_dims(STD, 0))], 1)

GAN_Weights = tf.Variable(tf.random_normal([2, 32]), name="GAN_W")
GAN_biases = tf.Variable(tf.zeros([1, 32]) + 0.1, name="GAN_b")
GAN_output = tf.matmul(DATA, GAN_Weights) + GAN_biases
GAN_output = tf.nn.relu(GAN_output)

GAN_Weights2 = tf.Variable(tf.random_normal([32, 32]), name="GAN_W2")
GAN_biases2 = tf.Variable(tf.zeros([1, 32]) + 0.1, name="GAN_b2")
GAN_output2 = tf.matmul(GAN_output, GAN_Weights2) + GAN_biases2
GAN_output2 = tf.nn.sigmoid(GAN_output2)

GAN_Weights3 = tf.Variable(tf.random_normal([32, 1]), name="GAN_W3")
GAN_biases3 = tf.Variable(tf.zeros([1, 1]) + 0.1, name="GAN_b3")
GAN_output3_ = tf.matmul(GAN_output2, GAN_Weights3) + GAN_biases3
GAN_output3 = tf.nn.sigmoid(GAN_output3_)

GAN_D_PARAMS = [GAN_Weights, GAN_biases,
                GAN_Weights2, GAN_biases2,
                GAN_Weights3, GAN_biases3]

d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_output3_, labels=y))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=GAN_output3_, labels=y))

d_optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(
    d_loss,
    global_step=tf.Variable(0),
    var_list=D_PARAMS
)

g_optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(
    g_loss,
    global_step=tf.Variable(0),
    var_list=G_PARAMS
)

d_loss_history = []
g_loss_history = []
epoch = 200
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    print('train GAN....')
    for step in range(epoch):
        for _ in range(100):
            real = sample_data(100,length=LENGTH)
            noise = random_data(100,length=LENGTH)
            generate = sess.run(G_output3, feed_dict={
                z: noise
            })
            X = list(real) + list(generate)
            X = preprocess_data(X)
            Y = [[1] for _ in range(len(real))] + [[0] for _ in range(len(generate))]
            d_loss_value, _ = sess.run([d_loss, d_optimizer], feed_dict={
                x: X,
                y: Y
            })
            d_loss_history.append(d_loss_value)
        dp_value = sess.run(D_PARAMS)
        for i, v in enumerate(GAN_D_PARAMS):
            sess.run(v.assign(dp_value[i]))

        for _ in range(100):
            noise = random_data(100,length=LENGTH)
            g_loss_value, _ = sess.run([g_loss, g_optimizer], feed_dict={
                z: noise,
                y: [[1] for _ in range(len(noise))]
            })
            g_loss_history.append(g_loss_value)
        #if step % 3 == 0 or step+1 == epoch:
        noise = random_data(1,length=LENGTH)
        generate = sess.run(G_output3, feed_dict={
            z: noise
        })
        print("[%4d] GAN-d-loss: %.12f  GAN-g-loss: %.12f   generate-mean: %.4f   generate-std: %.4f" % (step, d_loss_value, g_loss_value, generate.mean(), generate.std()))
        plot(d_loss_history, g_loss_history)

    print("train finish...")
    plt.ioff()
    plt.show()

#plt.subplot(211)
#plt.plot(d_loss_history)
#a = plt.subplot(212)
#plt.plot(g_loss_history,c="g")

# real = sample_data(1,length=LENGTH)
# (data, bins) = np.histogram(real[0])
# plt.plot(bins[:-1], data, c="g")
#
#
# (data, bins) = np.histogram(noise[0])
# plt.plot(bins[:-1], data, c="b")
#
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     generate = sess.run(G_output3, feed_dict={
# #             z: noise
# #     })
# (data, bins) = np.histogram(generate[0])
# plt.plot(bins[:-1], data, c="r")
#
# #x - x * z + log(1 + exp(-x))
#
# pre = np.array([1,0])
# real = np.array([0,1])
#
# pre-pre*real + np.log(1+np.exp(-pre))
