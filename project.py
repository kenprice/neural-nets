# Usage python project.py <number of hidden layers> <neurons in first hidden layer> <neurons in second hidden layer>
#
# TASK 1:  Neural network with only 1 hidden layer with 15 neurons.
# TASK 2:  Neural network with only 1 hidden layer with 150 neurons.
# TASK 3:  Neural network with 2 hidden layers with 100 neurons in first hidden layer and 15 neurons in the second hidden layer.
# TASK 4:  Neural network with 2 hidden layer with 500 neurons in first hidden layer and 150 neurons in the second hidden layer.
#
# Based off code from
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/multilayer_perceptron.py

from sys import argv
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#################################################
# COMMAND LINE ARGS
#################################################

num_layers = int(argv[1])
if len(argv) < 3:
    print "Please specify number of neurons in first hidden layer."
    exit()

hid_layer1 = int(argv[2])

if num_layers == 2 and len(argv) != 4:
    print "Please specify number of neurons in second hidden layer."
    exit()

if num_layers == 2:
    hid_layer2 = int(argv[3])


#################################################
# INPUT / OUTPUT
#################################################

INPUT_SIZE = 784         # 16x16 pixel images of handwritten numbers = 784 pixels each
CLASSIFICATION_SIZE = 10 # Numbers 0 to 9

# Image flattened to a vector of 784 pixels
x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
y = tf.placeholder(tf.float32, [None, CLASSIFICATION_SIZE])


#################################################
# MODEL
#################################################
# weight and bias matrices
w = [tf.Variable(tf.random_normal([INPUT_SIZE, hid_layer1]))]

if num_layers == 2:
    w.append(tf.Variable(tf.random_normal([hid_layer1, hid_layer2])))

w_out =  tf.Variable(tf.random_normal([hid_layer2 if num_layers == 2 else hid_layer1, CLASSIFICATION_SIZE]))

b = [tf.Variable(tf.random_normal([hid_layer1]))]

if num_layers == 2:
    b.append(tf.Variable(tf.random_normal([hid_layer2])))

b_out =  tf.Variable(tf.random_normal([CLASSIFICATION_SIZE]))

# HIDDEN LAYERS
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w[0]), b[0]))
layer_2 = None

if num_layers == 2:
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w[1]), b[1]))
model = tf.matmul(layer_2 if layer_2 is not None else layer_1, w_out) + b_out


#################################################
# OPTIMIZATION
#################################################
# Softmax cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


#################################################
# TRAINING
#################################################
# Init and start session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for epoch in range(20):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/100)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch

    print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

#################################################
# TEST
#################################################

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(model,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print "Accuracy: " + str(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
