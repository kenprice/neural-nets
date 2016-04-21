# TASK 3:  Neural network with 2 hidden layers with 100 neurons in first hidden layer and 15 neurons in the second hidden layer.
# Credit: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/multilayer_perceptron.py#L22

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

INPUT_SIZE = 784         # 16x16 pixel images of handwritten numbers = 784 pixels each
CLASSIFICATION_SIZE = 10 # Numbers 0 to 9
HID_LAYER_1 = 100
HID_LAYER_2 = 15

#################################################
# INPUT / OUTPUT
#################################################
# Image flattened to a vector of 784 pixels
x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
# Image flattened to a vector of 784 pixels
y = tf.placeholder(tf.float32, [None, CLASSIFICATION_SIZE])


#################################################
#MODEL
#################################################
# weight and bias matrices
w       = [tf.Variable(tf.random_normal([INPUT_SIZE,                 HID_LAYER_1])),
           tf.Variable(tf.random_normal([HID_LAYER_1,                HID_LAYER_2]))]
w_out   =  tf.Variable(tf.random_normal([HID_LAYER_2,                CLASSIFICATION_SIZE]))

b       = [tf.Variable(tf.random_normal([HID_LAYER_1])),
           tf.Variable(tf.random_normal([HID_LAYER_2]))]
b_out   =  tf.Variable(tf.random_normal([CLASSIFICATION_SIZE]))

# Hidden layers
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w[0]), b[0]))
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w[1]), b[1]))
model = tf.matmul(layer_2, w_out) + b_out

# OPTIMIZATION
# Softmax cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))
# Other optimization methods can be used besides Gradient Descent
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost) # Adam Optimizer

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
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch

    print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)


# TEST Measuring effectiveness
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(model,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

# OUTPUT: 0.5295        GradientDescentOptimizer using tf.nn.relu
# OUTPUT 0.618          AdamOptimizer using tf.nn.relu

# OUTPUT 0.949          AdamOptimizer using tf.nn.sigmoid
# OUTPUT 0.5657         GradientDescentOptimizer using using tf.nn.sigmoid
