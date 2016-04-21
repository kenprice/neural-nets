# TASK 1:  Neural network with only 1 hidden layer with 15 neurons.
# TASK 2:  Neural network with only 1 hidden layer with 150 neurons.
# TASK 3:  Neural network with 2 hidden layers with 100 neurons in first hidden layer and 15 neurons in the second hidden layer.
# TASK 4:  Neural network with 2 hidden layer with 500 neurons in first hidden layer and 150 neurons in the second hidden layer.

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

INPUT_SIZE = 784         # 16x16 pixel images of handwritten numbers = 784 pixels each
CLASSIFICATION_SIZE = 10 # Numbers 0 to 9
NEURONS_IN_HIDDEN_LAYER = 15

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
w       = tf.Variable(tf.random_normal([INPUT_SIZE,                 NEURONS_IN_HIDDEN_LAYER]))
w_out   = tf.Variable(tf.random_normal([NEURONS_IN_HIDDEN_LAYER,    CLASSIFICATION_SIZE]))
b       = tf.Variable(tf.random_normal([NEURONS_IN_HIDDEN_LAYER]))
b_out   = tf.Variable(tf.random_normal([CLASSIFICATION_SIZE]))

layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w), b)) #Hidden layer with RELU activation

model = tf.matmul(layer_1, w_out) + b_out


# OPTIMIZATION
# Softmax cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))
# Other optimization methods can be used besides Gradient Descent
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
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

#OUTPUT 0.7029              GradientDescentOptimizer using tf.nn.relu
#OUTPUT 0.9183              AdamOptimizer using tf.nn.sigmoid
