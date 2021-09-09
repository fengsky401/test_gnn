from networkx import karate_club_graph,to_numpy_matrix
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf

import matplotlib.pyplot as plt
tf.disable_eager_execution()
zkc = karate_club_graph()
order=sorted(list(zkc.nodes()))
NODE_SIZE=len(order)

print(NODE_SIZE)

A=to_numpy_matrix(zkc,nodelist=order)
I=np.eye(zkc.number_of_nodes())

node_label = []
for i in range(34):
        label = zkc.nodes[i]
        if label['club'] == 'Officer':
                node_label.append(1)
        else:
                node_label.append(0)


#Step2: Parameter Settings
NODE_SIZE = 34
NODE_FEATURE_DIM = 34
HIDDEN_DIM1 = 10
num_classes = 2
training_epochs = 100
step = 10
lr=0.1


#Step3: network define
X = tf.placeholder(tf.float32, shape=[NODE_SIZE, NODE_FEATURE_DIM])
Y = tf.placeholder(tf.int32, shape=[NODE_SIZE])
#label = tf.one_hot(Y, num_classes)
Y_enc = tf.one_hot(Y, 2)   #就是Label
adj = tf.placeholder(tf.float32, shape=[NODE_SIZE, NODE_SIZE])
weights = {"hidden1": tf.Variable(tf.random_normal(dtype=tf.float32, shape=[NODE_FEATURE_DIM, HIDDEN_DIM1]), name='w1'),
           "hidden2": tf.Variable(tf.random_normal(dtype=tf.float32, shape=[HIDDEN_DIM1, num_classes]), 'w2')}

D_hat = tf.matrix_inverse(tf.matrix_diag(tf.reduce_sum(adj, axis=0)))
l1 = tf.matmul(tf.matmul(tf.matmul(D_hat, adj), X), weights['hidden1'])
output = tf.matmul(tf.matmul(tf.matmul(D_hat, adj), l1), weights['hidden2'])

#Step4 : define loss func and train
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_enc, logits=output))
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
init_op = tf.global_variables_initializer()
feed_dict = {adj: A, X: I, Y: node_label}
with tf.Session() as sess:
    sess.run(init_op)
    # dynamic display
    plt.ion()
    for epoch in range(training_epochs):
        c, _ = sess.run([loss, train_op], feed_dict)
        if epoch % step == 0:
            print(f'Epoch:{epoch} Loss {c}')

        represent = sess.run(output, feed_dict)
        plt.scatter(represent[:, 0], represent[:, 1], s=200, c=node_label)
        plt.pause(0.1)
        plt.cla()

