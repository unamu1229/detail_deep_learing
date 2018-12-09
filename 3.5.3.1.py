import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

M = 2
K = 3
n = 100
N = n * K

X1 = np.random.randn(n, M) + np.array([0, 10])
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])

Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)

W = tf.Variable(tf.zeros([M, K]))
b = tf.Variable(tf.zeros([K]))

x = tf.placeholder(tf.float32, shape=[None, M])
t = tf.placeholder(tf.float32, shape=[None, K])
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
# test = tf.constant([[1, 1, 1], [3, 3, 3]])
# sess = tf.Session()
# calc = sess.run(tf.reduce_sum(test, reduction_indices=[1]))
# print(calc)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))

batch_size = 50
n_batches = N

'''
モデル学習
'''
# 初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(20):
    # X_, Y_ = shuffle(X, Y)
    X_ = X
    Y_ = Y

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end]
        })

# X_, Y_ = shuffle(X, Y)
X_ = X
Y_ = Y

classified = correct_prediction.eval(session=sess, feed_dict={
    x: X_[0:10],
    t: Y_[0:10]
})

prob = y.eval(session=sess, feed_dict={
    x: X_[0:10]
})

print(classified)

print(prob)

print(sess.run(W))

print(sess.run(b))




