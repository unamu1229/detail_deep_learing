import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

mnist = datasets.fetch_mldata('MNIST original', data_home='.')

n = len(mnist.data)
N = 50000
indices = np.random.permutation(range(n))[:N]

train_size = 0.8

X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]

N_train = 20000
N_validation = 4000


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=N_train)

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=N_validation)



'''
モデル設定
'''
n_in = len(X[0])
n_hidden = 200
n_out = len(Y[0])


x = tf.placeholder(tf.float32, shape=[None, n_in])
t = tf.placeholder(tf.float32, shape=[None, n_out])
keep_prob = tf.placeholder(tf.float32)


W0 = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=0.01))
b0 = tf.Variable(tf.zeros([n_hidden]))
h0 = tf.nn.relu(tf.matmul(x, W0) + b0)
h0_drop = tf.nn.dropout(h0, keep_prob)


W1 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
b1 = tf.Variable(tf.zeros([n_hidden]))
h1 = tf.nn.relu(tf.matmul(h0, W1) + b1)
h1_drop = tf.nn.dropout(h1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
b2 = tf.Variable(tf.zeros([n_hidden]))
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
h2_drop = tf.nn.dropout(h2, keep_prob)


W3 = tf.Variable(tf.truncated_normal([n_hidden, n_out], stddev=0.01))
b3 = tf.Variable(tf.zeros([n_out]))
y = tf.nn.softmax(tf.matmul(h2_drop, W3) + b3)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), axis=1))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
モデル学習
'''
epochs = 50
batch_size = 200

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

n_batches = (int)(N * train_size)

history = {
    'val_loss': [],
    'val_acc': []
}

for epoch in range(epochs):
    X_, Y_ = shuffle(X_train, Y_train)

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end],
            keep_prob: 0.5
        })

    loss = cross_entropy.eval(session=sess, feed_dict={
        x: X_,
        t: Y_,
        keep_prob: 1.0
    })
    acc = accuracy.eval(session=sess, feed_dict={
        x: X_,
        t: Y_,
        keep_prob: 1.0
    })

    print('epoch:', epoch, ' loss:', loss, ' accuracy:', acc)

    var_loss = cross_entropy.eval(session=sess, feed_dict={
        x: X_validation,
        t: Y_validation,
        keep_prob: 1.0
    })

    history['val_loss'].append(var_loss)

    val_acc = accuracy.eval(session=sess, feed_dict={
        x: X_validation,
        t: Y_validation,
        keep_prob: 1.0
    })
    history['val_acc'].append(val_acc)


accuracy_rate = accuracy.eval(session=sess, feed_dict={
    x: X_test,
    t: Y_test,
    keep_prob: 1.0
})

print('accuracy: ', accuracy_rate)

plt.rc('font', family='serif')
fig = plt.figure()

plt.plot(range(epochs), history['val_acc'], label='acc', color='black')

plt.xlabel('epochs')
plt.ylabel('validation loss')

plt.show()


