import numpy as np
import tensorflow as tf

'''
モデル設定
'''
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))
# print(w)
# def y(x):
#     return sigmoid(np.dot(w, x) + b)
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])

# 誤差関数の定義
y = tf.nn.sigmoid(tf.matmul(x, w) + b)
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))

# 最適化手法の定義
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# 学習結果の定義
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)


'''
モデル学習
'''
# ORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

# セッションの定義
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 学習をする
for epoch in range(200):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })

'''
学習結果の確認
'''
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
print(classified)

# 誤差関数（出力確率）の確認
prob = y.eval(session=sess, feed_dict={
    x: X
})
print(prob)

# 重みづけとバイアスの確認
print('w:', sess.run(w))
print('b:', sess.run(b))




