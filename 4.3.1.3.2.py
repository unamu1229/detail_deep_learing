import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU

mnist = datasets.fetch_mldata('MNIST original', data_home='.')

n = len(mnist.data)
N = 10000
indices = np.random.permutation(range(n))[:N]
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

'''
モデル設定
'''
print(X)
print(Y)
n_in = len(X[0])
n_hidden = 200
n_out = len(Y[0])

alpha = 0.01

model = Sequential()
model.add(Dense(n_hidden, input_dim=n_in))
model.add(LeakyReLU(alpha=alpha))

model.add(Dense(n_hidden))
model.add(LeakyReLU(alpha=alpha))

model.add(Dense(n_hidden))
model.add(LeakyReLU(alpha=alpha))

model.add(Dense(n_hidden))
model.add(LeakyReLU(alpha=alpha))

model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])

'''
モデル学習
'''
epochs = 1000
batch_size = 100

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

'''
予測精度の評価
'''
loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)
