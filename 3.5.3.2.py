import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
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

model = Sequential()
model.add(Dense(input_dim=M, units=K))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))

minibatch_size = 50
model.fit(X, Y, epochs=20, batch_size=minibatch_size)

X_, Y_ = shuffle(X, Y)
# classes = model.predict_classes(X_[0:10], batch_size=minibatch_size)

results = []
for result in Y_[0:10]:
    results.append(np.argmax(result))
# print(results == classes)

#prob = model.predict_proba(X_[0:10], batch_size=1)

#print(prob)
print(X_[0:10])
modelResults = model.predict(X_[0:10])
print(modelResults)

for i in range(10):
    print(Y_[i])
    print(np.argmax(modelResults[i]))
    print(np.argmax(Y_[i]) == np.argmax(modelResults[i]))



