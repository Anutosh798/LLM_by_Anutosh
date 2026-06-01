import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
fashion_mnist=keras.datasets.fashion_mnist
(X_train_full,Y_train_full),(x_test,y_test)=fashion_mnist.load_data()
# print(X_train_full.shape)
# print(Y_train_full.shape)
# print(Y_train_full[:5])
# print(X_train_full[0])
# print(X_train_full[0].shape)
# print(y_test.shape)
# print(x_test.shape)
# print(X_train_full.dtype)

X_valid,X_train=X_train_full[:5000]/255.,X_train_full[5000:]/255.
Y_valid,Y_train=Y_train_full[:5000],Y_train_full[5000:]
x_test=x_test/255.

# print(len(X_valid))
plt.imshow(X_train[0],cmap="binary")
plt.title(Y_train[0])
plt.xlabel("x axis")
plt.ylabel("y axis")
# plt.axis('off')
plt.show()
