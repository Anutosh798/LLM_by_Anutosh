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
# plt.imshow(X_train[0],cmap="binary")
# plt.title(Y_train[0])
# plt.xlabel("x axis")
# plt.ylabel("y axis")
# # plt.axis('off')
# plt.show()

class_names=["T-shirts/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankel Boot"]
# print(class_names[Y_train[0]])
# print(Y_train[0])# returns 4 means 'coat' in class_names

# print("in x train",x_test.shape)
# print("in x valid",X_valid.shape)

# n_rows=4
# n_cols=10
# plt.figure(figsize=(n_cols*1.2,n_rows*1.2))
# for row in range(n_rows):
#     for col in range(n_cols):
#         index=n_cols*row+col
#         plt.subplot(n_rows,n_cols,index+1)
#         plt.imshow(X_train[index],cmap="binary",interpolation="nearest")
#         plt.axis('off')
#         plt.title(class_names[Y_train[index]],fontsize=12)
# plt.subplots_adjust(wspace=0.2,hspace=0.5)
# plt.show()

Model=keras.models.Sequential()
Model.add(keras.layers.Flatten(input_shape=[28,28]))
Model.add(keras.layers.Dense(300,activation='relu'))
Model.add(keras.layers.Dense(100,activation='relu'))
Model.add(keras.layers.Dense(10,activation='softmax'))

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# print(Model.layers)
# print(Model.summary())

# keras.utils.plot_model(Model,"my_fashion_mnist_model.png",show_shapes=True)

hidden1=Model.layers[1]
print(hidden1.name)

print(Model.get_layer(hidden1.name)  is hidden1)

weights,biases=hidden1.get_weights()
print(weights.shape)
# print(weights)
print(biases)
print(biases.shape)
