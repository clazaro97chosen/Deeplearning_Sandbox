import tensorflow as tf
from tensorflow import keras
import numpy as np
import sklearn

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, Y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255, X_train_full[5000:] / 255
X_test = X_test / 255

y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle book",
]


# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28,28]))#converts each input image into a 1D array
# model.add(keras.layers.Dense(300,activation="relu")) # each dense layer manages its own weight matrix
# #also manages a vector of bias terms(one per neuron)
# model.add(keras.layers.Dense(100, activation = "relu"))
# model.add(keras.layers.Dense(10,activation="softmax"))
# alternatively
from tensorflow.keras import initializers

model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(
            300,
            kernel_initializer=initializers.RandomNormal(stddev=0.01),
            activation="relu",
        ),
        keras.layers.Dense(100, kernel_initializer="random_normal", activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)
# keras.utils.plot_model(model, to_file="model.png")

# models summary
# model.summary()

# getting a model's layer by index

hidden1 = model.layers[1]

# all of the parameters of models layers can be accessed using the models get_weights() method
weights, biases = hidden1.get_weights()

# print(weights)
# print(biases)

# you can use a different matrix connection weights (i.e a kernel) by setting kernel_initializer:


#
# Compiling the model
#
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model.compile(
    loss=keras.losses.sparse_categorical_crossentropy,
    optimizer=keras.optimizers.SGD(),
    metrics=[keras.metrics.sparse_categorical_accuracy],
)

#
# Training and Evaluating the model
#

# history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
# afinal results of loss: 0.2434 - sparse_categorical_accuracy: 0.9123 -
# val_loss: 0.3244 - val_sparse_categorical_accuracy: 0.8832
# you could also provide validation_split =0.1 argument
# split the lst 10% of the data before shuffling for validation
# history = model.fit(X_train, y_train, epochs=30, validation_split=0.1)
