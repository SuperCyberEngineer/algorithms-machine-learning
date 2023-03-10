# prepare dataset

NUM_OF_SAMPLES = 1000

NUM_OF_EPOCH = 1000

NN_DIMENSION = 10

import numpy as np

import pandas as pd

# construct 2d manifoid of gassian

import tensorflow as tf

import tensorflow_probability as tfp

tfd = tfp.distributions

# Define a single scalar Normal distribution.
dist = tfd.Normal(loc=0., scale=3.)

# Evaluate the cdf at 1, returning a scalar.
# o = dist.cdf(0.)
# print(o)

# Define a batch of two scalar valued Normals.
# The first has mean 1 and standard deviation 11, the second 2 and 22.
# dist = tfd.Normal(loc=[1, 2.], scale=[11, 22.])
dist = tfd.Normal(loc=0, scale=30)

# Evaluate the pdf of the first distribution on 0, and the second on 1.5,
# returning a length two tensor.

# Get 3 samples, returning a 3 x 2 tensor.
X = dist.sample(NUM_OF_SAMPLES)
Y = dist.prob(X)

print('Y')
print(Y)

import matplotlib.pyplot as plt

fig = plt.figure(1)  #identifies the figure
plt.title("gassian distribution", fontsize='10')  #title
plt.scatter(Y, Y, s=3, color='hotpink')  #plot the points
plt.xlabel("Y", fontsize='10')  #adds a label in the x axis
plt.grid()  #shows a grid under the plot
# plt.show()

# plot
# import matplotlib.pyplot as plt

# import numpy as np

# fig = plt.figure(1)  #identifies the figure
# plt.title("gassian distribution", fontsize='10')  #title
# plt.scatter(X, Y, s=3)  #plot the points
# plt.xlabel("X", fontsize='10')  #adds a label in the x axis
# plt.grid()  #shows a grid under the plot
# plt.show()

# training

# construct the encoder decoder
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            # layers.Dense(1, activation='relu'),
            layers.Dense(NN_DIMENSION, activation='relu'),
            layers.Dense(NN_DIMENSION, activation='relu'),
            layers.Dense(NN_DIMENSION, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
    def call(self, x):
        encoded = self.encoder(x)
        return encoded


latent_dim = 1

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanAbsoluteError())

history = autoencoder.fit(Y,
                          Y,
                          epochs=NUM_OF_EPOCH,
                          shuffle=True,
                          validation_split=0.2)

# XX = tf.reshape(X, [1, len(X)])
YY = []

for i in range(0, len(X)):
    y = autoencoder.encoder(np.array([[X[0]]])).numpy()
    YY.append(y)

YY = [e[0] for e in YY]
# print(X)
# print(YY)
# plot result
# import matplotlib.pyplot as plt

# fig = plt.figure(1)  #identifies the figure
# plt.title("gassian distribution", fontsize='10')  #title
plt.scatter(Y, YY, s=3, color = 'blue')  #plot the points
plt.grid()  #shows a grid under the plot
plt.show()
