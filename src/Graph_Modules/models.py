import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.keras import models as m
from tensorflow.keras import activations
from tensorflow.keras import initializers
import tensorflow.keras.backend as k
from src.Graph_Modules import layers


class GraphRNN(m.Model):
    def __init__(self, encoder, decoder, nodes_embedding_dim, **kwargs):
        super(GraphRNN, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.mu_activation = "tanh"
        self.sigma_activation = "relu"
        self.dropout_rate = kwargs.get("dropout_rate")
        self.embedding_size = kwargs.get("embedding_size")
        self.nodes_embedding_dim = nodes_embedding_dim

        if self.encoder == "NTN":
            self.encoder = layers.NTN

        # self.prior_mean =
        self.encoder_mean = self.encoder(10, self.mu_activation)
        self.encoder_sigma = self.encoder(10, self.sigma_activation)

    def call(self, inputs, **kwargs):
        X, R, A = inputs


class GAHRT_Model(m.Model):
    def __init__(self, nodes_features_dim, relations_features_dim, **kwargs):
        super(GAHRT_Model, self).__init__(**kwargs)
        self.encoder = layers.RGHAT(nodes_features_dim, relations_features_dim)
        self.decoder = layers.TrippletDecoder()

    def call(self, inputs, **kwargs):
        enc = self.encoder(inputs)
        dec = activations.relu(self.decoder(enc))
        self.add_loss(tf.reduce_sum(tf.square(dec)))
        return dec


@tf.function
def square_loss(y_true, y_pred):
    print(y_true)
    print(y_pred)
    diff = tf.square(y_true.values - y_pred)
    return tf.reduce_sum(diff)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    nodes = 20
    ft = 10
    r = 5

    A = layers.make_data(1, 15, 20)
    A0 = tf.sparse.slice(A, (0, 0, 0, 0), (1, r, nodes, nodes))
    A0_dense = tf.sparse.to_dense(A0)
    A0_squeeze = tf.squeeze(A0_dense, axis=0)
    A = tf.sparse.from_dense(A0_squeeze)
    R = tf.Variable(np.random.normal(size=(r, ft)), dtype=tf.float32)
    X = tf.Variable(np.random.normal(size=(nodes, ft)), dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    epochs = 20
    model = GAHRT_Model(10, 10)
    out = model([X, R, A])
    print(out)
    print(A.values)
    losses = []
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            output = model([X, R, A])
            loss = square_loss(A, output)
            losses.append(loss)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    plt.plot(losses)

    A_dense = tf.sparse.to_dense(A)
    A_pred_dense = tf.sparse.to_dense(tf.sparse.SparseTensor(A.indices, output, A.shape))

    f, axes = plt.subplots(A.shape[0], 3)
    for i in range(A.shape[0]):
        axes[i, 0].imshow(A_dense[i, :, :], cmap="winter")
        axes[i, 1].imshow(A_pred_dense[i, :, :], cmap="winter")
        img = axes[i, 2].imshow(A_dense[i, :, :]-A_pred_dense[i, :, :], cmap="winter")
        f.colorbar(img, ax=axes[i, 2])
    plt.tight_layout()
    plt.show()
