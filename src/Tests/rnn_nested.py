import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import layers as l
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, _config_for_enable_caching_device, \
    _caching_device
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.keras import models as m
from tensorflow.keras import activations as a
from src.modules import utils, losses, models
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse


class SimpleRnn(DropoutRNNCellMixin, l.Layer):
    def __init__(self, units=10, activation="relu", dropout=0.1, recurrent_dropout=0.1, **kwargs):
        super(SimpleRnn, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.state_size = tf.TensorShape([units])
        self.output_size = tf.TensorShape([units])
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self._enable_caching_device = kwargs.pop('enable_caching_device', True)

    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        self.W_x = self.add_weight(shape=(input_shape[-1], self.units), name="W_x", initializer="glorot_normal",
                                   caching_device=default_caching_device)
        self.W_h = self.add_weight(shape=(self.units, self.units), name="W_h", initializer="glorot_normal",
                                   caching_device=default_caching_device)
        self.b = self.add_weight(shape=(self.units,), name="b", initializer="glorot_normal",
                                 caching_device=default_caching_device)

    def call(self, inputs, states, training, *args, **kwargs):
        h = states[0]
        if 0 < self.dropout < 1:
            inputs_mask = self.get_dropout_mask_for_cell(inputs=inputs, training=training, count=1)
            inputs = inputs * inputs_mask
        if 0 < self.recurrent_dropout < 1:
            h_mask = self.get_recurrent_dropout_mask_for_cell(inputs=h, training=training, count=1)
            h = h * h_mask
        z = inputs @ self.W_x + h @ self.W_h + self.b
        z = a.get(self.activation)(z)
        return z, z

    def get_config(self):
        config = {'dropout': self.dropout,
                'recurrent_dropout': self.recurrent_dropout,
                'units': self.units,
                'activation': self.activation}
        config.update(_config_for_enable_caching_device(self))
        base_config = super(SimpleRnn, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    os.chdir("../../")
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", action="store_true")
    args = parser.parse_args()
    keras = args.k
    '''
    a, x, N, t = utils.load_dataset()
    f = N
    a = tf.transpose(a, perm=(1, 0, 2, 3))
    x = tf.transpose(x, perm=(1, 0, 2, 3))
    #cell = models.NestedCell(N, f, 5, 5, 10)
    #cell = models.NestedCell2(N, f, 5, 5, 10)
    cell = models.NestedCell3(N, f, 5, 5, 10)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=False)
    input_1 = tf.keras.Input((None, N, f))
    input_2 = tf.keras.Input((None, N, N))
    outputs = rnn((input_1, input_2))
    model = tf.keras.models.Model([input_1, input_2], outputs)
    o = model([x, a], training=False)
    # print(o.shape)
    model.compile(optimizer="Adam",
                  loss=lambda true,pred : tf.reduce_mean(tf.math.square(true-pred), (-1, -2, -3)))
    history = model.fit([a, x], epochs=100)
    o = model([x, a])
    history = history.history["loss"]
    
    history = []
    epochs = 50
    optimizer = tf.keras.optimizers.Adam(0.001)
    for e in range(epochs):
        with tf.GradientTape(persistent=True) as tape:
            o = model([x, a])
            loss = losses.zero_inflated_lognormal_loss(tf.expand_dims(a, -1), o)
            history.append(loss)
            print(loss.numpy())
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    plt.plot(history)
    ln = utils.zero_inflated_lognormal(o)
    samples = ln.sample(1)
    samples = tf.squeeze(samples)
    samples = tf.clip_by_value(tf.math.log(samples), 0,1e10)
    a = tf.squeeze(a)
    a = tf.clip_by_value(tf.math.log(a), 0, 1e10)
    fig, axs = plt.subplots(1,2)
    axs[0].set_title("Pred")
    axs[1].set_title("True")

    def update(i):
        fig.suptitle(f"Time {i}")
        axs[0].cla()
        axs[1].cla()
        axs[0].imshow(samples[i])
        axs[1].imshow(a[i])
    anim = animation.FuncAnimation(fig, update, 20, repeat=True)
    plt.show()
    '''
    i1 = k.Input((None, 8), batch_size=10)
    if not keras:
        cell = SimpleRnn(100, activation="tanh")
        cell1 = SimpleRnn(100, activation="tanh")
        cell2 = SimpleRnn(10, activation="tanh")
        output = l.TimeDistributed(l.Dense(8, activation="tanh"))
        rnn = l.RNN([cell, cell1, cell2], return_sequences=True, stateful=True)
        o_rnn = rnn(i1)
        o = output(o_rnn)
        model = m.Model(i1, o)
        model.compile("Adam", "mse")
    else:
        cell_k = l.SimpleRNNCell(100, activation="tanh", recurrent_dropout=0.2)
        cell_k0 = l.SimpleRNNCell(100, activation="tanh", recurrent_dropout=0.2)
        cell_k1 = l.SimpleRNNCell(10, activation="tanh", recurrent_dropout=0.2)
        output = l.TimeDistributed(l.Dense(8, activation="tanh"))
        rnn_k = l.RNN([cell_k, cell_k0, cell_k1], return_sequences=True, stateful=False)(i1)
        o = output(rnn_k)
        model = m.Model(i1, o)
        model.compile("Adam", "mse")
        # lambda y, y_pred: tf.math.sqrt(tf.reduce_mean(tf.math.square(y - y_pred)), (1,2))
    xt = []
    x = np.linspace(0, 5 * np.pi, 100)
    for i in range(1, 5):
        x1 = np.cos(x * 0.5 * np.pi / i)[:, None]
        x2 = np.sin(x * 0.5 * np.pi / i)[:, None]
        xt.append(x1)
        xt.append(x2)
    xt = np.concatenate(xt, axis=-1)
    x_batched = []
    for i in range(0, 100, 10):
        x_batch = xt[i:i + 10]
        x_batched.append(x_batch)
    x_batched = np.asarray(x_batched)
    history = model.fit(x_batched[:, :-1, :], x_batched[:, 1:, :], epochs=300)
    history = history.history["loss"]
    plt.plot(history)
    y = model.predict(x_batched)
    y = y.reshape(100, 8)
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(xt[:, :4])
    axs[0].set_title("True")
    axs[1].plot(y[:, :4])
    axs[1].set_title("Pred")
    plt.show()
