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
from src.modules.utils import compute_sigma


class SimpleRnn(DropoutRNNCellMixin, l.Layer):
    def __init__(self, units=10, activation="relu", dropout=0.2, recurrent_dropout=0.2, **kwargs):
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
    parser.add_argument("--gat", action="store_true")
    parser.add_argument("--stateful", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--logits", action="store_true")
    parser.add_argument("--seq_len", type=int, default=6)
    args = parser.parse_args()
    keras = args.k
    gat = args.gat
    stateful = args.stateful
    dropout = args.dropout
    logits = args.logits
    seq_len = args.seq_len

    if gat:
        a, x, N, t = utils.load_dataset(baci=True)
        f = x.shape[-1]
        a = tf.transpose(a, perm=(1, 0, 2, 3))  # BxTxNxN
        x = tf.transpose(x, perm=(1, 0, 2, 3))  # # BxTxNxf
        input_1 = tf.keras.Input((None, N, f))
        input_2 = tf.keras.Input((None, N, N))
        if logits:
            if stateful:
                batches_a = np.empty(shape=(1, seq_len, N, N))
                batches_x = np.empty(shape=(1, seq_len, N, N))
                for i in range(0, t-seq_len, seq_len):
                    batches_a = np.concatenate([batches_a, a[:1, i:i + seq_len, :]], axis=0)
                    batches_x = np.concatenate([batches_x, x[:1, i:i + seq_len, :]], axis=0)
                a = np.concatenate([batches_a, a[:1, t-seq_len:, :]], axis=0)[1:]
                x = np.concatenate([batches_x, x[:1, t-seq_len:, :]], axis=0)[1:]
                input_1 = tf.keras.Input((None, N, f), batch_size=a.shape[0])
                input_2 = tf.keras.Input((None, N, N), batch_size=a.shape[0])

            cell = models.RecurrentEncoderDecoder(nodes=N,
                                                  features=f,
                                                  channels=5,
                                                  attention_heads=8,
                                                  hidden_size=25,
                                                  dropout=dropout,
                                                  recurrent_dropout=dropout)
            rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=False, stateful=stateful)
            outputs = rnn((input_1, input_2))
            model = tf.keras.models.Model([input_1, input_2], outputs)
            model.compile(optimizer="rmsprop",
                          loss=lambda true, logits:
                          tf.reduce_sum(
                              losses.zero_inflated_lognormal_loss(tf.expand_dims(true, -1),
                                                                  logits,
                                                                  reduce_axis=(-1, -2, -3)
                                                                  ),
                              axis=1)
                          )
            history = model.fit(x=[x[:, :-1, :], a[:, :-1, :]], y=a[:, 1:, :], epochs=150, verbose=1)
            o = model([x, a])
            o = tf.reshape(o, (-1, N, N, 3))
            history = history.history["loss"]
            plt.plot(history)
            ln = utils.zero_inflated_lognormal(o)
            samples = ln.sample(1)
            samples = tf.squeeze(samples)
            samples = tf.clip_by_value(tf.math.log(samples), 0, 1e10)
            a = tf.reshape(a, (-1, N, N))
            a = tf.clip_by_value(tf.math.log(a), 0, 1e10)

            fig_logits, ax_logits = plt.subplots(1, 3)


            def update_p(i):
                fig_logits.suptitle(f"Time {i}")
                ax_logits[0].cla()
                ax_logits[1].cla()
                ax_logits[2].cla()
                ax_logits[0].imshow(tf.nn.sigmoid(o[i, :, :, 0]).numpy())
                ax_logits[1].imshow(o[i, :, :, 1].numpy())
                ax_logits[2].imshow(compute_sigma(o[i, :, :, 2]).numpy())
                ax_logits[0].set_title("P")
                ax_logits[1].set_title("Mu")
                ax_logits[2].set_title("Sigma")


            anim_p = animation.FuncAnimation(fig_logits, update_p, o.shape[0] - 2, repeat=True)

            fig, axs = plt.subplots(1, 2)
            axs[0].set_title("Pred")
            axs[1].set_title("True")


            def update(i):
                fig.suptitle(f"Time {i}")
                axs[0].cla()
                axs[1].cla()
                axs[0].imshow(samples[i])
                axs[1].imshow(a[i])


            anim = animation.FuncAnimation(fig, update, samples.shape[0] - 2, repeat=True)
            plt.show()
        else:
            cell = models.RecurrentEncoderDecoder2(nodes=N,
                                                   features=f,
                                                   channels=5,
                                                   attention_heads=8,
                                                   hidden_size=25,
                                                   dropout=dropout,
                                                   recurrent_dropout=dropout)
            rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=False, stateful=stateful)
            outputs = rnn((input_1, input_2))
            model = tf.keras.models.Model([input_1, input_2], outputs)
            model.compile(optimizer="rmsprop",
                          loss=lambda true, pred:
                          tf.math.reduce_sum(
                              tf.math.sqrt(
                                  tf.reduce_mean(
                                      tf.math.square(true - pred)
                                  ), axis=(-1, -2)
                              )
                          )
                          )
            history = model.fit(x=[x[:, :-1, :],a[:, :-1, :]], y=a[:, 1:, :], epochs=250, verbose=1)
            history = history.history["loss"]
            samples = tf.squeeze(model([x,a])).numpy()
            fig, axs = plt.subplots(1,2)
            def update(i):
                fig.suptitle(f"Time {i}")
                axs[0].cla()
                axs[1].cla()
                axs[0].imshow(samples[i])
                axs[1].imshow(a[i])


            anim = animation.FuncAnimation(fig, update, samples.shape[0] - 2, repeat=True)
            plt.show()
    else:
        if stateful:
            i1 = k.Input((None, 8), batch_size=10)
        else:
            i1 = k.Input((None, 8))
        if not keras:
            cell = SimpleRnn(100, activation="tanh", recurrent_dropout=dropout, dropout=dropout)
            cell1 = SimpleRnn(100, activation="tanh", recurrent_dropout=dropout, dropout=dropout)
            cell2 = SimpleRnn(10, activation="tanh", recurrent_dropout=dropout, dropout=dropout)
            output = l.TimeDistributed(l.Dense(8, activation="tanh"))
            rnn = l.RNN([cell, cell1, cell2], return_sequences=True, stateful=stateful)
            o_rnn = rnn(i1)
            o = output(o_rnn)
            model = m.Model(i1, o)
            model.compile("Adam", "mse")
        else:
            cell_k = l.SimpleRNNCell(100, activation="tanh", recurrent_dropout=dropout, dropout=dropout)
            cell_k0 = l.SimpleRNNCell(100, activation="tanh", recurrent_dropout=dropout, dropout=dropout)
            cell_k1 = l.SimpleRNNCell(10, activation="tanh", recurrent_dropout=dropout, dropout=dropout)
            output = l.TimeDistributed(l.Dense(8, activation="tanh"))
            rnn_k = l.RNN([cell_k, cell_k0, cell_k1], return_sequences=True, stateful=stateful)(i1)
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