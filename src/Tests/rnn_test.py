import numpy as np
import tensorflow as tf
from src.modules.layers import GCNDirected
import tensorflow.keras.layers as l
import tensorflow.keras.models as m
import tensorflow.keras.activations as act
import tensorflow.keras.optimizers as opt
import tensorflow.keras.backend as k
from spektral.layers import GATConv, GCNConv
from spektral.utils.convolution import gcn_filter
from src.modules.layers import BatchBilinearDecoderDense
import pickle as pkl
import os
from src.modules.losses import zero_inflated_lognormal_loss
from src.modules.utils import zero_inflated_lognormal
import argparse
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
import matplotlib as mplt
import matplotlib.pyplot as plt
from src.graph_data_loader import plot_names
from scipy import stats
import tqdm
import datetime
import matplotlib.animation as animation
from networkx import in_degree_centrality, out_degree_centrality
from networkx.convert_matrix import from_numpy_matrix
import networkx as nx


class GRUGAT(l.Layer):
    def __init__(self, hidden_size=10, attn_heads=10, dropout=0.2, hidden_activation="relu"):
        super(GRUGAT, self).__init__()
        self.gnn_u = GATConv(channels=hidden_size // 2, attn_heads=attn_heads, concat_heads=True,
                             activation=hidden_activation, dropout_rate=dropout)
        self.gnn_r = GATConv(channels=hidden_size // 2, attn_heads=attn_heads, concat_heads=True,
                             activation=hidden_activation, dropout_rate=dropout)
        self.gnn_c = GATConv(channels=hidden_size // 2, attn_heads=attn_heads, concat_heads=True,
                             activation=hidden_activation, dropout_rate=dropout)

        self.hidden_activation = hidden_activation
        self.hidden_size = (hidden_size // 2) * attn_heads
        self.drop = l.Dropout(dropout)
        # self.state_size = self.hidden_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        x, a = inputs
        return tf.zeros(shape=(*x.shape[:-1], self.hidden_size))

    def build(self, input_shape):
        self.b_u = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal", name="b_u")
        self.b_r = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal", name="b_r")
        self.b_c = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal", name="b_c")
        self.W_u = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal",
                                   name="W_u")
        self.W_r = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal",
                                   name="W_r")
        self.W_c = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal",
                                   name="W_c")

    def call(self, inputs, state, training, *args, **kwargs):
        x, a = inputs
        # Encoding
        if state is None:
            h = self.get_initial_state(inputs)
        else:
            h = state

        conv_u = self.gnn_u(inputs, training=training)  # B x N x d
        conv_r = self.gnn_r(inputs, training=training)  # B x N x d
        conv_c = self.gnn_c(inputs, training=training)  # B x N x d

        # Recurrence
        u = tf.nn.sigmoid(self.b_u + tf.concat([conv_u, h], -1) @ self.W_u)
        r = tf.nn.sigmoid(self.b_r + tf.concat([conv_r, h], -1) @ self.W_r)
        c = tf.nn.tanh(self.b_c + tf.concat([conv_c, r * h], -1) @ self.W_c)
        h_prime = u * h + (1 - u) * c
        h_prime = self.drop(h_prime, training=training)
        return h_prime


class GRUGCN(l.Layer):
    def __init__(self, hidden_size=10, dropout=0.2, hidden_activation="relu"):
        super(GRUGCN, self).__init__()
        self.gnn_u = GCNConv(channels=hidden_size, concat_heads=True,
                             activation=hidden_activation, dropout_rate=dropout)
        self.gnn_r = GCNConv(channels=hidden_size, concat_heads=True,
                             activation=hidden_activation, dropout_rate=dropout)
        self.gnn_c = GCNConv(channels=hidden_size, concat_heads=True,
                             activation=hidden_activation, dropout_rate=dropout)

        self.hidden_activation = hidden_activation
        self.hidden_size = hidden_size
        self.drop = l.Dropout(dropout)
        self.state_size = self.hidden_size

    def get_initial_state(self, inputs):
        x, a = inputs
        self.state_size = (*x.shape[:-1], self.hidden_size)
        return tf.zeros(shape=(*self.state_size,))

    def build(self, input_shape):
        self.b_u = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal")
        self.b_r = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal")
        self.b_c = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal")
        self.W_u = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal")
        self.W_r = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal")
        self.W_c = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal")

    def call(self, inputs, states, training, *args, **kwargs):
        x, a = inputs
        if states is None:
            h = self.get_initial_state(inputs)
        else:
            h = states

        # Encoding
        a = gcn_filter(a.numpy(), symmetric=False)
        inputs = [a, inputs[1]]
        conv_u = self.gnn_u(inputs, training=training)
        conv_r = self.gnn_r(inputs, training=training)
        conv_c = self.gnn_c(inputs, training=training)

        # Recurrence
        u = tf.nn.sigmoid(self.b_u + tf.concat([conv_u, h], -1) @ self.W_u)
        r = tf.nn.sigmoid(self.b_r + tf.concat([conv_r, h], -1) @ self.W_r)
        c = tf.nn.tanh(self.b_c + tf.concat([conv_c, r * h], -1) @ self.W_c)
        h_prime = u * h + (1 - u) * c
        h_prime = self.drop(h_prime, training=training)
        return h_prime


class GRUGCNDirected(l.Layer):
    def __init__(self, hidden_size=10, dropout=0.2, hidden_activation="relu"):
        super(GRUGCNDirected, self).__init__()
        self.gnn_u = GCNDirected(hidden_size=hidden_size, activation=hidden_activation, dropout_rate=dropout)
        self.gnn_r = GCNDirected(hidden_size=hidden_size, activation=hidden_activation, dropout_rate=dropout)
        self.gnn_c = GCNDirected(hidden_size=hidden_size, activation=hidden_activation, dropout_rate=dropout)

        self.hidden_activation = hidden_activation
        self.hidden_size = hidden_size
        self.drop = l.Dropout(dropout)
        self.state_size = self.hidden_size

    def get_initial_state(self, inputs):
        x, a = inputs
        self.state_size = (*x.shape[:-1], self.hidden_size)
        return tf.zeros(shape=(*self.state_size,))

    def build(self, input_shape):
        self.b_u = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal")
        self.b_r = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal")
        self.b_c = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal")
        self.W_u = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal")
        self.W_r = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal")
        self.W_c = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal")

    def call(self, inputs, states, training, *args, **kwargs):
        x, a = inputs
        if states is None:
            h = self.get_initial_state(inputs)
        else:
            h = states

        # Encoding
        conv_u = self.gnn_u(inputs, training=training)
        conv_r = self.gnn_r(inputs, training=training)
        conv_c = self.gnn_c(inputs, training=training)

        # Recurrence
        u = tf.nn.sigmoid(self.b_u + tf.concat([conv_u, h], -1) @ self.W_u)
        r = tf.nn.sigmoid(self.b_r + tf.concat([conv_r, h], -1) @ self.W_r)
        c = tf.nn.tanh(self.b_c + tf.concat([conv_c, r * h], -1) @ self.W_c)
        h_prime = u * h + (1 - u) * c
        h_prime = self.drop(h_prime, training=training)
        return h_prime


class GRU(l.Layer):
    def __init__(self, hidden_size=10, dropout=0.2, hidden_activation="relu"):
        super(GRU, self).__init__()
        self.enc_u = l.Dense(hidden_size, activation=hidden_activation)
        self.enc_r = l.Dense(hidden_size, activation=hidden_activation)
        self.enc_c = l.Dense(hidden_size, activation=hidden_activation)
        self.hidden_activation = hidden_activation
        self.drop = l.Dropout(dropout)
        self.hidden_size = hidden_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        x, a = inputs
        return tf.zeros(shape=(*x.shape[:-1], self.hidden_size))

    def build(self, input_shape):
        self.b_u = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal", name="b_u")
        self.b_r = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal", name="b_r")
        self.b_c = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal", name="b_c")
        self.W_u = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal",
                                   name="W_u")
        self.W_r = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal",
                                   name="W_r")
        self.W_c = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal",
                                   name="W_c")

    def call(self, inputs, state, training, *args, **kwargs):
        x, a = inputs
        # Encoding
        if state is None:
            h = self.get_initial_state(inputs)
        else:
            h = state

        conv_u = self.enc_u(x, training=training)  # B x N x d
        conv_r = self.enc_r(x, training=training)  # B x N x d
        conv_c = self.enc_c(x, training=training)  # B x N x d

        # Recurrence
        u = tf.nn.sigmoid(self.b_u + tf.concat([conv_u, h], -1) @ self.W_u)
        r = tf.nn.sigmoid(self.b_r + tf.concat([conv_r, h], -1) @ self.W_r)
        c = tf.nn.tanh(self.b_c + tf.concat([conv_c, r * h], -1) @ self.W_c)
        h_prime = u * h + (1 - u) * c
        h_prime = self.drop(h_prime, training=training)
        return h_prime


class GRUGATLognormal(m.Model):
    def __init__(self, hidden_size=4, attn_heads=4, dropout=0.2, hidden_activation="relu"):
        super(GRUGATLognormal, self).__init__()
        # Encoders
        self.GatRnn_p = GRUGAT(hidden_size=hidden_size, attn_heads=attn_heads, dropout=dropout,
                               hidden_activation=hidden_activation)
        self.GatRnn_mu = GRUGAT(hidden_size=hidden_size, attn_heads=attn_heads, dropout=dropout,
                                hidden_activation=hidden_activation)
        self.GatRnn_sigma = GRUGAT(hidden_size=hidden_size, attn_heads=attn_heads, dropout=dropout,
                                   hidden_activation=hidden_activation)

        # Decoders
        self.decoder_mu = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_sigma = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_p = BatchBilinearDecoderDense(activation=None, qr=False)

    def call(self, inputs, states, training=None, mask=None):
        # Encoding
        h_prime_p = self.GatRnn_p(inputs, states[0])
        h_prime_mu = self.GatRnn_mu(inputs, states[1])
        h_prime_sigma = self.GatRnn_sigma(inputs, states[2])

        # Decoding
        x_p, p = self.decoder_p(h_prime_p)
        p = tf.expand_dims(p, -1)
        x_mu, mu = self.decoder_mu(h_prime_mu)
        mu = tf.expand_dims(mu, -1)
        x_sigma, sigma = self.decoder_sigma(h_prime_sigma)
        sigma = tf.expand_dims(sigma, -1)
        logits = tf.concat([p, mu, sigma], -1)
        return logits, h_prime_p, h_prime_mu, h_prime_sigma


class GRUGCNLognormal(m.Model):
    def __init__(self, hidden_size=15, dropout=0.2, hidden_activation="tanh"):
        super(GRUGCNLognormal, self).__init__()
        # Encoders
        self.GatRnn_p = GRUGCN(hidden_size=hidden_size, dropout=dropout,
                               hidden_activation=hidden_activation)
        self.GatRnn_mu = GRUGCN(hidden_size=hidden_size, dropout=dropout,
                                hidden_activation=hidden_activation)
        self.GatRnn_sigma = GRUGCN(hidden_size=hidden_size, dropout=dropout,
                                   hidden_activation=hidden_activation)

        # Decoders
        self.decoder_mu = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_sigma = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_p = BatchBilinearDecoderDense(activation=None, qr=False)

    def call(self, inputs, states, training=None, mask=None):
        # Encoding
        h_prime_p = self.GatRnn_p(inputs, states[0])
        h_prime_mu = self.GatRnn_mu(inputs, states[1])
        h_prime_sigma = self.GatRnn_sigma(inputs, states[2])

        # Decoding
        x_p, p = self.decoder_p(h_prime_p)
        p = tf.expand_dims(p, -1)
        x_mu, mu = self.decoder_mu(h_prime_mu)
        mu = tf.expand_dims(mu, -1)
        x_sigma, sigma = self.decoder_sigma(h_prime_sigma)
        sigma = tf.expand_dims(sigma, -1)
        logits = tf.concat([p, mu, sigma], -1)
        return logits, h_prime_p, h_prime_mu, h_prime_sigma


class GRUGCNDirectedLognormal(m.Model):
    def __init__(self, hidden_size=15, dropout=0.2, hidden_activation="tanh"):
        super(GRUGCNDirectedLognormal, self).__init__()
        # Encoders
        self.GatRnn_p = GRUGCNDirected(hidden_size=hidden_size, dropout=dropout,
                               hidden_activation=hidden_activation)
        self.GatRnn_mu = GRUGCNDirected(hidden_size=hidden_size, dropout=dropout,
                                hidden_activation=hidden_activation)
        self.GatRnn_sigma = GRUGCNDirected(hidden_size=hidden_size, dropout=dropout,
                                   hidden_activation=hidden_activation)

        # Decoders
        self.decoder_mu = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_sigma = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_p = BatchBilinearDecoderDense(activation=None, qr=False)

    def call(self, inputs, states, training=None, mask=None):
        # Encoding
        h_prime_p = self.GatRnn_p(inputs, states[0])
        h_prime_mu = self.GatRnn_mu(inputs, states[1])
        h_prime_sigma = self.GatRnn_sigma(inputs, states[2])

        # Decoding
        x_p, p = self.decoder_p(h_prime_p)
        p = tf.expand_dims(p, -1)
        x_mu, mu = self.decoder_mu(h_prime_mu)
        mu = tf.expand_dims(mu, -1)
        x_sigma, sigma = self.decoder_sigma(h_prime_sigma)
        sigma = tf.expand_dims(sigma, -1)
        logits = tf.concat([p, mu, sigma], -1)
        return logits, h_prime_p, h_prime_mu, h_prime_sigma


class GRULognormal(m.Model):
    def __init__(self, hidden_size=15, dropout=0.2, hidden_activation="tanh"):
        super(GRULognormal, self).__init__()
        # Encoders
        self.rnn_p = GRU(hidden_size=hidden_size, dropout=dropout,
                         hidden_activation=hidden_activation)
        self.rnn_mu = GRU(hidden_size=hidden_size, dropout=dropout,
                          hidden_activation=hidden_activation)
        self.rnn_sigma = GRU(hidden_size=hidden_size, dropout=dropout,
                             hidden_activation=hidden_activation)

        # Decoders
        self.decoder_mu = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_sigma = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_p = BatchBilinearDecoderDense(activation=None, qr=False)

    def call(self, inputs, states, training=None, mask=None):
        # Encoding
        h_prime_p = self.rnn_p(inputs, states[0])
        h_prime_mu = self.rnn_mu(inputs, states[1])
        h_prime_sigma = self.rnn_sigma(inputs, states[2])

        # Decoding
        x_p, p = self.decoder_p(h_prime_p)
        p = tf.expand_dims(p, -1)
        x_mu, mu = self.decoder_mu(h_prime_mu)
        mu = tf.expand_dims(mu, -1)
        x_sigma, sigma = self.decoder_sigma(h_prime_sigma)
        sigma = tf.expand_dims(sigma, -1)
        logits = tf.concat([p, mu, sigma], -1)
        return logits, h_prime_p, h_prime_mu, h_prime_sigma


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--noise", type=str, default="independent")
    parser.add_argument("--pred_years", type=int, default=11)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--baci", action="store_true")
    args = parser.parse_args()
    save = args.save
    encoder = args.encoder
    epochs = args.epochs
    noise = args.noise
    pred_years = args.pred_years
    drop_rate = args.dropout
    test_split = args.test_split
    sparse = args.sparse
    baci = args.baci

    n = 174
    os.chdir("../../")
    if baci:
        with open(os.path.join(os.getcwd(), "Data", "baci_sparse_price.pkl"), "rb") as file:
            t = 25
            data = pkl.load(file)
            data = tf.sparse.reorder(data)
            data_slice = tf.sparse.slice(data, (0, 0, 0, 31), (t, n, n, 1))
            data_dense = tf.sparse.reduce_sum(data_slice, -1)
    else:
        with open(
                os.path.join(os.getcwd(), "Data", "complete_data_final_transformed_no_duplicate.pkl"),
                "rb",
        ) as file:
            data_np = pkl.load(file)
        data_sp = tf.sparse.SparseTensor(
            data_np[:, :4], data_np[:, 4], np.max(data_np, 0)[:4] + 1
        )
        data_sp = tf.sparse.reorder(data_sp)
        t = 55
        data_slice = tf.sparse.slice(data_sp, (0, 31, 0, 0), (t, 1, n, n))
        data_dense = tf.sparse.reduce_sum(data_slice, 1)

    a_train = tf.expand_dims(data_dense, 1)
    times = []
    time_range = np.linspace(0, a_train.shape[0], a_train.shape[0])
    for i in range(4):
        time = np.cos(time_range * np.pi / (1+i*2))
        times.append(time)
    times_embedding = np.asarray([times] * a_train.shape[-1]).swapaxes(0, 2).swapaxes(1, 2)[:, None, :]
    Xt = [np.eye(n)] * t
    x = np.asarray(Xt, dtype=np.float32)
    x = tf.expand_dims(x, 1)
    loss_hist = []
    if encoder == "gat":
        rnn_cell = GRUGATLognormal(dropout=drop_rate)
    elif encoder == "gcn":
        rnn_cell = GRUGCNLognormal(dropout=drop_rate)
    elif encoder == "gcn_dir":
        rnn_cell = GRUGCNDirectedLognormal(dropout=drop_rate)
    elif encoder == "nn":
        rnn_cell = GRULognormal(dropout=drop_rate)
    else:
        raise AttributeError(f"No encoder named {encoder}")

    optimizer = opt.Adam(0.001)
    epochs = tqdm.tqdm(np.arange(epochs))
    train_mask_in = np.random.choice((0, 1), p=(test_split, 1 - test_split), size=(*a_train.shape,))  # TxBxNxN
    train_mask_in = np.asarray(train_mask_in, dtype=np.float32)
    test_mask_in = 1 - train_mask_in
    train_mask_out = tf.transpose([train_mask_in] * 3, perm=(1, 2, 3, 4, 0))
    test_mask_out = 1 - train_mask_out
    for e in epochs:
        tot_loss_train = 0
        tot_loss_test = 0
        states = [None, None, None]
        states_test = [None, None, None]
        prev_noise = [[] for i in range(100)]
        with tf.GradientTape(persistent=True) as tape:
            for t in range(a_train.shape[0] - pred_years):
                if sparse:
                    a_train_t = tf.sparse.from_dense(tf.squeeze(a_train[t]))
                    x_train_t = tf.squeeze(x[t])
                else:
                    x_train_t = x[t]
                    a_train_t = a_train[t] * train_mask_in[t]
                logits, h_prime_p, h_prime_mu, h_prime_sigma = rnn_cell([x_train_t, a_train_t * train_mask_in[t]],
                                                                        states=states, training=True)
                logits_test, h_prime_p_test, h_prime_mu_test, h_prime_sigma_test = rnn_cell(
                    [x[t], a_train[t] * test_mask_in[t]], states=states_test, training=True)
                #logits_test = logits
                states = [h_prime_p, h_prime_mu, h_prime_sigma]
                states_test = [h_prime_p_test, h_prime_mu_test, h_prime_sigma_test]
                l_train = zero_inflated_lognormal_loss(labels=tf.expand_dims(a_train[t + 1] * train_mask_in[t + 1], -1),
                                                       logits=logits * train_mask_out[t + 1])
                l_test = zero_inflated_lognormal_loss(labels=tf.expand_dims(a_train[t + 1] * test_mask_in[t + 1], -1),
                                                      logits=logits_test * test_mask_out[t + 1])
                tot_loss_train = 0.5 * tot_loss_train + 0.5 * l_train
                tot_loss_test = 0.5 * tot_loss_test + 0.5 * l_test
                gradients = tape.gradient(tot_loss_train, rnn_cell.trainable_weights)
                if noise == "uncorrelated":
                    for g_i, g in enumerate(gradients):
                        noise_curr = tf.random.normal(tf.shape(g), 0, .01)
                        if t == 0:
                            gradients[g_i] = g + noise_curr
                        else:
                            gradients[g_i] = g + noise_curr - prev_noise[g_i]
                        prev_noise[g_i] = noise_curr
                elif noise == "independent":
                    for g_i, g in enumerate(gradients):
                        gradients[g_i] = g + tf.random.normal(tf.shape(g), 0, 0.01)
                optimizer.apply_gradients(zip(gradients, rnn_cell.trainable_weights))
                if not t % 6:
                    tape.reset()
        loss_hist.append((tot_loss_train, tot_loss_test))
        # print(f"Epoch {e} Loss:{tot_loss.numpy()}")

    a_t = zero_inflated_lognormal(logits).sample(1)
    if sparse:
        x_mu = h_prime_mu
        x_var = h_prime_sigma
        x_p = h_prime_p
        a_t = tf.squeeze(a_t)
    else:
        x_mu = tf.squeeze(h_prime_mu)
        x_var = tf.squeeze(h_prime_sigma)
        x_p = tf.squeeze(h_prime_p)
        a_t = tf.squeeze(a_t, (0, 1))
    a_t = tf.math.log(a_t)
    a_t = tf.clip_by_value(a_t, 0.0, 1e12)
    a_t = a_t.numpy()
    a = tf.clip_by_value(tf.math.log(a_train[-1, 0, :, :]).numpy(), 0, 1e10).numpy()

    today = datetime.datetime.now().isoformat().replace(":", "_")
    weights_dir = os.path.join(os.getcwd(), "src", "Tests", "Rnn Bilinear Lognormal", today, "Weights")
    figures_dir = os.path.join(os.getcwd(), "src", "Tests", "Rnn Bilinear Lognormal", today, "Figures")
    if save:
        os.makedirs(weights_dir)
        os.makedirs(figures_dir)
        rnn_cell.save_weights(os.path.join(weights_dir, "gat_rnn"))

    ############################################# PLOTTING #############################################################
    loss_hist = np.asarray(loss_hist)
    mplt.rcParams['figure.figsize'] = (15, 10)
    losses = plt.plot(loss_hist)
    plt.legend(losses, ["Train", "Test"])
    plt.title("Loss History")
    if save:
        plt.savefig(os.path.join(figures_dir, "loss.png"))
        plt.close()

    plt.figure()
    plt.imshow(a)
    plt.colorbar()
    plt.title("True Adj")
    if save:
        plt.savefig(os.path.join(figures_dir, "adj_true.png"))
        plt.close()

    plt.figure()
    plt.imshow(a_t)
    plt.colorbar()
    plt.title("Pred Weighted Adj")
    if save:
        plt.savefig(os.path.join(figures_dir, "adj_pred.png"))
        plt.close()

    plt.figure()
    diff = a - a_t
    plt.imshow(diff)
    plt.colorbar()
    plt.title("Difference Weighted - True Adj")
    if save:
        plt.savefig(os.path.join(figures_dir, "error.png"))
        plt.close()

    scale = tf.math.maximum(
        tf.nn.softplus(logits[..., 2:]),
        tf.math.sqrt(tf.keras.backend.epsilon()))

    plt.figure()
    if sparse:
        scale = scale.numpy().squeeze((-1))
    else:
        scale = scale.numpy().squeeze((0, -1))
    plt.imshow(scale)
    plt.colorbar()
    plt.title("Variance")
    if save:
        plt.savefig(os.path.join(figures_dir, "adj_variance.png"))
        plt.close()

    x_tnse = TSNE(n_components=2, perplexity=80).fit_transform(x_mu.numpy())
    clustering = SpectralClustering(n_clusters=10, affinity="nearest_neighbors").fit(
        x_tnse
    )
    labels = clustering.labels_
    colormap = plt.cm.get_cmap("Set1")
    colors = colormap(labels)
    fig, ax = plt.subplots()
    ax.scatter(x_tnse[:, 0], x_tnse[:, 1], color=colors)
    ax.set_title("Mean Embedding")
    plot_names(x_tnse, ax, baci=baci)
    if save:
        plt.savefig(os.path.join(figures_dir, "embeddings_mean.png"))
        plt.close()

    x_tnse_var = TSNE(n_components=2, perplexity=80).fit_transform(x_var.numpy())
    clustering = SpectralClustering(n_clusters=10, affinity="nearest_neighbors").fit(
        x_tnse_var
    )
    labels = clustering.labels_
    colormap = plt.cm.get_cmap("Set1")
    colors = colormap(labels)
    fig1, ax1 = plt.subplots()
    ax1.scatter(x_tnse_var[:, 0], x_tnse_var[:, 1], color=colors)
    ax1.set_title("Variance Embedding")
    plot_names(x_tnse_var, ax1, baci=baci)
    if save:
        plt.savefig(
            os.path.join(figures_dir, "embeddings_var.png"))
        plt.close()

    a_pred = a_t.flatten()
    a_pred = a_pred[a_pred > 0.2]
    a_true = a.flatten()
    a_true = a_true[a_true > 0.2]

    plt.figure()
    plt.hist(a_pred, 100, color="red", alpha=0.5, density=True, label="Pred")
    plt.hist(a_true, 100, color="blue", alpha=0.5, density=True, label="True")
    plt.legend()
    if save:
        plt.savefig(
            os.path.join(figures_dir, "edge_distr.png"))
        plt.close()

    plt.figure()
    diff = diff[(diff > 0.01) | (diff < -0.01)]
    plt.hist(diff.flatten(), bins=100)
    if save:
        plt.savefig(
            os.path.join(figures_dir, "true_pred_edge_distr.png"))
        plt.close()

    plt.figure()
    stats.probplot(diff.flatten(), dist="norm", plot=plt)
    plt.title("QQ-plot True-Pred")
    if save:
        plt.savefig(os.path.join(figures_dir, "error_distr.png"))
        plt.close()

    '''R = model.layers[0].trainable_weights[0]
    X = model.layers[0].trainable_weights[1]
    plt.figure()
    plt.imshow(R.numpy())
    plt.colorbar()
    plt.title("R Coefficient")
    if save:
        plt.savefig(os.path.join(figures_dir, "Rnn Bilinear Lognormal", today,"coeff_R.png"))
        plt.close()

    plt.figure()
    img = plt.imshow(X.numpy().T)
    plt.colorbar(img, fraction=0.0046, pad=0.04)
    plt.title("X embeddings")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(figures_dir, "Rnn Bilinear Lognormal", today,"x_embs.png"))
        plt.close()
    '''

    plt.figure()
    bin_true = tf.reshape(tf.where(a == 0, 0.0, 1.0), [-1]).numpy()
    p_pred = tf.reshape(tf.nn.sigmoid(logits[..., :1]), [-1]).numpy()
    fpr, tpr, thr = roc_curve(bin_true, p_pred, drop_intermediate=False)
    cmap = plt.cm.get_cmap("viridis")
    plt.scatter(fpr, tpr, color=cmap(thr), s=2)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot((0, 1), (0, 1), color="black")
    plt.colorbar()
    if save:
        plt.savefig(os.path.join(figures_dir, "roc.png"))
        plt.close()

    plt.figure()
    prec, rec, thr = precision_recall_curve(bin_true, p_pred)
    plt.scatter(prec[:-1], rec[:-1], color=cmap(thr), s=2)
    plt.title("Precision Recall Curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.plot((0, 1), (1, 0), color="black")
    plt.colorbar()
    if save:
        plt.savefig(os.path.join(figures_dir, "prec_rec.png"))
        plt.close()

    plt.figure()
    true_pos = p_pred[bin_true == 1]
    false_negative = p_pred[bin_true == 0]
    plt.hist(true_pos, bins=100, density=True, alpha=0.4, color="green", label="positive preds")
    plt.hist(false_negative, bins=100, density=True, alpha=0.4, color="red", label="negative preds")
    plt.plot((0.5, 0.5), (0, 70), lw=1)
    plt.legend()
    if save:
        plt.savefig(os.path.join(figures_dir, "distr_preds.png"))
        plt.close()

    # Prediction
    states = [None, None, None]
    t_samples = []
    for t in range(a_train.shape[0]):
        if sparse:
            a_train_t = tf.sparse.from_dense(tf.squeeze(a_train[t]))
            x_train_t = tf.squeeze(x[t])
        else:
            x_train_t = x[t]
            a_train_t = a_train[t] * train_mask_in[t]
        logits, h_prime_p, h_prime_mu, h_prime_sigma = rnn_cell([x_train_t, a_train_t], states=states,
                                                                time=times_embedding[t], training=False)
        states = [h_prime_p, h_prime_mu, h_prime_sigma]
        sample = tf.squeeze(zero_inflated_lognormal(logits).sample(1)).numpy()
        t_samples.append(sample)

    a = tf.squeeze(tf.clip_by_value(tf.math.log(a_train), 0, 1e100)).numpy()
    t_samples = tf.clip_by_value(tf.math.log(t_samples), 0, 1e100).numpy()
    fig, ax = plt.subplots(1, 2)


    def update(i):
        ax[0].clear()
        ax[1].clear()
        fig.suptitle(f"Year {1964 + i}")
        ax[0].set_title("Pred")
        t = ax[0].imshow(t_samples[i], animated=True)
        ax[1].set_title("True")
        p = ax[1].imshow(a[i], animated=True)


    anim = animation.FuncAnimation(fig, update, frames=a.shape[0], repeat=True)
    if save:
        anim.save(os.path.join(figures_dir, "temporal_adj.gif"), writer="pillow")
        plt.close(fig)

    fig1, ax1 = plt.subplots(1, 1)
    range_axis = (1.0, 30)


    def update1(i):
        ax1.clear()
        fig1.suptitle(f"Year {1964 + i}")
        a_pred = t_samples[i].flatten()
        a_true = a[i].flatten()
        ax1.hist(a_true, bins=60, alpha=0.5, range=range_axis, color="blue", density=True, label="True")
        ax1.set_ylim((0, .21))
        ax1.hist(a_pred, bins=60, alpha=0.5, range=range_axis, color="red", density=True, label="Pred")
        ax1.legend()


    anim2 = animation.FuncAnimation(fig1, update1, frames=a.shape[0], repeat=True)
    if save:
        anim2.save(os.path.join(figures_dir, "temporal_distr.gif"), writer="pillow")
        plt.close(fig1)

    '''fig3, axs3 = plt.subplots(2, 1)
    def update3(i):
        in_degree_true = np.sum(a[t], 0)
        in_degree_pred = np.sum(t_samples[t], 0)
        out_degree_true = np.sum(a[t], 1)
        out_degree_pred = np.sum(t_samples[t], 1)
        axs3[0].hist(in_degree_true, bins=10, density=True, alpha=0.4, color="blue")
        axs3[0].hist(in_degree_pred, bins=10, density=True, alpha=0.4, color="red")
        axs3[0].set_label(f"In Degree Distribution at time {1964 + i}")
        axs3[1].hist(out_degree_true, bins=10, density=True, alpha=0.4, color="blue")
        axs3[1].hist(out_degree_pred, bins=10, density=True, alpha=0.4, color="red")
        axs3[1].set_label(f"Out Degree Distribution at time {1964 + i}")

    anim3 = animation.FuncAnimation(fig3, update3, frames=a.shape[0], repeat=True)
    if save:
        anim3.save(os.path.join(figures_dir, "temporal_inDegree_distr.gif"), writer="pillow")
        plt.close(fig3)'''

    if save:
        with open(os.path.join(figures_dir, "settings.txt"), "w") as file:
            settings = f"encoder_type: {encoder} \n" \
                       f"training_years: from {1964} to {1964 + a.shape[0] - pred_years} \n" \
                       f"pred_years: from {1964 + a.shape[0] - pred_years} to {1964 + a.shape[0]} \n" \
                       f"dropout_rate: {drop_rate} \n" \
                       f"sparse: {sparse} \n" \
                       f"baci:{baci}"
            file.write(settings)
    plt.show()
