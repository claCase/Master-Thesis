import numpy as np
import tensorflow as tf
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


class RNNGATLognormal(m.Model):
    def __init__(self, hidden_size=5, attn_heads=5, dropout=0.2, hidden_activation="relu"):
        super(RNNGATLognormal, self).__init__()
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


class RNNGCNLognormal(m.Model):
    def __init__(self, hidden_size=15, dropout=0.2, hidden_activation="tanh"):
        super(RNNGCNLognormal, self).__init__()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--encoder", type=str, default="gat")
    args = parser.parse_args()
    save = args.save
    encoder = args.encoder

    n = 174
    t = 30
    os.chdir("../../")
    with open(
            os.path.join(os.getcwd(), "Data", "complete_data_final_transformed_no_duplicate.pkl"),
            "rb",
    ) as file:
        data_np = pkl.load(file)
    data_sp = tf.sparse.SparseTensor(
        data_np[:, :4], data_np[:, 4], np.max(data_np, 0)[:4] + 1
    )
    data_sp = tf.sparse.reorder(data_sp)

    data_slice = tf.sparse.slice(data_sp, (20, 10, 0, 0), (t, 1, n, n))

    data_dense = tf.sparse.reduce_sum(data_slice, 1)
    a = tf.expand_dims(data_dense, 1)
    Xt = [np.eye(n)] * t
    x = np.asarray(Xt, dtype=np.float32)
    x = tf.expand_dims(x, 1)
    loss_hist = []
    if encoder == "gat":
        rnn_cell = RNNGATLognormal()
    else:
        rnn_cell = RNNGCNLognormal()
    optimizer = opt.Adam(0.001)
    for e in range(120):
        h_t = None
        initialize = True
        tot_loss = 0
        states = [None, None, None]
        with tf.GradientTape(persistent=True) as tape:
            for t in range(a.shape[0] - 1):
                logits, h_prime_p, h_prime_mu, h_prime_sigma = rnn_cell([x[t], a[t]],
                                                                        states=states
                                                                        )
                states = [h_prime_p, h_prime_mu, h_prime_sigma]
                l = zero_inflated_lognormal_loss(labels=tf.expand_dims(a[t + 1], -1), logits=logits)
                tot_loss = tot_loss + l
                gradients = tape.gradient(tot_loss, rnn_cell.trainable_weights)
                optimizer.apply_gradients(zip(gradients, rnn_cell.trainable_weights))
        loss_hist.append(tot_loss)
        print(f"Epoch {e} Loss:{tot_loss.numpy()}")

    x_mu = tf.squeeze(h_prime_mu)
    x_var = tf.squeeze(h_prime_sigma)
    x_p = tf.squeeze(h_prime_p)
    a_t = zero_inflated_lognormal(logits).sample(1)
    a_t = tf.squeeze(a_t, (0, 1))
    a_t = tf.math.log(a_t)
    a_t = tf.clip_by_value(a_t, 0.0, 1e12)
    a_t = a_t.numpy()
    a = tf.clip_by_value(tf.math.log(a[-1].numpy()[0]), 0.0, 1e12)
    a = a.numpy()

    ############################################# PLOTTING #############################################################
    loss_hist = np.asarray(loss_hist)
    mplt.rcParams['figure.figsize'] = (15, 10)
    plt.plot(loss_hist)
    plt.title("Loss History")
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "loss.png"))
        plt.close()

    plt.figure()
    plt.imshow(a)
    plt.colorbar()
    plt.title("True Adj")
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "adj_true.png"))
        plt.close()

    plt.figure()
    plt.imshow(a_t)
    plt.colorbar()
    plt.title("Pred Weighted Adj")
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "adj_pred.png"))
        plt.close()

    plt.figure()
    diff = a - a_t
    plt.imshow(diff)
    plt.colorbar()
    plt.title("Difference Weighted - True Adj")
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "error.png"))
        plt.close()

    scale = tf.math.maximum(
        tf.nn.softplus(logits[..., 2:]),
        tf.math.sqrt(tf.keras.backend.epsilon()))

    plt.figure()
    plt.imshow(scale.numpy().squeeze((0, -1)))
    plt.colorbar()
    plt.title("Variance")
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "adj_variance.png"))
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
    plot_names(x_tnse, ax)
    if save:
        plt.savefig(
            os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "embeddings_mean.png"))
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
    plot_names(x_tnse_var, ax1)
    if save:
        plt.savefig(
            os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "embeddings_var.png"))
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
            os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "edge_distr.png"))
        plt.close()

    plt.figure()
    diff = diff[(diff > 0.01) | (diff < -0.01)]
    plt.hist(diff.flatten(), bins=100)
    if save:
        plt.savefig(
            os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "true_pred_edge_distr.png"))
        plt.close()

    plt.figure()
    stats.probplot(diff.flatten(), dist="norm", plot=plt)
    plt.title("QQ-plot True-Pred")
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "error_distr.png"))
        plt.close()

    '''R = model.layers[0].trainable_weights[0]
    X = model.layers[0].trainable_weights[1]
    plt.figure()
    plt.imshow(R.numpy())
    plt.colorbar()
    plt.title("R Coefficient")
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "coeff_R.png"))
        plt.close()

    plt.figure()
    img = plt.imshow(X.numpy().T)
    plt.colorbar(img, fraction=0.0046, pad=0.04)
    plt.title("X embeddings")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "x_embs.png"))
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
    plt.colorbar()
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "roc.png"))
        plt.close()

    plt.figure()
    prec, rec, thr = precision_recall_curve(bin_true, p_pred)
    plt.scatter(prec[:-1], rec[:-1], color=cmap(thr), s=2)
    plt.title("Precision Recall Curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.colorbar()
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "prec_rec.png"))
        plt.close()

    plt.figure()
    true_pos = p_pred[bin_true == 1]
    false_negative = p_pred[bin_true == 0]
    plt.hist(true_pos, bins=100, density=True, alpha=0.4, color="green", label="positive preds")
    plt.hist(false_negative, bins=100, density=True, alpha=0.4, color="red", label="negative preds")
    plt.plot((0.5, 0.5), (0, 70), lw=1)
    plt.legend()
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Rnn Bilinear Lognormal", "distr_preds.png"))
        plt.close()

    plt.show()
