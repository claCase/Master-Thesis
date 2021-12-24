import tensorflow as tf
from tensorflow import scan
from tensorflow.keras.layers import LSTM
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
import tensorflow.keras as k
from tensorflow.keras.layers import LayerNormalization
from spektral.layers.convolutional import GATConv
from src.modules.layers import BilinearDecoderDense, BilinearDecoderSparse
from src.modules.losses import square_loss, mse_uncertainty, embedding_smoothness, temporal_embedding_smoothness, \
    DensePenalizedMSE
from src.modules.utils import generate_list_lower_triang, sum_gradients
from src.modules.models import GAT_BIL_spektral_dense
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
from src.graph_data_loader import plot_names
from tensorflow.keras.models import Model
import os
import pickle as pkl
import tqdm
import argparse


def _get_positional_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


class TGNN(k.models.Model):
    def __init__(self, dropout_rate=0.5, lag=10):
        super(TGNN, self).__init__()
        # self.encoder = GATConv(channels=10, attn_heads=10, concat_heads=True)
        self.lag = lag
        self.dropout_rate = dropout_rate
        self.decoder = BilinearDecoderDense(qr=True)
        self.drop = k.layers.Dropout(self.dropout_rate)
        self.temporal_module = GATConv(
            channels=2, attn_heads=1, concat_heads=True, dropout_rate=0.5
        )
        self.lower_triang = generate_list_lower_triang(n, t, self.lag)

    def call(self, inputs, training=None, mask=None):
        """
        inputs: Tuple (X, A):
                - X is a Tensor of shape TxNxf where T is time, N is the number of nodes, f is the feature dimension
                - X is a Tensor of shape TxNxN where T is time and N is the number of nodes
        """
        Xt, At = inputs
        # prepare Xt for temporal self attention: TxNxf -> NxTxf
        Xt_perm = tf.transpose(Xt, [1, 0, 2])
        # temporal encoding: NxTxf -> NxTxf
        temp_enc = self.temporal_module([Xt_perm, self.lower_triang])
        print(temp_enc)
        # prepare for decoding: NxTxf -> TxNxf
        temp_enc = tf.transpose(temp_enc, [1, 0, 2])
        if self.dropout_rate:
            temp_enc = self.drop(temp_enc)
        # predict adjacency matrix
        temp_enc, t_adjs = self.decoder([temp_enc, At])
        return temp_enc, t_adjs


class RNNBIL(k.models.Model):
    def __init__(self, channels=5, attn_heads=3, rnn_units=25, skip_connection=False):
        super(RNNBIL, self).__init__()
        self.skip_connection = skip_connection
        self.attn_heads = attn_heads
        self.rnn_units = rnn_units
        self.channels = channels
        if self.skip_connection:
            self.connect = k.layers.Dense(self.channels * self.attn_heads, "relu")
        self.rnn_lower = k.layers.SimpleRNN(
            units=self.rnn_units,
            return_state=True,
            return_sequences=True,
            stateful=False,
            time_major=True,
            activation="relu",
            dropout=0.5,
            recurrent_dropout=0.5,
            use_bias=True,
            bias_regularizer="l2",
            bias_initializer="glorot_normal"
        )
        self.rnn_upper = k.layers.SimpleRNN(
            units=self.rnn_units,
            return_state=True,
            return_sequences=True,
            stateful=False,
            time_major=True,
            activation="relu",
            dropout=0.5,
            recurrent_dropout=0.5,
            use_bias=True,
            bias_regularizer="l2",
            bias_initializer="glorot_normal"
        )
        self.encoder_lower = GATConv(
            channels=self.channels,
            attn_heads=self.attn_heads,
            add_self_loops=False,
            activation="relu",
            concat_heads=True,
            # kernel_regularizer="l2",
            # bias_regularizer="l2",
            use_bias=True
        )
        self.encoder_upper = GATConv(
            channels=self.channels,
            attn_heads=self.attn_heads,
            add_self_loops=False,
            activation="relu",
            concat_heads=True,
            # kernel_regularizer="l2",
            # bias_regularizer="l2",
            use_bias=True
        )
        self.decoder_w = BilinearDecoderDense(activation="relu", qr=True)
        # self.decoder_s = BilinearDecoderDense(activation="relu", qr=True)
        self.ln = LayerNormalization()
        self.initial = True
        self.state_lower = None
        self.state_upper = None

    def call(self, inputs, training=True, mask=None):
        x, a = inputs
        a_lower = tf.linalg.LinearOperatorLowerTriangular(a).to_dense()
        a_lower = a_lower + tf.transpose(a_lower)
        a_upper = tf.linalg.LinearOperatorLowerTriangular(tf.transpose(a)).to_dense()
        a_upper = a_upper + tf.transpose(a_upper)
        x_enc_lower = self.encoder_lower([x, a_lower])
        x_enc_upper = self.encoder_upper([x, a_upper])
        x_enc_lower = self.ln(x_enc_lower)
        x_enc_upper = self.ln(x_enc_upper)
        if self.initial:
            init_state = tf.constant(np.zeros(shape=(x.shape[0], self.rnn_units), dtype=np.float32))
            x_prime_lower, state_lower = self.rnn_lower(
                tf.expand_dims(x_enc_lower, 0), initial_state=init_state, training=training
            )
            x_prime_upper, state_upper = self.rnn_upper(
                tf.expand_dims(x_enc_upper, 0), initial_state=init_state, training=training
            )
            self.state_lower = state_lower
            self.state_upper = state_upper
        else:
            x_prime_lower, state_lower = self.rnn_lower(
                tf.expand_dims(x_enc_lower, 0), training=training
            )
            x_prime_upper, state_upper = self.rnn_upper(
                tf.expand_dims(x_enc_upper, 0), training=training
            )
            self.state_lower = state_lower
            self.state_upper = state_upper

        x_prime = tf.concat([x_prime_lower, x_prime_upper], -1)
        x_prime = tf.squeeze(x_prime)
        if self.skip_connection:
            x_prime = tf.concat([x_prime, x_enc_upper, x_enc_upper], -1)
            x_prime = self.connect(x_prime)
        x_prime = self.ln(x_prime)
        x_enc, pred_adj = self.decoder_w([x_prime, a])
        # _, pred_sigma = self.decoder_s([x_prime, a])
        return x_enc, pred_adj  # , pred_sigma


class RNNBIL2(k.models.Model):
    def __init__(self, channels=5, attn_heads=3, rnn_units=25, skip_connection=False):
        super(RNNBIL2, self).__init__()
        self.skip_connection = skip_connection
        self.attn_heads = attn_heads
        self.rnn_units = rnn_units
        self.channels = channels
        if self.skip_connection:
            self.connect = k.layers.Dense(self.channels * self.attn_heads, "relu")
        self.rnn_lower = k.layers.SimpleRNN(
            units=self.rnn_units,
            return_state=True,
            return_sequences=True,
            stateful=False,
            time_major=True,
            activation="relu",
            dropout=0.5,
            recurrent_dropout=0.5,
            use_bias=True,
            bias_regularizer="l2",
            bias_initializer="glorot_normal"
        )
        '''self.rnn_upper = k.layers.SimpleRNN(
            units=self.rnn_units,
            return_state=True,
            return_sequences=True,
            stateful=False,
            time_major=True,
            activation="relu",
            dropout=0.5,
            recurrent_dropout=0.5,
            use_bias=True,
            bias_regularizer="l2",
            bias_initializer="glorot_normal"
        )'''
        self.encoder_lower = GATConv(
            channels=self.channels,
            attn_heads=self.attn_heads,
            add_self_loops=False,
            activation="relu",
            concat_heads=True,
            # kernel_regularizer="l2",
            # bias_regularizer="l2",
            use_bias=True
        )
        '''self.encoder_upper = GATConv(
            channels=self.channels,
            attn_heads=self.attn_heads,
            add_self_loops=False,
            activation="relu",
            concat_heads=True,
            # kernel_regularizer="l2",
            # bias_regularizer="l2",
            use_bias=True
        )'''
        self.decoder_w = BilinearDecoderDense(activation="relu", qr=True)
        # self.decoder_s = BilinearDecoderDense(activation="relu", qr=True)
        self.ln = LayerNormalization()
        self.initial = True

    def call(self, inputs, training=None, mask=None):
        x,a = inputs
        x_lower = self.encoder_lower([x, a])
        if self.initial:
            initial_state = tf.constant(np.zeros(shape=(x.shape[0], self.rnn_units), dtype=np.float32))
            x_lower = self.rnn_lower(x_lower, initial_state=initial_state)
        else:
            x_lower = self.rnn_lower(x_lower)

        a = self.decoder_w([x_lower, a])
        return x_lower, a


class SparseRNNBIL(k.models.Model):
    def __init__(self, channels=2, attn_heads=10, rnn_units=20, skip_connection=True):
        super(SparseRNNBIL, self).__init__()
        self.skip_connection = skip_connection
        self.attn_heads = attn_heads
        self.rnn_units = rnn_units
        self.channels = channels
        if self.skip_connection:
            self.connect = k.layers.Dense(self.channels * self.attn_heads, "relu")

        self.rnn = k.layers.SimpleRNN(
            units=self.rnn_units,
            return_state=True,
            return_sequences=True,
            stateful=False,
            time_major=True,
            activation="relu",
            dropout=0.5,
            recurrent_dropout=0.5
        )
        self.encoder = GATConv(
            channels=self.channels,
            attn_heads=self.attn_heads,
            add_self_loops=False,
            activation="relu",
            concat_heads=True,
            kernel_regularizer="l2",
            bias_regularizer="l2",
            use_bias=True
        )

        self.decoder_w = BilinearDecoderSparse(activation="relu")
        self.ln = LayerNormalization()
        self.initial = True
        self.state = None

    def call(self, inputs, training=None, mask=None):
        x, a, a1 = inputs
        x_enc = self.encoder([x, a])
        x_enc = self.ln(x_enc)
        if self.initial:
            init_state = tf.constant(np.zeros(shape=(x.shape[0], self.rnn_units), dtype=np.float32))
            x_prime, state = self.rnn(
                tf.expand_dims(x_enc, 0), initial_state=init_state
            )
            self.state = state
        else:
            x_prime, state = self.rnn(
                tf.expand_dims(x_enc, 0), initial_state=self.state
            )
            self.state = state

        x_prime = tf.squeeze(x_prime)
        if self.skip_connection:
            x_prime = tf.concat([x_prime, x_enc], -1)
            x_prime = self.connect(x_prime)
        x_prime = self.ln(x_prime)
        x_enc, pred_adj = self.decoder_w([x_prime, a1])
        return x_enc, pred_adj


if __name__ == "__main__":
    tf.get_logger().setLevel('INFO')
    CUDA_VISIBLE_DEVICES = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--t", type=int, default=6)
    parser.add_argument("--n", type=int, default=174)
    parser.add_argument("--batches", default=2)
    parser.add_argument("--epochs", default=120)
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    synthetic = args.synthetic
    t = args.t
    n = args.n
    sparse = args.sparse
    batches = args.batches
    epochs = args.epochs
    if synthetic:
        input_shape_nodes = (t, n, n)
        input_shape_adj = (t, n, n)
        At = np.random.choice((0, 1), size=input_shape_adj)
        At = np.asarray(At, dtype=np.float32)
    else:
        with open(
                "A:\\Users\\Claudio\\Documents\\PROJECTS\\Master-Thesis\\Data\\complete_data_final_transformed_no_duplicate.pkl",
                "rb",
        ) as file:
            data_np = pkl.load(file)
        data_sp = tf.sparse.SparseTensor(
            data_np[:, :4], data_np[:, 4], np.max(data_np, 0)[:4] + 1
        )
        data_sp = tf.sparse.reorder(data_sp)

        data_slice = tf.sparse.slice(data_sp, (0, 10, 0, 0), (50, 1, n, n))

        data_dense = tf.sparse.reduce_sum(data_slice, 1)
        data_dense = tf.math.log(data_dense)
        At = tf.clip_by_value(data_dense, 0.0, 1e12)

        # At = At[1:] - At[:-1]
        # At = np.diff(At.numpy(), 0)[:-1]

    # pos = _get_positional_encoding_matrix(t, n)
    Xt = [np.eye(n)] * t
    Xt = np.asarray(Xt, dtype=np.float32)
    """for i in range(n):
        Xt[:,i,:] += pos
    """
    X = np.eye(n)
    if sparse:
        model = SparseRNNBIL()
    else:
        model = RNNBIL(skip_connection=True)
    # l = k.losses.MeanSquaredError()
    # l = DensePenalizedMSE()
    # lb = k.losses.binary_crossentropy
    l = square_loss
    # l = mse_uncertainty

    loss_hist = []
    all_preds = []
    all_embs = []

    def l_sheduler(l0, i, decay):
        return l0 * np.exp(-decay * i)

    batches = np.random.choice(np.arange(At.shape[0] - t), size=batches)
    bar = tqdm.tqdm(total=epochs * (t-1) * len(batches), position=0, leave=True, desc="Training")
    epoch_batch_loss = []
    for i in range(epochs):
        grads_b = []
        batch_loss = []
        optimizer = k.optimizers.Adam(0.001)
        preds_b = []
        embs_b = []
        for b in batches:
            A_t_b = At[b:b + t]
            preds_t = []
            embs_t = []
            tot_loss = []
            grads_t = []
            with tf.GradientTape(persistent=True) as tape:
                for i in range(t - 1):
                    if sparse:
                        A = tf.sparse.from_dense(A_t_b[i])
                        A1 = tf.sparse.from_dense(A_t_b[i + 1])
                    else:
                        A = A_t_b[i]
                        A1 = A_t_b[i + 1]
                    if sparse:
                        x, a = model([X, A, A1])
                        preds_t.append(tf.sparse.to_dense(a))
                    else:
                        x, a = model([X, A])
                        preds_t.append(a)
                    embs_t.append(x)
                    loss = l(A1, a)
                    # loss_bin = k.losses.binary_crossentropy(tf.reshape(At[i+1], [-1]), tf.reshape(a_bin, [-1]))
                    # loss += loss_bin
                    tot_loss.append(loss)

                    #grads_t.append(grads)
                    bar.update(1)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                model.initial = True
            #grads_sum_t = sum_gradients(grads_t, "sum")
            #grads_b.append(grads_sum_t)
            batch_loss.append(tf.reduce_mean(tot_loss))
            embs_b.append(embs_t)
            preds_b.append(preds_t)
        all_embs.append(embs_b)
        all_preds.append(preds_b)
        #grads_mean_b = sum_gradients(grads_b, "mean")
        #optimizer.apply_gradients(zip(grads_mean_b, model.trainable_variables))
        bl = tf.reduce_mean(batch_loss, 0)
        epoch_batch_loss.append(bl)

    plt.plot(epoch_batch_loss)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    #plt.show()

    a = np.asarray(all_preds[-1][-1][-1])

    xT = np.swapaxes(np.asarray(all_embs[-1][-1]), 0, 1)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i in range(n):
        ax.plot3D(xT[i, :, 0], xT[i, :, 1], xT[i, :, 2])

    fig, ax = plt.subplots(1, 3, figsize=(6, 15))
    t = ax[0].imshow(At[-1])
    ax[0].set_title("True")
    fig.colorbar(t, ax=ax[0], fraction=0.05)
    p = ax[1].imshow(a)
    ax[1].set_title("Pred")
    fig.colorbar(p, ax=ax[1], fraction=0.05)
    d = ax[2].imshow(At[-1] - a)
    ax[2].set_title("Diff")
    fig.colorbar(d, ax=ax[2], fraction=0.05)

    '''fig, ax = plt.subplots(1, 3, figsize=(6, 15))
    t = ax[0].imshow(At[-1])
    ax[0].set_title("True")
    fig.colorbar(t, ax=ax[0], fraction=0.05)
    p = ax[1].imshow(a[-1][-1])
    ax[1].set_title("Pred")
    fig.colorbar(p, ax=ax[1], fraction=0.05)
    d = ax[2].imshow(At[-1] - a[-1][-1])
    ax[2].set_title("Diff")
    fig.colorbar(d, ax=ax[2], fraction=0.05)
    '''
    x_tnse = TSNE(n_components=2, perplexity=80).fit_transform(x.numpy())
    clustering = SpectralClustering(n_clusters=10, affinity="nearest_neighbors").fit(
        x_tnse
    )
    labels = clustering.labels_
    colormap = cm.get_cmap("Set1")
    colors = colormap(labels)
    fig, ax = plt.subplots()
    ax.scatter(x_tnse[:, 0], x_tnse[:, 1], color=colors)
    plot_names(x_tnse, ax)

    a_pred = a.flatten()
    a_pred = a_pred[a_pred > 0.2]
    a_true = At[-1].numpy().flatten()
    a_true = a_true[a_true > 0.2]
    plt.figure()
    plt.hist(a_pred, 100, color="red", alpha=0.5, density=True, label="Pred")
    plt.hist(a_true, 100, color="blue", alpha=0.5, density=True, label="True")
    plt.legend()
    plt.figure()
    diff = a - At[-1].numpy()
    diff = diff[(diff > 0.01) | (diff < -0.01)]
    plt.hist(diff.flatten(), bins=100)

    plt.figure()
    stats.probplot(diff.flatten(), dist="norm", plot=plt)
    plt.title("QQ-plot True-Pred")
    plt.show()
