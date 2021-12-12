import tensorflow as tf
from tensorflow import scan
from tensorflow.keras.layers import LSTM
import tensorflow.keras as k
from tensorflow.keras.layers import LayerNormalization
from spektral.layers.convolutional import GATConv
from src.modules.layers import BilinearDecoderDense
from src.modules.losses import square_loss, embedding_smoothness, temporal_embedding_smoothness
from src.modules.utils import generate_list_lower_triang
from src.modules.models import GAT_BIL_spektral_dense
import numpy as np
import matplotlib.pyplot as plt
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


class MCLSTM(k.layers.LSTM):
    def __init__(self, units, **kwargs):
        super(MCLSTM, self).__init__(units, **kwargs)

    def call(self, inputs, training=True, mask=None, initial_state=None):
        return super(MCLSTM, self).call(
            inputs,
            mask=mask,
            training=True,
            initial_state=initial_state,
        )


class LSTMBIL(k.models.Model):
    def __init__(self, dropout_rate=0.8):
        super(LSTMBIL, self).__init__()
        self.dropout_rate = dropout_rate
        self.encoder = GATConv(
            channels=15,
            attn_heads=5,
            concat_heads=True,
            dropout_rate=self.dropout_rate,
            activation="tanh",
            add_self_loops=False,
        )
        self.temporal_enc = k.layers.LSTM(
            15,
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            time_major=True,
            activation="tanh",
            # kernel_regularizer="l2",
            # recurrent_regularizer="l2"
        )
        """self.temporal_enc2 = k.layers.LSTM(5, return_sequences=True, dropout=self.dropout_rate,
                                          recurrent_dropout=self.dropout_rate,
                                          time_major=True,
                                          activation="tanh")
        """
        self.decoder = BilinearDecoderDense(qr=True, activation="relu")

    def call(self, inputs, training=None, mask=None):
        Xt, At = inputs
        # Xt = self.encoder([Xt, At])
        t_emb = self.temporal_enc(Xt)
        # t_emb = self.temporal_enc2(t_emb)
        temp_enc, t_adjs = self.decoder([t_emb, At])
        return temp_enc, t_adjs


class RNNBIL(k.models.Model):
    def __init__(self):
        super(RNNBIL, self).__init__()
        self.rnn_lower = k.layers.SimpleRNN(
            units=10,
            return_state=True,
            return_sequences=True,
            stateful=False,
            time_major=True,
            activation="relu",
            dropout=0.5,
            recurrent_dropout=0.5
        )
        self.rnn_upper = k.layers.SimpleRNN(
            units=10,
            return_state=True,
            return_sequences=True,
            stateful=False,
            time_major=True,
            activation="relu",
            dropout=0.5,
            recurrent_dropout=0.5
        )
        self.encoder_lower = GATConv(
            channels=5,
            attn_heads=10,
            add_self_loops=False,
            activation="relu",
            concat_heads=True,
        )
        self.encoder_upper = GATConv(
            channels=5,
            attn_heads=10,
            add_self_loops=False,
            activation="relu",
            concat_heads=True,
        )
        self.mix_prev_curr = k.layers.Dense(10, "relu")
        self.decoder = BilinearDecoderDense(activation="relu", qr=True)
        self.ln = LayerNormalization()
        self.initial = True
        self.state_lower = None
        self.state_upper = None

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        a_lower = tf.linalg.LinearOperatorLowerTriangular(a).to_dense()
        a_lower = a_lower + tf.transpose(a_lower)
        a_upper = tf.linalg.LinearOperatorLowerTriangular(tf.transpose(a)).to_dense()
        a_upper = tf.transpose(a_upper)
        x_enc_lower = self.encoder_lower([x, a_lower])
        x_enc_upper = self.encoder_upper([x, a_upper])
        # x_lower_upper = tf.concat([x_enc_lower, x_enc_upper], -1)
        # x_enc = self.encoder_lower([x, a])
        # x_prime, _ = self.rnn(tf.expand_dims(x_lower_upper, 0))
        if self.initial:
            init_state = tf.constant(np.zeros(shape=(x.shape[0], 10), dtype=np.float32))
            x_prime_lower, state_lower = self.rnn_lower(
                tf.expand_dims(x_enc_lower, 0), initial_state=init_state
            )
            x_prime_upper, state_upper = self.rnn_upper(
                tf.expand_dims(x_enc_upper, 0), initial_state=init_state
            )
            self.state_lower = state_lower
            self.state_upper = state_upper
            self.initial = False
        else:
            x_prime_lower, state_lower = self.rnn_lower(
                tf.expand_dims(x_enc_lower, 0), initial_state=self.state_lower
            )
            x_prime_upper, state_upper = self.rnn_upper(
                tf.expand_dims(x_enc_upper, 0), initial_state=self.state_upper
            )
            self.state_lower = state_lower
            self.state_upper = state_upper
        # x_prime, _ = self.rnn(tf.expand_dims(x_enc, 0))
        x_prime = tf.concat([x_prime_lower, x_prime_upper], -1)
        x_prime = tf.squeeze(x_prime)
        x_prime = self.ln(x_prime)
        # x_enc = self.ln(x_enc)
        # x_prev_curr = tf.concat([x_lower_upper, x_prime], -1)
        # x_prev_curr = tf.concat([x_enc, x_prime], -1)
        # x_temp = self.mix_prev_curr(x_prev_curr)
        # x_enc, pred_adj = self.decoder([x_temp, a])
        x_enc, pred_adj = self.decoder([x_prime, a])
        return x_enc, pred_adj


if __name__ == "__main__":
    CUDA_VISIBLE_DEVICES = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--t", type=int, default=15)
    parser.add_argument("--n", type=int, default=174)

    args = parser.parse_args()
    synthetic = args.synthetic
    t = args.t
    n = args.n

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

        data_slice = tf.sparse.slice(data_sp, (20, 10, 0, 0), (t, 1, n, n))
        data_dense = tf.sparse.reduce_sum(data_slice, 1)
        data_dense = tf.math.log(data_dense)
        At = tf.clip_by_value(data_dense, 0.0, 1e12)
        # At = np.diff(At.numpy(), 0)[:-1]

    # pos = _get_positional_encoding_matrix(t, n)
    Xt = [np.eye(n)] * t
    Xt = np.asarray(Xt, dtype=np.float32)
    """for i in range(n):
        Xt[:,i,:] += pos
    """
    X = np.eye(n)
    # model = TGNN()
    # model = LSTMBIL()
    model = RNNBIL()
    # l = k.losses.MeanSquaredError()
    l = square_loss
    lb = k.losses.binary_crossentropy
    loss_hist = []
    epochs = 50
    bar = tqdm.tqdm(total=epochs * t, position=0, leave=True, desc="Training")

    all_preds = []
    all_embs = []

    def l_sheduler(l0, i, decay):
        return l0*np.exp(-decay*i)

    for i in range(epochs):
        preds = []
        embs = []
        tot_loss = []
        # optimizer = k.optimizers.Adam(l_sheduler(0.01, i, 0.1))
        optimizer = k.optimizers.Adam(0.001)
        with tf.GradientTape(persistent=True) as tape:
            for i in range(t - 1):
                x, a = model([X, At[i]])
                preds.append(a)
                embs.append(x)
                loss = l(a, At[i + 1])
                if i!=0:
                    loss += temporal_embedding_smoothness(x, embs[i-1])
                tot_loss.append(loss)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                bar.update(1)
        model.initial = True
        all_preds.append(preds)
        all_embs.append(embs)
        loss_hist.append(np.mean(tot_loss))

    plt.plot(loss_hist)
    plt.show()
    a = np.asarray(all_preds)
    xT = np.swapaxes(np.asarray(all_embs)[-1], 0, 1)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i in range(n):
        ax.plot3D(xT[i, :, 0], xT[i, :, 1], xT[i, :, 2])

    fig, ax = plt.subplots(1, 3, figsize=(6, 15))
    t = ax[0].imshow(At[0])
    ax[0].set_title("True")
    fig.colorbar(t, ax=ax[0], fraction=0.05)
    p = ax[1].imshow(a[-1][0])
    ax[1].set_title("Pred")
    fig.colorbar(p, ax=ax[1], fraction=0.05)
    d = ax[2].imshow(At[0] - a[-1][0])
    ax[2].set_title("Diff")
    fig.colorbar(d, ax=ax[2], fraction=0.05)

    fig, ax = plt.subplots(1, 3, figsize=(6, 15))
    t = ax[0].imshow(At[-1])
    ax[0].set_title("True")
    fig.colorbar(t, ax=ax[0], fraction=0.05)
    p = ax[1].imshow(a[-1][-1])
    ax[1].set_title("Pred")
    fig.colorbar(p, ax=ax[1], fraction=0.05)
    d = ax[2].imshow(At[-1] - a[-1][-1])
    ax[2].set_title("Diff")
    fig.colorbar(d, ax=ax[2], fraction=0.05)


    plt.show()
