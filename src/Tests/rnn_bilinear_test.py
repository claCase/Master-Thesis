import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt
import tqdm
import argparse
import os
import pickle as pkl
from src.modules.layers import BatchBilinearDecoderDense
from spektral.layers.convolutional import GATConv


class RnnBil(k.models.Model):
    def __init__(self,
                 gat_channels=5,
                 gat_heads=5,
                 rnn_units=25,
                 activation="relu",
                 dropout_rate=0.5,
                 residual_con=False):
        super(RnnBil, self).__init__()
        self.gat_channels = gat_channels
        self.gat_heads = gat_heads
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.residual_con = residual_con
        self.initial = True

        self.rnn_lower = k.layers.SimpleRNN(
            units=self.rnn_units,
            activation=self.activation,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            return_sequences=True,
            time_major=True
        )
        self.rnn_upper = k.layers.SimpleRNN(
            units=self.rnn_units,
            activation=self.activation,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            return_sequences=True,
            time_major=True
        )
        self.encoder_lower = GATConv(
            channels=self.gat_channels,
            attn_heads=self.gat_heads,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            concat_heads=True,
        )
        self.encoder_upper = GATConv(
            channels=self.gat_channels,
            attn_heads=self.gat_heads,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            concat_heads=True,
        )
        self.decoder = BatchBilinearDecoderDense(
            activation=self.activation,
            qr=True
        )
        self.ln_enc_lower = k.layers.LayerNormalization()
        self.ln_enc_upper = k.layers.LayerNormalization()
        self.ln_rnn_lower = k.layers.LayerNormalization()
        self.ln_rnn_upper = k.layers.LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        """
        inputs:
            - X: shape(TxNxd)
            - A: shape(TxNxN)
        """
        x, a = inputs
        a_lower = tf.linalg.LinearOperatorLowerTriangular(a).to_dense()
        a_lower = a_lower + tf.transpose(a_lower)
        a_upper = tf.linalg.LinearOperatorLowerTriangular(tf.transpose(a)).to_dense()
        a_upper = a_upper + tf.transpose(a_upper)
        # encode and normalize
        x_enc_lower = self.encoder_lower([x, a_lower])
        x_enc_lower = self.ln_enc_lower(x_enc_lower)
        x_enc_upper = self.encoder_upper([x, a_upper])
        x_enc_upper = self.ln_enc_upper(x_enc_upper)
        x_enc = tf.concat([x_enc_lower, x_enc_upper])

        # recurrent connection and normalize
        if self.initial:
            initial_state = np.zeros(shape=(x.shape[1], self.rnn_units))
            x_rec_lower = self.rnn_lower(x_enc_lower, initial_state=initial_state)
            x_rec_upper = self.rnn_lower(x_enc_upper, initial_state=initial_state)
            self.initial = False
        else:
            x_rec_lower = self.rnn_lower(x_enc_lower)
            x_rec_upper = self.rnn_lower(x_enc_upper)

        x_rec_lower = self.ln_rnn_lower(x_rec_lower)
        x_rec_upper = self.ln_rnn_upper(x_rec_upper)
        x_rec = tf.concat([x_rec_lower, x_rec_upper], -1)
        if self.residual_con:
            x_rec = tf.concat([x_enc, x_rec], -1)
        _, A = self.decoder([x_rec, a])
        return x_rec, A


class GatBil(k.models.Model):
    def __init__(self,
                 gat_channels=10,
                 gat_heads=10,
                 rnn_units=50,
                 activation="relu",
                 dropout_rate=0.5,
                 residual_con=True):
        super(GatBil, self).__init__()
        self.gat_channels = gat_channels
        self.gat_heads = gat_heads
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.residual_con = residual_con

        self.rnn = GATConv(
            channels=self.gat_channels,
            attn_heads=self.gat_heads,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            concat_heads=True,
        )
        self.encoder = GATConv(
            channels=self.gat_channels,
            attn_heads=self.gat_heads,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            concat_heads=True,
        )
        self.decoder = BatchBilinearDecoderDense(
            activation=self.activation,
            qr=False
        )

    @staticmethod
    def generate_list_lower_triang(batch, t, lag):
        lower_adj = np.tril(np.ones(shape=(t, t)))
        prev = np.zeros(shape=(lag, t))
        sub_lower = np.vstack([prev, lower_adj])[:-lag]
        lower_adj = lower_adj - sub_lower
        return np.asarray([lower_adj] * batch)

    def call(self, inputs, training=None, mask=None):
        """
        inputs:
            - X: shape(TxNxd)
            - A: shape(TxNxN)
        """
        x, a = inputs
        t, n, _ = a.shape
        a_mask = self.generate_list_lower_triang(n, t, 4)  # NxTxT
        x_enc = self.encoder([x, a])
        x_enc_t = tf.transpose(x_enc, (1, 0, 2))  # NxTxd
        x_rec = self.rnn([x_enc_t, a_mask])
        x_rec = tf.transpose(x_rec, (1, 0, 2))  # TxNxd
        if self.residual_con:
            x_rec = tf.concat([x_enc, x_rec], -1)
        _, A = self.decoder([x_rec, a])
        return x_rec, A


if __name__ == "__main__":
    tf.get_logger().setLevel('INFO')
    CUDA_VISIBLE_DEVICES = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--t", type=int, default=35)
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

        data_slice = tf.sparse.slice(data_sp, (0, 10, 0, 0), (t, 1, n, n))

        data_dense = tf.sparse.reduce_sum(data_slice, 1)
        data_dense = tf.math.log(data_dense)
        At = tf.clip_by_value(data_dense, 0.0, 1e12)

        # At = At[1:] - At[:-1]
        # At = np.diff(At.numpy(), 0)[:-1]

    # pos = _get_positional_encoding_matrix(t, n)
    Xt = [np.eye(n)] * t
    Xt = np.asarray(Xt, dtype=np.float32)

    optimizer = k.optimizers.Adam(0.001)
    loss_hist = []
    epochs = 200
    #model = RnnBil()
    model = GatBil()
    bar = tqdm.tqdm(total=epochs, leave=True)
    for i in range(epochs):
        with tf.GradientTape() as tape:
            x_, a_ = model([Xt[:-1], At[:-1]])
            At_flat = tf.reshape(At[1:], [-1])
            #a_mask = tf.where(At_flat > 0, 1.0, 0.0)
            a_flat = tf.reshape(a_, [-1])
            #a_flat_masked = a_flat * a_mask
            loss = k.losses.mean_squared_error(At_flat, a_flat)
            loss_hist.append(loss)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        bar.update(1)
    plt.plot(loss_hist)
    fig, ax = plt.subplots(2, 1)
    t = ax[0].imshow(At[-1].numpy())
    ax[0].set_title("True")
    fig.colorbar(t, ax=ax[0], fraction=0.05)
    p = ax[1].imshow(a_[-1])
    ax[1].set_title("Pred")
    fig.colorbar(p, ax=ax[1], fraction=0.05)

    plt.show()
