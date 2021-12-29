import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt
import tqdm
import argparse
import os
import pickle as pkl
from src.modules.layers import BatchBilinearDecoderDense
from src.modules.utils import sum_gradients
from spektral.layers.convolutional import GATConv
import matplotlib.animation as animation


class RnnBil(k.models.Model):
    def __init__(
        self,
        gat_channels=5,
        gat_heads=5,
        rnn_units=5,
        activation="relu",
        dropout_rate=0.5,
        residual_con=False,
        rnn_type="rnn"
    ):
        super(RnnBil, self).__init__()
        self.gat_channels = gat_channels
        self.gat_heads = gat_heads
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.residual_con = residual_con
        self.initial = True
        self.rnn_type = rnn_type
        if self.rnn_type == "rnn":
            self.rnn = k.layers.SimpleRNN(
                units=self.rnn_units,
                activation=self.activation,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                return_sequences=True,
                time_major=True,
            )
        elif self.rnn_type == "lstm":
            self.rnn = k.layers.LSTM(
                units=self.rnn_units,
                activation=self.activation,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                return_sequences=True,
                time_major=True,
            )
        elif self.rnn_type == "gru":
            self.rnn = k.layers.GRU(
                units=self.rnn_units,
                activation=self.activation,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                return_sequences=True,
                time_major=True,
            )
        else:
            raise TypeError(f"No rnn of type {self.rnn_type} found")

        self.encoder = GATConv(
            channels=self.gat_channels,
            attn_heads=self.gat_heads,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            concat_heads=True,
        )
        self.decoder = BatchBilinearDecoderDense(activation=self.activation, qr=True)
        self.ln_enc = k.layers.LayerNormalization()
        self.ln_rnn = k.layers.LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        """
        inputs:
            - X: shape(TxNxd)
            - A: shape(TxNxN)
        """
        x, a = inputs
        # encode and normalize
        x_enc = self.encoder([x, a])
        x_enc = self.ln_enc(x_enc)
        # recurrent connection and normalize
        x_rec = self.rnn(x_enc)
        x_rec = self.ln_rnn(x_rec)
        if self.residual_con:
            x_rec = tf.concat([x_enc, x_rec], -1)
        _, A = self.decoder([x_rec, a])
        return x_rec, A


class GatBil(k.models.Model):
    def __init__(
        self,
        gat_channels=5,
        gat_heads=15,
        activation="relu",
        dropout_rate=0.5,
        lag=5,
        residual_con=True,
    ):
        super(GatBil, self).__init__()
        self.gat_channels = gat_channels
        self.gat_heads = gat_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.lag = lag
        self.residual_con = residual_con

        self.self_attention = GATConv(
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
        self.decoder = BatchBilinearDecoderDense(activation=self.activation, qr=False)
        self.ln_enc = k.layers.LayerNormalization(axis=-1)
        self.ln_rec = k.layers.LayerNormalization(axis=-1)

    # Generate temporal masking tensor
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
        a_mask = self.generate_list_lower_triang(n, t, self.lag)  # NxTxT
        x_enc = self.encoder([x, a])
        x_enc = self.ln_enc(x_enc)
        x_enc_t = tf.transpose(x_enc, (1, 0, 2))  # NxTxd
        x_rec = self.self_attention([x_enc_t, a_mask])
        x_rec = self.ln_rec(x_rec)
        x_rec = tf.transpose(x_rec, (1, 0, 2))  # TxNxd
        if self.residual_con:
            x_rec = tf.concat([x_enc, x_rec], -1)
        _, A = self.decoder([x_rec, a])
        return x_rec, A


if __name__ == "__main__":
    tf.get_logger().setLevel("INFO")
    CUDA_VISIBLE_DEVICES = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--t", type=int, default=15)
    parser.add_argument("--n", type=int, default=174)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--rnn_type", type=str, default="gat")
    args = parser.parse_args()
    synthetic = args.synthetic
    t = args.t
    n = args.n
    sparse = args.sparse
    batch_size = args.batch_size
    epochs = args.epochs
    train_perc = args.train
    save = args.save
    rnn_type = args.rnn_type

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

        data_slice = tf.sparse.slice(data_sp, (0, 0, 0, 0), (data_sp.shape[0], 21, n, n))

        data_dense = tf.sparse.reduce_sum(data_slice, 1)
        data_dense = tf.math.log(data_dense)
        At = tf.clip_by_value(data_dense, 0.0, 1e12)
        # At = At[1:] - At[:-1]
        # At = np.diff(At.numpy(), 0)[:-1]

    # pos = _get_positional_encoding_matrix(t, n)
    Xt = [np.eye(n)] * data_sp.shape[0]
    Xt = np.asarray(Xt, dtype=np.float32)
    Xt_train = Xt[:-1]

    # Train & Test Edge Masking
    train_mask = np.random.choice(
        (0, 1), p=(1 - train_perc, train_perc), size=(data_sp.shape[0], n, n)
    )
    test_mask = 1 - train_mask
    At_train_in = At[:-1] * train_mask[:-1]
    At_train_out = At[1:] * train_mask[1:]
    At_test_in = At[:-1] * test_mask[:-1]
    At_test_out = At[1:] * test_mask[1:]

    if rnn_type == "gat":
        model = GatBil(lag=6, residual_con=False, dropout_rate=0.1)
    elif rnn_type == "rnn":
        model = RnnBil(rnn_type="rnn")
    elif rnn_type == "lstm":
        model = RnnBil(rnn_type="lstm")
    elif rnn_type == "gru":
        model = RnnBil(rnn_type="gru")
    else:
        raise TypeError(f"No rnn of type {rnn_type} found")

    #from_time = np.asarray([np.random.choice(np.arange(60 - t - 1), size=(60-t-1), replace=False) for _ in range(epochs)])
    #from_time = np.tile(np.arange(batch_size)+10, epochs).reshape(epochs, -1)
    #to_time = from_time + t
    optimizer = k.optimizers.Adam(0.0005)
    loss_hist = []
    bar = tqdm.tqdm(total=epochs, leave=True)
    for i in range(epochs):
        '''batch_grad = []
        batch_loss = []
        for b in range(batch_size):'''
        with tf.GradientTape() as tape:
            f = 0#from_time[i, b]
            t = 60 #to_time[i, b]
            x_train, a_train = model([Xt_train[f:t], At_train_in[f:t]])
            a_train = a_train * train_mask[1:][f:t]
            x_test, a_test = model([Xt_train[f:t], At_test_in[f:t]])
            a_test = a_test * test_mask[1:][f:t]
            At_flat_train = tf.reshape(At_train_out[f:t], [-1])
            a_flat_train = tf.reshape(a_train, [-1])
            At_flat_test = tf.reshape(At_test_out[f:t], [-1])
            a_flat_test = tf.reshape(a_test, [-1])
            loss_train = k.losses.mean_squared_error(At_flat_train, a_flat_train)
            loss_test = k.losses.mean_squared_error(At_flat_test, a_flat_test)
            # loss_hist.append((loss_train, loss_test))
            #batch_loss.append((loss_train, loss_test))
            grads = tape.gradient(loss_train, model.trainable_weights)
            #batch_grad.append(grads)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        #mean_grads = sum_gradients(batch_grad, "mean")
        #loss_hist.append(np.mean(batch_loss, 0))
        loss_hist.append((loss_train, loss_test))
        #optimizer.apply_gradients(zip(mean_grads, model.trainable_weights))
        bar.update(1)

    model.save_weights("./RNN-GATBIL/Model")
    losses = plt.plot(loss_hist)
    plt.legend(iter(losses), ("Train", "Test"))
    fig, ax = plt.subplots(1, 2)

    def update(i):
        ax[0].clear()
        ax[1].clear()
        fig.suptitle(f"Year {10 + i}")
        ax[0].set_title("Pred")
        t = ax[0].imshow(a_train[i], animated=True)
        ax[1].set_title("True")
        p = ax[1].imshow(At_train_out[i], animated=True)
        # fig.colorbar(t, ax=ax[0], fraction=0.05)
        # fig.colorbar(p, ax=ax[1], fraction=0.05)

    anim = animation.FuncAnimation(fig, update, frames=len(a_train) - 1, repeat=True)
    '''writer = animation.FFMpegWriter(50)
    anim.save(
        "A:\\Users\\Claudio\\Documents\\PROJECTS\\Master-Thesis\\src\\Tests\\Figures\\RNN-GATBIL\\animation.gif",
        writer="pillow",
    )'''
    # print("mp4 saved to {}".format(os.path.join("./animation.mp4")))

    plt.show()
