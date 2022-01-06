import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt
import tqdm
import argparse
import os
import pickle as pkl
from src.modules.layers import BatchBilinearDecoderDense, SelfAttention
from src.modules.utils import sum_gradients, get_positional_encoding_matrix
from src.modules.losses import TemporalSmoothness
from spektral.layers.convolutional import GATConv
import matplotlib.animation as animation


class RnnBil(k.models.Model):
    def __init__(
            self,
            gat_channels=5,
            gat_heads=5,
            rnn_units=15,
            activation="relu",
            output_activation="relu",
            dropout_rate=0.5,
            residual_con=True,
            rnn_type="rnn",
    ):
        super(RnnBil, self).__init__()
        self.gat_channels = gat_channels
        self.gat_heads = gat_heads
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.output_activation = output_activation
        self.residual_con = residual_con
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
        self.decoder = BatchBilinearDecoderDense(activation=self.output_activation, qr=True)
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
            gat_heads=10,
            activation="relu",
            output_activation="relu",
            dropout_rate=0.5,
            lag=5,
            residual_con=True,
            return_attn_coeff=False,
            rnn_type="transformer"
    ):
        super(GatBil, self).__init__()
        self.gat_channels = gat_channels
        self.gat_heads = gat_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.output_activation = output_activation
        self.lag = lag
        self.residual_con = residual_con
        self.return_attn_coef = return_attn_coeff
        self.rnn_type = rnn_type

        if self.rnn_type == "gat":
            self.self_attention = GATConv(
                channels=self.gat_channels,
                attn_heads=self.gat_heads,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                concat_heads=True,
                return_attn_coef=self.return_attn_coef,
            )
        elif self.rnn_type == "transformer":
            self.self_attention = SelfAttention(
                channels=self.gat_channels,
                attn_heads=self.gat_heads,
                dropout_rate=self.dropout_rate,
                lags=self.lag,
                return_attn=self.return_attn_coef,
                concat_heads=True
            )
        self.encoder = GATConv(
            channels=self.gat_channels,
            attn_heads=self.gat_heads,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            concat_heads=True,
        )
        self.decoder = BatchBilinearDecoderDense(activation=self.output_activation, qr=True)
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
        x_enc = self.ln_enc(x_enc)  # TxNxd
        x_enc_t = tf.transpose(x_enc, (1, 0, 2))  # NxTxd
        if self.return_attn_coef:
            x_rec, attn_coeff = self.self_attention([x_enc_t, a_mask])
        else:
            x_rec = self.self_attention([x_enc_t, a_mask])
        x_rec = self.ln_rec(x_rec)
        x_rec = tf.transpose(x_rec, (1, 0, 2))  # TxNxd
        if self.residual_con:
            x_rec = tf.concat([x_enc, x_rec], -1)
        _, A = self.decoder([x_rec, a])
        if self.return_attn_coef:
            return x_rec, A, attn_coeff
        else:
            return x_rec, A


if __name__ == "__main__":
    tf.get_logger().setLevel("INFO")
    CUDA_VISIBLE_DEVICES = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.chdir("../../")

    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--t", type=int, default=15)
    parser.add_argument("--n", type=int, default=174)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--rnn_type", type=str, default="gat")
    parser.add_argument("--lag", type=int, default=50)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--smooth", type=str, default="euclidian")
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
    lag = args.lag
    binary = args.binary
    smoothness = args.smooth
    if binary:
        output_activation = "sigmoid"
    else:
        output_activation = "relu"
    if synthetic:
        input_shape_nodes = (t, n, n)
        input_shape_adj = (t, n, n)
        At = np.random.choice((0, 1), size=input_shape_adj)
        if not binary:
            At = np.asarray(At, dtype=np.float32)
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

        data_slice = tf.sparse.slice(
            data_sp, (0, 0, 0, 0), (data_sp.shape[0], 31, n, n)
        )

        data_dense = tf.sparse.reduce_sum(data_slice, 1)
        data_dense = tf.math.log(data_dense)
        At = tf.clip_by_value(data_dense, 0.0, 1e12)
        if binary:
            At = tf.constant(np.where(At == 0.0, 0.0, 1.0), dtype=tf.float32)
        # At = At[1:] - At[:-1]
        # At = np.diff(At.numpy(), 0)[:-1]

    pos = get_positional_encoding_matrix(data_sp.shape[0], n)  # TxN
    # pos = np.expand_dims(pos, 1)
    Xt = [np.eye(n)] * data_sp.shape[0]
    Xt = np.asarray(Xt, dtype=np.float32)  # TxNxN
    '''X = np.empty(shape=(data_sp.shape[0], n, 2*n))
    #Xt = np.add(Xt, pos)
    for i, x in enumerate(Xt):
        pos_tile = np.tile(pos[i], n).reshape(-1, n)
        X[i] = np.concatenate([Xt[i], pos_tile], -1)
    Xt = X'''
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

    def train(temp_loss=True, rnn_type="gat", binary=False):
        dropout_rate = 0.1
        if rnn_type == "gat":
            model = GatBil(
                lag=lag, residual_con=False, dropout_rate=dropout_rate, return_attn_coeff=True, rnn_type="gat",
                output_activation=output_activation
            )
        elif rnn_type == "transformer":
            model = GatBil(
                lag=lag, residual_con=False, dropout_rate=dropout_rate, return_attn_coeff=True, rnn_type="transformer",
                output_activation=output_activation
            )
        elif rnn_type == "rnn":
            model = RnnBil(rnn_type="rnn", dropout_rate=dropout_rate)
        elif rnn_type == "lstm":
            model = RnnBil(rnn_type="lstm", dropout_rate=dropout_rate)
        elif rnn_type == "gru":
            model = RnnBil(rnn_type="gru", dropout_rate=dropout_rate)
        else:
            raise TypeError(f"No rnn of type {rnn_type} found")

        if temp_loss:
            temp_smooth = TemporalSmoothness(temp_loss)
        loss_hist = []
        optimizer = k.optimizers.Adam(0.0005)
        bar = tqdm.tqdm(total=epochs, leave=True, position=0)
        for i in range(epochs):
            """batch_grad = []
            batch_loss = []
            for b in range(batch_size):"""
            with tf.GradientTape() as tape:
                f = 10  # from_time[i, b]
                t = 60  # to_time[i, b]
                if rnn_type == "gat" or rnn_type == "transformer":
                    x_train, a_train, attn_coeff_train = model(
                        [Xt_train[f:t], At_train_in[f:t]]
                    )
                else:
                    x_train, a_train = model([Xt_train[f:t], At_train_in[f:t]])
                if rnn_type == "gat" or rnn_type == "transformer":
                    x_test, a_test, attn_coeff_test = model(
                        [Xt_train[f:t], At_test_in[f:t]]
                    )
                else:
                    x_test, a_test = model([Xt_train[f:t], At_test_in[f:t]])
                a_train = a_train * train_mask[1:][f:t]
                a_test = a_test * test_mask[1:][f:t]
                At_flat_train = tf.reshape(At_train_out[f:t], [-1])
                a_flat_train = tf.reshape(a_train, [-1])
                At_flat_test = tf.reshape(At_test_out[f:t], [-1])
                a_flat_test = tf.reshape(a_test, [-1])
                if binary:
                    loss_train = k.losses.binary_crossentropy(At_flat_train, a_flat_train)
                    loss_test = k.losses.binary_crossentropy(At_flat_test, a_flat_test)
                else:
                    loss_train = k.losses.mean_squared_error(At_flat_train, a_flat_train)
                    loss_test = k.losses.mean_squared_error(At_flat_test, a_flat_test)
                if temp_loss and not binary:
                    loss_train += temp_smooth([Xt_train, At_train_in])
                # loss_hist.append((loss_train, loss_test))
                # batch_loss.append((loss_train, loss_test))
                grads = tape.gradient(loss_train, model.trainable_weights)
                # batch_grad.append(grads)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            # mean_grads = sum_gradients(batch_grad, "mean")
            # loss_hist.append(np.mean(batch_loss, 0))
            loss_hist.append((loss_train, loss_test))
            # optimizer.apply_gradients(zip(mean_grads, model.trainable_weights))
            bar.update(1)
        return model, attn_coeff_train, attn_coeff_test, a_train, a_test, loss_hist

    loss_histories = []
    test_losses = (False, "euclidian", "rotation", "angle")
    for loss_type in test_losses:
        model, attn_coeff_train, attn_coeff_test, a_train, a_test, loss_hist = train(temp_loss=loss_type,
                                                                                     rnn_type=rnn_type,
                                                                                     binary=binary)
        loss_histories.append(loss_hist)

    # model.save_weights("./RNN-GATBIL/Model")
    losses_plots = []
    colors = plt.get_cmap("Set1")
    for i, loss in enumerate(loss_histories):
        losses_plot = plt.plot(loss, color=colors(i))
        losses_plots.append(losses_plot)

    legend_names = [(f"Train {loss_name}", f"Test {loss_name}") for loss_name in test_losses]
    legend_names = [name for names in legend_names for name in names]
    plots = [plot for plots in losses_plots for plot in plots]
    plt.legend(plots, legend_names)

    if rnn_type == "gat":
        row, col = 2, 2
        fig2, axs = plt.subplots(row, col)
        mean_attn = np.mean(attn_coeff_train.numpy(), axis=2)
        mean_attn = np.squeeze(mean_attn)
        mean_attn_idx = np.random.choice(np.arange(170), size=4)
        mean_attn = mean_attn[mean_attn_idx]
        counter = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(mean_attn[counter])
                counter += 1

    if rnn_type == "transformer":
        row, col = 2, 2
        fig2, axs = plt.subplots(row, col)
        mean_attn = np.mean(attn_coeff_train.numpy(), axis=1)
        mean_attn = np.squeeze(mean_attn)
        counter = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(mean_attn[counter])
                counter += 1

    fig, ax = plt.subplots(1, 2)


    def update(i):
        ax[0].clear()
        ax[1].clear()
        fig.suptitle(f"Year {10 + i}")
        ax[0].set_title("Pred")
        t = ax[0].imshow(a_train[i], animated=True)
        ax[1].set_title("True")
        p = ax[1].imshow(At_train_out[i], animated=True)


    anim = animation.FuncAnimation(fig, update, frames=len(a_train) - 1, repeat=True)

    fig1, ax1 = plt.subplots(1, 1)

    if binary:
        threshold = -0.01
    else:
        threshold = 0.5

    def update2(i):
        ax1.clear()
        fig1.suptitle(f"Year {10 + i}")
        a_pred = a_train[i].numpy().flatten()
        a_pred = a_pred[a_pred > threshold]
        if binary:
            a_pred = np.asarray(a_pred>0.5, dtype=np.float32)
        a_true = At_train_out[i].numpy().flatten()
        a_true = a_true[a_true > threshold]
        ax1.hist(a_true, bins=60, alpha=0.5, color="blue", density=True, label="True")
        ax1.hist(a_pred, bins=60, alpha=0.5, color="red", density=True, label="Pred")
        ax1.legend()


    anim2 = animation.FuncAnimation(fig1, update2, frames=len(a_train) - 1, repeat=True)

    """writer = animation.FFMpegWriter(50)
    anim.save(
        "A:\\Users\\Claudio\\Documents\\PROJECTS\\Master-Thesis\\src\\Tests\\Figures\\RNN-GATBIL\\animation.gif",
        writer="pillow",
    )"""
    # print("mp4 saved to {}".format(os.path.join("./animation.mp4")))
    plt.show()
