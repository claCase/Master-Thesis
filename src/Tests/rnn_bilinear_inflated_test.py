import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow.keras as k
import matplotlib.pyplot as plt
import tqdm
import argparse
import os
import pickle as pkl
from src.modules.layers import BatchBilinearDecoderDense, SelfAttention
from src.modules.utils import sum_gradients, get_positional_encoding_matrix
from src.modules.losses import TemporalSmoothness, zero_inflated_lognormal_loss, zero_inflated_logGamma_loss
from src.modules.utils import zero_inflated_lognormal, zero_inflated_logGamma
from spektral.layers.convolutional import GATConv, GCNConv
from spektral.utils.convolution import gcn_filter
from tensorflow.keras import layers as l
from tensorflow.keras import models as m
import matplotlib.animation as animation


class RnnBil(k.models.Model):
    def __init__(
            self,
            gat_channels=10,
            gat_heads=15,
            rnn_units=25,
            activation="relu",
            output_activation="relu",
            dropout_rate=0.5,
            residual_con=True,
            rnn_type="rnn",
            qr=False
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
        self.qr = qr
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
        self.decoder_mu = BatchBilinearDecoderDense(activation=None, qr=self.qr)
        # self.decoder_sigma = BatchBilinearDecoderDense(activation=None, qr=self.qr)
        self.decoder_p = BatchBilinearDecoderDense(activation=None, qr=self.qr)

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
        x_enc = self.encoder([x, a])  # TxNxd
        x_enc = self.ln_enc(x_enc)
        # recurrent connection and normalize
        x_rec = self.rnn(x_enc)  # TxNxd
        x_rec = self.ln_rnn(x_rec)
        if self.residual_con:
            x_rec = tf.concat([x_enc, x_rec], -1)
        _, mu = self.decoder_mu(x_rec)
        # _, sigma = self.decoder_sigma([x_rec, a])
        sigma = tf.ones_like(mu)
        _, p = self.decoder_p(x_rec)  # TxNxN
        return x_rec, mu, sigma, p


class GatBil(m.Model):
    def __init__(
            self,
            gat_channels=10,
            gat_heads=15,
            activation="relu",
            output_activation="softplus",
            dropout_rate=0.1,
            lag=5,
            residual_con=True,
            return_attn_coeff=False,
            rnn_type="transformer",
            qr=False,
            t_pos_enc=False,
            constant_sigma=True,
            encoder_type="gcn",
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
        self.qr = qr
        self.t_pos_enc = t_pos_enc
        self.constant_sigma = constant_sigma
        self.encoder_type = encoder_type

        if self.rnn_type == "gat":
            self.self_attention_mu = GATConv(
                channels=self.gat_channels,
                attn_heads=self.gat_heads,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                concat_heads=True,
                return_attn_coef=self.return_attn_coef,
            )
            if not self.constant_sigma:
                self.self_attention_sigma = GATConv(
                    channels=self.gat_channels,
                    attn_heads=self.gat_heads,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    concat_heads=True,
                    return_attn_coef=self.return_attn_coef,
                )
        elif self.rnn_type == "transformerK":
            self.self_attention_mu = l.MultiHeadAttention(
                num_heads=self.gat_heads,
                key_dim=self.gat_channels,
                value_dim=self.gat_channels,
                dropout=self.dropout_rate,
                use_bias=True
            )
            if not self.constant_sigma:
                self.self_attention_sigma = l.MultiHeadAttention(
                    num_heads=self.gat_heads,
                    key_dim=self.gat_channels,
                    value_dim=self.gat_channels,
                    dropout=self.dropout_rate,
                    use_bias=True
                )
        elif self.rnn_type == "transformer":
            self.self_attention_mu = SelfAttention(
                channels=self.gat_channels,
                attn_heads=self.gat_heads,
                dropout_rate=self.dropout_rate,
                return_attn=self.return_attn_coef,
                concat_heads=True
            )
            if not self.constant_sigma:
                self.self_attention_sigma = SelfAttention(
                    channels=self.gat_channels,
                    attn_heads=self.gat_heads,
                    dropout_rate=self.dropout_rate,
                    return_attn=self.return_attn_coef,
                    concat_heads=True
                )

        if not self.constant_sigma:
            self.encoder_sigma = GATConv(
                channels=self.gat_channels,
                attn_heads=self.gat_heads,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                concat_heads=True,
            )
            self.decoder_sigma = BatchBilinearDecoderDense(activation=None, qr=self.qr)
            self.ln_enc_sigma = l.LayerNormalization(axis=-1)
            self.ln_rec_sigma = l.LayerNormalization(axis=-1)
        if self.encoder_type == "gcn":
            self.encoder_mu = GCNConv(
                channels=self.gat_channels,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
            )
        else:
            self.encoder_mu = GATConv(
                channels=self.gat_channels,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
            )

        self.decoder_mu = BatchBilinearDecoderDense(activation=None, qr=self.qr)
        self.ln_enc_mu = l.LayerNormalization(axis=-1)
        self.ln_rec_mu = l.LayerNormalization(axis=-1)
        self.decoder_p = BatchBilinearDecoderDense(activation=None, qr=self.qr)

    # Generate temporal masking tensor
    @staticmethod
    def generate_list_lower_triang(batch, t, lag):
        lower_adj = np.tril(np.ones(shape=(t, t)))
        prev = np.zeros(shape=(lag, t))
        sub_lower = np.vstack([prev, lower_adj])[:-lag]
        lower_adj = lower_adj - sub_lower
        return np.asarray([lower_adj] * batch)

    @staticmethod
    def positional_encoding_matrix(t, n, d_emb):
        pos_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
                if pos != 0
                else np.zeros(d_emb)
                for pos in range(t)
            ]
        )
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        pos_enc = tf.constant([pos_enc] * n, tf.float32)  # NxTxd
        return pos_enc

    def call(self, inputs, training=None, mask=None):
        """
        inputs:
            - X: shape(TxNxd)
            - A: shape(TxNxN)
        """
        x, a = inputs
        t, n, _ = a.shape
        a_mask = self.generate_list_lower_triang(n, t, self.lag)  # NxTxT
        if self.encoder_type == "gcn":
            a = gcn_filter(a.numpy(), symmetric=False)
        x_enc_mu = self.encoder_mu([x, a])  # TxNxd
        x_enc_mu = self.ln_enc_mu(x_enc_mu)  # TxNxd
        x_enc_t_mu = tf.transpose(x_enc_mu, (1, 0, 2))  # NxTxd

        if not self.constant_sigma:
            x_enc_sigma = self.encoder_sigma([x, a])  # TxNxd
            x_enc_sigma = self.ln_enc_sigma(x_enc_sigma)  # TxNxd
            x_enc_t_sigma = tf.transpose(x_enc_sigma, (1, 0, 2))  # NxTxd

        if self.t_pos_enc:
            x_enc_t_mu = tf.concat([x_enc_t_mu, self.positional_encoding_matrix(t, n, 5)], -1)  # NxTxd
            if not self.constant_sigma:
                x_enc_t_sigma = tf.concat([x_enc_t_sigma, self.positional_encoding_matrix(t, n, 5)], -1)  # NxTxd

        if self.return_attn_coef:
            if self.rnn_type == "transformerK":
                x_rec_mu, attn_coeff_mu = self.self_attention_mu(query=x_enc_t_mu, key=x_enc_t_mu, value=x_enc_t_mu,
                                                                 attention_mask=a_mask,
                                                                 return_attention_scores=self.return_attn_coef)
                if not self.constant_sigma:
                    x_rec_sigma, attn_coeff_sigma = self.self_attention_sigma(query=x_enc_t_sigma, key=x_enc_t_sigma,
                                                                              value=x_enc_t_sigma,
                                                                              attention_mask=a_mask,
                                                                              return_attention_scores=self.return_attn_coef)
            else:
                x_rec_mu, attn_coeff_mu = self.self_attention_mu([x_enc_t_mu, a_mask])
                if not self.constant_sigma:
                    x_rec_sigma, attn_coeff_sigma = self.self_attention_sigma([x_enc_t_sigma, a_mask])
        else:
            if self.rnn_type == "transformerK":
                x_rec_mu, attn_coeff_mu = self.self_attention_mu(query=x_enc_t_mu, key=x_enc_t_mu, value=x_enc_t_mu,
                                                                 attention_mask=a_mask,
                                                                 return_attention_scores=self.return_attn_coef)
                if not self.constant_sigma:
                    x_rec_sigma, attn_coeff_sigma = self.self_attention_sigma(query=x_enc_t_sigma, key=x_enc_t_sigma,
                                                                              value=x_enc_t_sigma,
                                                                              attention_mask=a_mask,
                                                                              return_attention_scores=self.return_attn_coef)
            else:
                x_rec_mu = self.self_attention([x_enc_t_mu, a_mask])
                if not self.constant_sigma:
                    x_rec_sigma = self.self_attention([x_enc_t_sigma, a_mask])
        x_rec_mu = self.ln_rec_mu(x_rec_mu)
        x_rec_mu = tf.transpose(x_rec_mu, (1, 0, 2))  # TxNxd
        if not self.constant_sigma:
            x_rec_sigma = self.ln_rec_mu(x_rec_sigma)
            x_rec_sigma = tf.transpose(x_rec_sigma, (1, 0, 2))  # TxNxd
        if self.residual_con:
            x_rec_mu = tf.concat([x_enc_mu, x_rec_mu], -1)
            if not self.constant_sigma:
                x_rec_sigma = tf.concat([x_enc_sigma, x_rec_sigma], -1)
        _, mu = self.decoder_mu(x_rec_mu)
        if not self.constant_sigma:
            _, sigma = self.decoder_sigma(x_rec_sigma)
        else:
            sigma = tf.ones_like(mu) * 4
        _, p = self.decoder_p(x_rec_mu)
        if self.return_attn_coef:
            return x_rec_mu, mu, sigma, p, attn_coeff_mu
        else:
            return x_rec_mu, mu, sigma, p


class CNN1DGATBIL(m.Model):
    def __init__(self, kernel_sizes=[15,15], filter_sizes=[10,10], gat_channels=[5,5], attention_heads=[5,2],
                 concat=True, dropout_rate=0.2, normalize=False):
        super(CNN1DGATBIL, self).__init__()
        self.cnn1d_layers_mu = [l.Conv1D(kernel_size=k, filters=f, name=f"conv_mu_{i}", padding="causal", activation="relu")
                             for i, (k, f) in enumerate(zip(kernel_sizes, filter_sizes))]
        self.cnn1d_layers_sigma = [l.Conv1D(kernel_size=k, filters=f, name=f"conv_s_{i}", padding="causal", activation="relu")
                             for i, (k, f) in enumerate(zip(kernel_sizes, filter_sizes))]
        self.gat_layers = [
            GATConv(channels=c, attn_head=h, concat_heads=concat, dropout_rate=dropout_rate, name=f"gat_{i}",
                    activation="relu")
            for i, (c, h) in enumerate(zip(gat_channels, attention_heads))
        ]
        self.drop1 = l.Dropout(dropout_rate)
        self.drop2 = l.Dropout(dropout_rate)
        self.decoder_mu = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_sigma = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_p = BatchBilinearDecoderDense(activation=None, qr=False)
        self.normalize = normalize
        if self.normalize:
            self.norm_gat = l.LayerNormalization()
            self.norm_cnn = l.LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        for g in self.gat_layers:
            x = g([x, a])
        if self.normalize:
            x = self.norm_gat(x)
        x_mu = x_sigma = tf.transpose(x, perm=(1, 0, 2))  # N x T x d
        for c, s in zip(self.cnn1d_layers_mu, self.cnn1d_layers_sigma):
            x_mu = c(x_mu)
            x_sigma = s(x_sigma)
        if self.normalize:
            x = self.norm_cnn(x)
        x_mu = tf.transpose(x_mu, perm=(1, 0, 2))  # T x N x d
        x_sigma = tf.transpose(x_sigma, perm=(1, 0, 2))  # T x N x d
        mu, a_mu = self.decoder_mu(x_mu)
        # a_mu = tf.expand_dims(a_mu, -1)
        p, a_p = self.decoder_p(x_mu)
        # a_p = tf.expand_dims(a_p, -1)
        sigma, a_sigma = self.decoder_sigma(x_sigma)
        #a_sigma = tf.ones_like(a_p) * 2
        # a_sigma = tf.expand_dims(a_sigma, -1)
        # logits = tf.stack([a_p, a_mu, a_sigma], -1)
        return x, a_mu, a_sigma, a_p


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
    parser.add_argument("--lag", type=int, default=10)
    parser.add_argument("--smooth", type=str, default="euclidian")
    parser.add_argument("--loss", type=str, default="lognormal")
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
    smoothness = args.smooth
    loss_type = args.loss

    if synthetic:
        input_shape_nodes = (t, n, n)
        input_shape_adj = (t, n, n)
        At = np.random.choice((0, 1), size=input_shape_adj)
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
            data_sp, (0, 31, 0, 0), (data_sp.shape[0], 1, n, n)
        )

        data_dense = tf.sparse.reduce_sum(data_slice, 1)
        # data_dense = tf.math.log(data_dense)
        At = tf.clip_by_value(data_dense, 0.0, 1e12)

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


    def train(temp_loss=True, rnn_type="gat", loss_type="lognormal", lag=10, f=10, t=60):
        dropout_rate = 0.1
        if rnn_type in ("gat", "transformerK", "transformer"):
            model = GatBil(gat_channels=5,
                           lag=lag, residual_con=False, dropout_rate=dropout_rate, return_attn_coeff=True,
                           rnn_type=rnn_type,
                           output_activation=None
                           )
        elif rnn_type in ("rnn", "lstm", "gru"):
            model = RnnBil(rnn_type=rnn_type, dropout_rate=dropout_rate)
        elif rnn_type == "cnn":
            model = CNN1DGATBIL(dropout_rate=dropout_rate)
        else:
            raise TypeError(f"No rnn of type {rnn_type} found")

        if temp_loss:
            temp_smooth = TemporalSmoothness(temp_loss)
        loss_hist = []
        optimizer = k.optimizers.Adam(0.0005)
        bar = tqdm.tqdm(total=epochs, leave=True, position=0)
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                if rnn_type in ("gat", "transformer", "transformerK"):
                    x_train, mu_train, sigma_train, p_train, attn_coeff_train = model(
                        [Xt_train[f:t], At_train_in[f:t]]
                    )
                    x_test, mu_test, sigma_test, p_test, attn_coeff_test = model(
                        [Xt_train[f:t], At_test_in[f:t]]
                    )
                elif rnn_type in ("rnn", "lstm", "gru"):
                    x_train, mu_train, sigma_train, p_train = model([Xt_train[f:t], At_train_in[f:t]])
                    x_test, mu_test, sigma_test, p_test = model([Xt_train[f:t], At_test_in[f:t]])
                elif rnn_type == "cnn":
                    x_train, mu_train, sigma_train, p_train = model([Xt_train[f:t], At_train_in[f:t]])
                    x_test, mu_test, sigma_test, p_test = model([Xt_train[f:t], At_test_in[f:t]])
                logits_train = tf.transpose(
                    [p_train * train_mask[1:][f:t], mu_train * train_mask[1:][f:t], sigma_train * train_mask[1:][f:t]],
                    perm=(1, 2, 3, 0))

                if loss_type == "lognormal":
                    loss_train = zero_inflated_lognormal_loss(tf.expand_dims(At_train_in[f:t], -1), logits_train)
                else:
                    loss_train = zero_inflated_logGamma_loss(tf.expand_dims(At_train_in[f:t], -1), logits_train)

                logits_test = tf.transpose(
                    [p_test * test_mask[1:][f:t], mu_test * test_mask[1:][f:t], sigma_test * test_mask[1:][f:t]],
                    perm=(1, 2, 3, 0))
                if loss_type == "lognormal":
                    loss_test = zero_inflated_lognormal_loss(tf.expand_dims(At_test_in[f:t], -1), logits_test)
                else:
                    loss_test = zero_inflated_logGamma_loss(tf.expand_dims(At_test_in[f:t], -1), logits_test)

                loss_train = tf.squeeze(loss_train)
                loss_train = tf.reduce_mean(loss_train)
                loss_test = tf.squeeze(loss_test)
                loss_test = tf.reduce_mean(loss_test)
                if temp_loss:
                    loss_train += temp_smooth([Xt_train, At_train_in])
            grads = tape.gradient(loss_train, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_hist.append((loss_train, loss_test))
            bar.update(1)
        if rnn_type in ("gat", "transformer", "transformerK"):
            return model, attn_coeff_train, attn_coeff_test, logits_train, logits_test, loss_hist
        else:
            return model, logits_train, logits_test, loss_hist


    f = 20
    t = 63
    loss_histories = []
    test_losses_smoothness = (False,)  # "euclidian", "rotation", "angle")
    for smoothness in test_losses_smoothness:
        if rnn_type in ("gat", "transformer", "transformerK"):
            model, attn_coeff_train, attn_coeff_test, logits_train, logits_test, loss_hist = train(temp_loss=smoothness,
                                                                                                   loss_type=loss_type,
                                                                                                   rnn_type=rnn_type,
                                                                                                   lag=lag,
                                                                                                   f=f,
                                                                                                   t=t)
        else:
            model, logits_train, logits_test, loss_hist = train(temp_loss=loss_type,
                                                                rnn_type=rnn_type,
                                                                f=f,
                                                                t=t)
        loss_histories.append(loss_hist)

    # model.save_weights("./RNN-GATBIL/Model")
    losses_plots = []
    colors = plt.get_cmap("Set1")
    for i, loss in enumerate(loss_histories):
        losses_plot = plt.plot(loss, color=colors(i))
        losses_plots.append(losses_plot)

    legend_names = [(f"Train {loss_name}", f"Test {loss_name}") for loss_name in test_losses_smoothness]
    legend_names = [name for names in legend_names for name in names]
    plots = [plot for plots in losses_plots for plot in plots]
    plt.legend(plots, legend_names)
    if save:
        plt.savefig("./src/Tests/Figures/RNN-GATBIL/losses.png")
        plt.close()

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
        if save:
            plt.savefig("./src/Tests/Figures/RNN-GATBIL/attention_coeff_gat.png", fig=fig2)
            plt.close(fig2)
    if rnn_type in ("transformer", "transformerK"):
        row, col = 2, 2
        fig2, axs = plt.subplots(row, col)
        mean_attn = np.mean(attn_coeff_train.numpy(), axis=1)
        mean_attn = np.squeeze(mean_attn)
        counter = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(mean_attn[counter])
                counter += 1
        if save:
            plt.savefig("./src/Tests/Figures/RNN-GATBIL/attention_coeff_transformer.png")
            plt.close(fig2)

    fig, ax = plt.subplots(1, 2)
    if rnn_type in ("gat", "transformer", "transformerK"):
        x_rec, mu, sigma, p, attn = model([Xt, At])
    else:
        x_rec, mu, sigma, p = model([Xt, At])

    logits = tf.transpose([p, mu, sigma], perm=(1, 2, 3, 0))  # TxNxNx3
    if loss_type == "lognormal":
        a_train = zero_inflated_lognormal(logits)
    else:
        a_train = zero_inflated_logGamma(logits)
    a_train = a_train.sample(1)
    # a_train = tf.reduce_mean(a_train, 0)
    a_train = tf.squeeze(a_train)
    a_train = tf.clip_by_value(tf.math.log(a_train), 0.0, 1e12)
    At_train_out = tf.clip_by_value(tf.math.log(At_train_out), 0.0, 1e12)


    def update(i):
        ax[0].clear()
        ax[1].clear()
        fig.suptitle(f"Year {10 + i}")
        ax[0].set_title("Pred")
        t = ax[0].imshow(a_train[i], animated=True)
        ax[1].set_title("True")
        p = ax[1].imshow(At_train_out[i], animated=True)


    anim = animation.FuncAnimation(fig, update, frames=len(a_train) - 1, repeat=True)
    if save:
        anim.save("./src/Tests/Figures/RNN-GATBIL/adj_animation.gif", writer="pillow")
        plt.close(fig)

    fig1, ax1 = plt.subplots(1, 1)
    range_axis = (1.0, 30)


    def update2(i):
        ax1.clear()
        fig1.suptitle(f"Year {10 + i}")
        a_pred = a_train[i].numpy().flatten()
        a_true = At_train_out[i].numpy().flatten()
        ax1.hist(a_true, bins=60, alpha=0.5, range=range_axis, color="blue", density=True, label="True")
        ax1.set_ylim((0, .21))
        ax1.hist(a_pred, bins=60, alpha=0.5, range=range_axis, color="red", density=True, label="Pred")
        ax1.legend()


    anim2 = animation.FuncAnimation(fig1, update2, frames=len(a_train) - 1, repeat=True)
    if save:
        anim2.save("./src/Tests/Figures/RNN-GATBIL/hist_anim.gif", writer="pillow")

    plt.figure()
    exp = [166, 169, 40]
    imp = [81, 40, 166]
    samples = 5
    a_train_10 = zero_inflated_lognormal(logits).sample(samples)  # 10xTxNxN
    a_train_10 = tf.math.log(tf.clip_by_value(a_train_10, 1, 1e10)).numpy()
    for exp, imp in zip(exp, imp):
        edges_timeseries = a_train_10[:, :, exp, imp]
        edges_timeseries = edges_timeseries.reshape(samples, -1)
        plt.scatter([np.arange(edges_timeseries.shape[-1])] * samples, edges_timeseries, s=1)

        true_edges_timeseries = At_train_out[:, exp, imp].numpy().flatten()
        plt.plot(true_edges_timeseries, linewidth=2)
    plt.show()
