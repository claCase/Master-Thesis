import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.losses import MeanSquaredError as mse
from src.modules.layers import (
    AttentionBasedRelationPrediction,
    LinearScoringSparse,
    BilinearRelationalScoringSparse,
    LinearScoringDense,
)
from src.modules.losses import (
    square_loss,
    nll,
    SparsePenalizedMSE,
    mixed_discrete_continuous_nll_sparse,
)
from sklearn.manifold import TSNE
from src.graph_data_loader import plot_names, plot_products
import numpy as np
import scipy.stats as stats
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import tqdm
import pickle as pkl
import argparse


def generate_all_indices(r, i, j):
    ijr = np.stack(np.meshgrid(r, i, j), axis=-1).reshape(-1, 3)
    return ijr


class AttentionModel(k.models.Model):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.kgat = AttentionBasedRelationPrediction(10, 10, 10)
        self.decoder = LinearScoringSparse(activation="relu")

    def call(self, inputs, *args, **kwargs):
        x, r, a = inputs
        x_h, r_h = self.kgat([x, r, a])
        scores = self.decoder([x_h, r_h, a])
        scores = tf.reshape(scores, [-1])
        pred = tf.sparse.SparseTensor(a.indices, scores, a.shape)
        return x_h, r_h, pred


class RBilinear(k.models.Model):
    def __init__(self):
        super(RBilinear, self).__init__()
        self.score_mu = BilinearRelationalScoringSparse("relu", 0.5, True)
        # self.score_sigma = BilinearRelationalScoringSparse("relu", 0.5, True)
        self.score_p = BilinearRelationalScoringSparse("sigmoid", 0.5, True)
        self.x_kern = k.layers.Dense(20, "tanh")
        # self.x_kern2 = k.layers.Dense(10, "tanh")

        self.r_kern = k.layers.Dense(20, "tanh")
        # self.r_kern2 = k.layers.Dense(10, "tanh")

    def call(self, inputs, training=None, mask=None):
        x, r, a = inputs
        x_emb = self.x_kern(x)
        # x_emb = self.x_kern2(x_emb)
        r_emb = self.r_kern(r)
        # r_emb = self.r_kern2(r_emb)
        mu = self.score_mu([x_emb, r_emb, a])
        # sigma = self.score_sigma([x_emb, r_emb, a])
        p = self.score_p([x_emb, r_emb, a])
        mu = tf.sparse.SparseTensor(a.indices, mu, a.shape)
        # sigma = tf.sparse.SparseTensor(a.indices, tf.exp(sigma), a.shape)
        p = tf.sparse.SparseTensor(a.indices, p, a.shape)
        return x_emb, r_emb, mu, p


class RLinearSparse(k.models.Model):
    def __init__(self):
        super(RLinearSparse, self).__init__()
        self.scoref = LinearScoringSparse("relu")
        self.x_kern = k.layers.Dense(20, None)
        self.r_kern = k.layers.Dense(20, None)

    def call(self, inputs, training=None, mask=None):
        x, r, a = inputs
        x_emb = self.x_kern(x)
        r_emb = self.r_kern(r)
        score = self.scoref([x_emb, r_emb, a])
        score = tf.reshape(score, [-1])
        adj = tf.sparse.SparseTensor(a.indices, score, a.shape)
        return x_emb, r_emb, adj


class RLinearDense(k.models.Model):
    def __init__(self, emb_dim=15, emb_act="tanh", out_act="relu"):
        super(RLinearDense, self).__init__()
        self.emb_act = emb_act
        self.emb_dim = emb_dim
        self.score_f = LinearScoringDense()
        self.xs_f = k.layers.Dense(self.emb_dim, self.emb_act)
        # self.xt_f = k.layers.Dense(self.emb_dim, self.emb_act)
        self.r_f = k.layers.Dense(self.emb_dim, self.emb_act)
        self.out_act = out_act

    def call(self, inputs, training=None, mask=None):
        x, r, a = inputs
        xs = self.xs_f(x)
        # xt = self.xt_f(x)
        xr = self.r_f(r)
        score = self.score_f([xs, xr, a])
        score = k.activations.get(self.out_act)(score)
        return xs, xr, score


class LinearScoringSparseModel(k.models.Model):
    def __init__(self):
        super(LinearScoringSparseModel, self).__init__()
        self.score_mu_sigma_p = LinearScoringSparse(activation="relu", units=2)
        self.encode_x = k.layers.Dense(15, "relu")
        self.encode_r = k.layers.Dense(15, "relu")

    def call(self, inputs, training=None, mask=None):
        x, r, a = inputs
        x = self.encode_x(x)
        r = self.encode_r(r)
        mu_sigma_p = self.score_mu_sigma([x, r, a])
        mu, sigma, p = tf.split(mu_sigma_p, 3, -1)
        p = k.activations.get("sigmoid")(p)
        mu = tf.sparse.SparseTensor(a.indices, mu, a.shape)
        sigma = tf.sparse.SparseTensor(a.indices, tf.exp(sigma), a.shape)
        mu = tf.sparse.SparseTensor(a.indices, p, a.shape)
        return x, r, mu, sigma, p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--r", type=int, default=20)
    parser.add_argument("--n", type=int, default=174)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--perc_edges", default=0.6)
    parser.add_argument("--test_perc", default=0.3)
    args = parser.parse_args()
    synthetic = args.synthetic
    r = args.r
    n = args.n
    sparse = args.sparse
    print(f"SPARSE {sparse}")
    perc_edges = args.perc_edges
    test_perc = args.test_perc
    if synthetic:
        a = tf.sparse.from_dense(
            np.random.normal(size=(r, n, n)) * np.random.choice(size=(r, n, n))
        )
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
        data_slice = tf.sparse.slice(data_sp, (50, 0, 0, 0), (1, r, n, n))
        a = tf.sparse.reduce_sum(data_slice, 0, output_is_sparse=False)
        # a = tf.sparse.to_dense(a)
        # a += 1
        a = tf.math.log(a)
        a = tf.clip_by_value(a, 0, 1e100)
        a_sparse = tf.sparse.from_dense(a)
        idx_tot = a_sparse.indices.numpy()
        values_tot = a_sparse.values.numpy()
        len_idx_tot = len(idx_tot)
        idx_range = np.arange(len_idx_tot)
        np.random.shuffle(idx_range)
        a_index_test_size = int(len_idx_tot * test_perc)
        idx_train = idx_range[a_index_test_size:]
        idx_test = idx_range[:a_index_test_size]
        a_index_train = idx_tot[idx_train]
        a_index_test = idx_tot[idx_test]
        values_test = values_tot[idx_train]
        values_train = values_tot[idx_test]
        a_train = tf.sparse.SparseTensor(a_index_train, values_test, a_sparse.shape)
        a_test = tf.sparse.SparseTensor(a_index_test, values_train, a_sparse.shape)

        zero_edges = (tf.where(a <= 0)).numpy()
        np.random.shuffle(zero_edges)
        zero_edges = zero_edges[: int((r * n ** 2 - len(zero_edges)) * perc_edges)]
        zero_edges_size = len(zero_edges)
        idx_range = np.arange(zero_edges_size)
        np.random.shuffle(idx_range)
        idx_test_size = int(zero_edges_size * test_perc)
        idx_test = idx_range[:idx_test_size]
        idx_train = idx_range[idx_test_size:]
        zero_edges_train = zero_edges[idx_train]
        zero_edges_test = zero_edges[idx_test]
        a_zero_edges_train = tf.sparse.SparseTensor(
            zero_edges_train, np.zeros(len(idx_train), dtype=np.float32), a_sparse.shape
        )
        a_zero_edges_test = tf.sparse.SparseTensor(
            zero_edges_test, np.zeros(len(idx_test), dtype=np.float32), a_sparse.shape
        )

        A_train = tf.sparse.add(a_train, a_zero_edges_train)
        A_train = tf.sparse.reorder(A_train)
        A_train_bin = tf.sparse.SparseTensor(
            A_train.indices, tf.where(A_train.values > 0.0, 1.0, 0.0), A_train.shape
        )
        A_test = tf.sparse.add(a_test, a_zero_edges_test)
        A_test = tf.sparse.reorder(A_test)
        A_test_bin = tf.sparse.SparseTensor(
            A_test.indices, tf.where(A_test.values > 0.0, 1.0, 0.0), A_test.shape
        )
        """
        A = tf.sparse.from_dense(a)
        A = tf.sparse.add(A, A_zero_edges)
        A = tf.sparse.reorder(A)
        """
        # values = tf.math.log(a.values)
        # values = tf.clip_by_value(values, 0, 1e15)
        # A = tf.sparse.SparseTensor(a.indices, values, a.shape)

    X = np.eye(n)
    R = np.eye(r)

    # model = AttentionModel()
    model = RBilinear()
    # model = RLinearSparse()
    if not sparse:
        model = RLinearDense(15)
    if not sparse:
        A_train = tf.sparse.to_dense(A_train)
        A_test = tf.sparse.to_dense(A_test)

    x_h0, r_h0, a0, p0 = model([X, R, A_train])
    optimizer = tf.keras.optimizers.Adam(0.01)
    loss_hist = []
    epochs = 260
    bar = tqdm.tqdm(total=epochs, leave=True)
    for i in range(epochs):
        with tf.GradientTape() as tape:
            x_h, r_h, mu, p = model([X, R, A_train])
            x_h, r_h, mu_test, p_test = model([X, R, A_test])
            loss_train = mixed_discrete_continuous_nll_sparse(A_train, mu, p)
            loss_test = mixed_discrete_continuous_nll_sparse(A_test, mu_test, p_test)
            grads = tape.gradient(loss_train, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_hist.append((loss_train, loss_test))
        bar.update(1)
    if sparse:
        a_dense = tf.sparse.to_dense(mu).numpy()
        A_dense = tf.sparse.to_dense(A_train).numpy()
    else:
        a_dense = a
        A_dense = A_train
    loss_plot = plt.plot(loss_hist)
    plt.legend(iter(loss_plot), ["Train", "Test"])
    plt.show()

    fig, axs = plt.subplots(2, 2)
    x1 = axs[0, 0].imshow(a_dense[0])
    axs[0, 0].set_title("Pred")
    x2 = axs[0, 1].imshow(A_dense[0])
    axs[0, 1].set_title("True")
    fig.colorbar(x1, ax=axs[0, 0])
    fig.colorbar(x2, ax=axs[0, 1])

    x1 = axs[1, 0].imshow(a_dense[1])
    axs[1, 0].set_title("Pred")
    x2 = axs[1, 1].imshow(A_dense[1])
    axs[1, 1].set_title("True")
    fig.colorbar(x1, ax=axs[1, 0])
    fig.colorbar(x2, ax=axs[1, 1])

    x_init = TSNE(n_components=2, perplexity=50).fit_transform(x_h0)
    x_emb = TSNE(n_components=2, perplexity=50).fit_transform(x_h)
    r_init = TSNE(n_components=2, perplexity=50).fit_transform(r_h0)
    r_emb = TSNE(n_components=2, perplexity=50).fit_transform(r_h)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(x_init[:, 0], x_init[:, 1])
    plot_names(x_init, axs[0, 0])
    axs[0, 0].set_title("Initial Country Embeddings")
    axs[0, 1].scatter(x_emb[:, 0], x_emb[:, 1])
    axs[0, 1].set_title("Final Country Embeddings")
    plot_names(x_emb, axs[0, 1])
    # axs[1, 0].scatter(r_init[:, 0], r_init[:, 1])
    axs[1, 0].set_title("Initial Products Embeddings")
    plot_products(r_emb, axs[1, 0])
    # axs[1, 1].scatter(r_emb[:, 0], r_emb[:, 1])
    plot_products(r_emb, axs[1, 1])
    axs[1, 1].set_title("Final Produtcs Embeddings")
    if not sparse:
        a_v = tf.sparse.from_dense(a)
        A_v = tf.sparse.from_dense(A_train)
    else:
        a_v = mu
        A_v = A_train
    a_v = a_v.values.numpy()
    a_v = a_v[a_v > 1]
    A_v = A_v.values.numpy()
    A_v = A_v[A_v > 1]
    fit_params_g = stats.gamma.fit(A_v)
    fit_params_n = stats.norm.fit(A_v)
    x = np.linspace(0, 30, 300)
    pdf_g = stats.gamma.pdf(x, *fit_params_g)
    pdf_n = stats.norm.pdf(x, *fit_params_n)
    pdf_l = stats.logistic.pdf(x, *fit_params_n)
    ll_g = np.sum(stats.gamma.logpdf(A_v, *fit_params_g))
    ll_n = np.sum(stats.norm.logpdf(A_v, *fit_params_n))
    skew = stats.skew(A_v)
    skew_t, skew_p = stats.skewtest(A_v)
    kurt = stats.kurtosis(A_v)
    kurt_t, kurt_p = stats.kurtosistest(A_v)
    plt.figure()
    plt.hist(a_v, bins=100, color="red", alpha=0.5, density=True, label="Pred")
    plt.hist(A_v, bins=100, color="blue", alpha=0.5, density=True, label="True")
    plt.plot(x, pdf_g, color="blue", lw=2, label=f"Gamma fit {round(ll_g, 2)}")
    plt.plot(
        x,
        pdf_n,
        color="cyan",
        lw=2,
        label=f"Normal fit {round(ll_n, 2)} \n"
        f"skewness {round(skew, 2)} \n "
        f"skew_p {round(skew_p * 100, 2)}% \n"
        f"kurtosis {round(kurt, 2)} \n"
        f"kurtosis_p {round(kurt_p * 100, 2)}%",
    )
    print(f"kurt_p {kurt_p}\n skewness_p {skew_p}")
    plt.legend()

    plt.figure()
    plt.subplot(131)
    stats.probplot(A_v, dist="norm", plot=plt)
    plt.title("QQ-plot True")
    plt.subplot(132)
    stats.probplot(a_v, dist="norm", plot=plt)
    plt.title("QQ-plot Pred")
    plt.subplot(133)
    if sparse:
        diff = (A_train.values - mu.values).numpy()
    else:
        diff = (A_train - a).numpy().flatten()
    stats.probplot(diff, dist="norm", plot=plt)
    plt.title("QQ-plot Errors")
    plt.show()
