import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import tensorflow.keras.backend as K
from scipy.linalg import block_diag
import os
import pickle as pkl
from sklearn.metrics import explained_variance_score


@tf.function
def symmetrize(A, out_degree=False):
    ones = tf.ones_like(A)
    if out_degree:
        A = tf.transpose(A)
    mask_a = tf.linalg.band_part(ones, -1, 0)
    diag_mask = tf.linalg.band_part(ones, 0, 0)
    mask_a = mask_a - diag_mask
    A_masked = tf.math.multiply(A, mask_a)
    A_symm = A_masked + tf.transpose(A_masked)
    return A_symm


def sample_zero_edges(A, p):
    edges = A.indices
    edgelist = edges.numpy().tolist()
    edge_set = set()
    for e in edgelist:
        edge_set.add(tuple(e))
    if len(A.shape) == 2:
        # i = set(np.unique(edges[:, 0]).tolist())
        # j = set(np.unique(edges[:, 1]).tolist())
        # indices = i.union(j)
        # indices = list(indices)
        indices = np.arange(A.shape[-1])
        mesh = np.meshgrid(indices, indices)
        p1 = mesh[0].reshape(-1)
        p2 = mesh[1].reshape(-1)
        tot_edges = np.concatenate([p1[:, np.newaxis], p2[:, np.newaxis]], 1)
    elif len(A.shape) == 3:
        i = set(np.unique(edges[:, 1]).tolist())
        j = set(np.unique(edges[:, 2]).tolist())
        relations = np.unique(edges[:, 0]).tolist()
        indices = i.union(j)
        indices = list(indices)
        mesh = np.meshgrid(relations, indices, indices)
        p1 = mesh[0].reshape(-1)
        p2 = mesh[1].reshape(-1)
        p3 = mesh[2].reshape(-1)
        tot_edges = np.concatenate(
            [p1[:, np.newaxis], p2[:, np.newaxis], p3[:, np.newaxis]], 1
        )
    tot_edges = np.unique(tot_edges, axis=0)
    tot_edges = [tuple(a) for a in tot_edges]
    tot_edges = set(tot_edges)
    non_zero_edges = np.unique(A.indices.numpy(), axis=0)
    non_zero_edges = [tuple(a) for a in non_zero_edges]
    non_zero_edges = set(non_zero_edges)
    allowed_edges = tot_edges - non_zero_edges
    allowed_edges = np.asarray(list(allowed_edges))
    allowed_edges_idx = np.arange(len(allowed_edges))
    sampled_edges = np.random.choice(
        allowed_edges_idx, replace=False, size=int(len(allowed_edges) * p)
    )
    sampled_edges = allowed_edges[sampled_edges]
    sparse_zero = tf.sparse.SparseTensor(
        sampled_edges, np.zeros(len(sampled_edges), dtype=np.float32), A.shape
    )
    return tf.sparse.reorder(tf.sparse.add(A, sparse_zero))


def mask_sparse(A_sparse, p=0.1):
    mask = np.random.choice((True, False), p=(p, 1 - p), size=len(A_sparse.values))
    A0_ind = np.asarray(A_sparse.indices)
    A0_values = np.asarray(A_sparse.values)
    A_train_ind = A0_ind[mask]
    A_train_values = A0_values[mask]
    A_test_ind = A0_ind[False & mask]
    A_test_values = A0_values[False & mask]
    A_train = tf.sparse.SparseTensor(A_train_ind, A_train_values, A_sparse.shape)
    A_test = tf.sparse.SparseTensor(A_test_ind, A_test_values, A_sparse.shape)
    return A_train, A_test, mask


def add_self_loops_data(data_sp: tf.sparse.SparseTensor):
    n_nodes = data_sp.shape[-1]
    years = data_sp.shape[0]
    r = data_sp.shape[1]
    self_edges = []
    self_values = []
    for t in range(years):
        data_t_dense = tf.sparse.to_dense(
            tf.sparse.slice(data_sp, (t, 0, 0, 0), (1, r, n_nodes, n_nodes))
        )[0]
        deg_out = tf.reduce_sum(data_t_dense, axis=2)
        deg_sparse = tf.sparse.from_dense(deg_out)
        self_edges_t = np.asarray(deg_sparse.indices)
        idx = self_edges_t[:, -1]
        self_edges_t = np.concatenate(
            [
                np.ones(len(idx), dtype=np.int64)[:, np.newaxis] * t,
                self_edges_t,
                idx[:, np.newaxis],
            ],
            axis=1,
        )
        self_values_t = deg_sparse.values
        self_edges.extend(self_edges_t.tolist())
        self_values.extend(self_values_t)
    self_data_sp = tf.sparse.SparseTensor(self_edges, self_values, data_sp.shape)
    data_sp = tf.sparse.add(data_sp, self_data_sp)
    data_sp = tf.sparse.reorder(data_sp)
    return data_sp


def add_self_loop(data: tf.sparse.SparseTensor):
    i = np.arange(data.shape[-1])
    edges = np.concatenate([i[:, None], i[:, None]], -1)
    values = tf.math.unsorted_segment_sum(
        data.values, data.indices[:, 0], data.shape[-1]
    )
    self_sparse = tf.sparse.SparseTensor(edges, values, data.shape)
    return tf.sparse.add(data, self_sparse)


def predict_all_sparse(A_true):
    if len(A_true.shape) == 2:
        # A_true = tf.sparse.to_dense(tf.sparse.reorder(A_true))
        i = np.arange(A_true.shape[-1])
        i, j = np.meshgrid(i, i)
        i = i.reshape(-1)[:, None]
        j = j.reshape(-1)[:, None]
        tot_edges = np.concatenate([i, j], -1)
        return tf.sparse.reorder(
            tf.sparse.SparseTensor(
                tot_edges, np.random.normal(size=len(tot_edges)), A_true.shape
            )
        )
    elif len(A_true.shape) == 3:
        # A_true = tf.sparse.to_dense(tf.sparse.reorder(A_true))
        i = np.arange(A_true.shape[-1])
        k = np.arange(A_true.shape[0])
        k, i, j = np.meshgrid(k, i, i)
        i = i.reshape(-1)[:, None]
        j = j.reshape(-1)[:, None]
        k = k.reshape(-1)[:, None]
        tot_edges = np.concatenate([k, i, j], -1)
        return tf.sparse.reorder(
            tf.sparse.SparseTensor(
                tot_edges, np.random.normal(size=len(tot_edges)), A_true.shape
            )
        )


def generate_list_lower_triang(batch, t, lag):
    lower_adj = np.tril(np.ones(shape=(t, t)))
    prev = np.zeros(shape=(lag, t))
    sub_lower = np.vstack([prev, lower_adj])[:-lag]
    lower_adj = lower_adj - sub_lower
    return np.asarray([lower_adj] * batch)


def simmetricity(A):
    # https://math.stackexchange.com/questions/2048817/metric-for-how-symmetric-a-matrix-is
    if isinstance(A, tf.Tensor):
        A = A.numpy()
    A_sym = 0.5 * (A + A.T)
    A_anti_sym = 0.5 * (A - A.T)
    A_sym_norm = np.sum(np.power(A_sym, 2), (0, 1))
    A_anti_sym_norm = np.sum(np.power(A_anti_sym, 2), (0, 1))
    score = (A_sym_norm - A_anti_sym_norm) / (A_sym_norm + A_anti_sym_norm)
    return score


def sum_gradients(grads_t, reduce="mean"):
    grad_vars = [[] for _ in range(len(grads_t[0]))]
    for t, grad_t in enumerate(grads_t):
        for i, grad_var in enumerate(grad_t):
            grad = tf.expand_dims(grad_var, 0)
            grad_vars[i].append(grad)

    for i, v in enumerate(grad_vars):
        if reduce == "mean":
            g = tf.reduce_mean(v, 0)
        if reduce == "sum":
            g = tf.reduce_sum(v, 0)
        g = tf.squeeze(g, 0)
        grad_vars[i] = g
    return grad_vars


def get_positional_encoding_matrix(max_len, d_emb):
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


def LogGamma(alpha, beta, l=0):
    return tfp.bijectors.Shift(l - 1)(tfp.bijectors.Log()(tfp.distributions.Gamma(alpha, beta)))


def zero_inflated_logGamma(logits=None, p=None, mu=None, sigma=None):
    """
    logits: TxNxNx3
    """
    if logits is not None:
        p, mu, sigma = tf.unstack(logits, axis=-1)
        p = tf.nn.sigmoid(p)
        sigma = tf.math.maximum(
            K.softplus(sigma),
            tf.math.sqrt(K.epsilon()))
    perm_axis = tf.concat((tf.range(len(p.shape)) + 1, (0,)), axis=0)
    p_mix = tf.transpose([1 - p, p], perm=perm_axis)
    a = tfd.Mixture(
        cat=tfd.Categorical(probs=p_mix),
        components=[
            tfd.Deterministic(loc=tf.zeros_like(p)),
            LogGamma(mu, sigma),
        ])
    return a


def zero_inflated_lognormal(logits=None, p=None, mu=None, sigma=None):
    """
    logits: TxNxNx3
    """
    if logits is not None:
        p, mu, sigma = tf.unstack(logits, axis=-1)
        p = tf.nn.sigmoid(p)
        sigma = compute_sigma(sigma)
    perm_axis = tf.concat((tf.range(len(p.shape)) + 1, (0,)), axis=0)
    p_mix = tf.transpose([1 - p, p], perm=perm_axis)
    a = tfd.Mixture(
        cat=tfd.Categorical(probs=p_mix),
        components=[
            tfd.Deterministic(loc=tf.zeros_like(p)),
            tfd.LogNormal(loc=mu, scale=sigma),
        ])
    return a


def compute_sigma(sigma):
    return tf.math.maximum(K.softplus(sigma), tf.math.sqrt(K.epsilon()))


def pair_wise_distance(X, distance="square", symmetric=True):
    x1 = tf.tile(X, [X.shape[0], 1])
    x2 = tf.repeat(X, X.shape[0], axis=0)
    if distance == "square":
        sd = tf.math.square(x1 - x2)
    elif distance == "abs":
        sd = tf.math.abs(x1 - x2)
    else:
        raise ValueError(f"Distance: {distance} not found")
    sd = tf.reduce_sum(sd, -1)
    if symmetric:
        sd = sd * 0.5 + tf.transpose(sd) * 0.5
    # print(sd)
    reshape_size = (*X.shape[:-1], X.shape[-2])
    sdm = tf.reshape(sd, reshape_size)
    return sdm


def entropy(p):
    H = tf.reduce_sum(p * tf.math.log(p), axis=None)
    return H


def cross_entropy(p, q):
    CH = tf.reduce_sum(p * tf.math.log(q), axis=None)
    return CH


def discrete_kl(p, q):
    H = entropy(p)
    CH = cross_entropy(p, q)
    return H + CH


# TODO
def graph_similarity(G1, G2, corr=True):
    G1 = np.asarray(G1 > 0, np.float32)
    G2 = np.asarray(G2 > 0, np.float32)
    d1_in = np.sum(G1, -1)
    d1_out = np.sum(G1, -2)
    d2_in = np.sum(G2, -1)
    d2_out = np.sum(G2, -2)
    sim_in = (G1).dot(G2.T) - (d1_in.T.dot(d2_in)) / G1.shape[-1]
    sim_out = (G1.T).dot(G2) - (d1_out.T.dot(d2_out)) / G1.shape[-1]
    if corr:
        sd1 = np.sqrt(np.sum(G1 - 1, ))
    return sim_in, sim_out


def generate_data_temporal_block_matrix(data, sparse=True):
    t, r, n = data.shape[0], data.shape[1], data.shape[2]
    times = []
    for i in range(t):
        times.append(block_diag(*data[i]))
    block = block_diag(*times)
    diag_inputs = np.ones(t * r * n) * 15
    diag = np.empty(block.shape)
    index = 0
    tot_values = diag.shape[0]
    for i in range(t):
        for j in range(r):
            index += 1
            diag_upper = np.diagflat(diag_inputs, index * n)[:tot_values, :tot_values]
            diag_lower = np.diagflat(diag_inputs, -index * n)[:tot_values, :tot_values]
            diag += diag_upper + diag_lower
    if sparse:
        return tf.sparse.from_dense(block), tf.sparse.from_dense(diag)
    return block, diag


def generate_data_relational_block_matrix(data, sparse=True):
    t, r, n = data.shape[0], data.shape[1], data.shape[2]
    times = []
    for i in range(t):
        block = block_diag(*data[i])
        times.append(block)
    times = np.asarray(times)
    diag_inputs = np.ones(r * n) * 15
    diag = np.zeros(block.shape)
    tot_values = diag.shape[0]
    index = 0
    for j in range(r):
        index += 1
        diag_upper = np.diagflat(diag_inputs, index * n)[:tot_values, :tot_values]
        diag_lower = np.diagflat(diag_inputs, -index * n)[:tot_values, :tot_values]
        diag += diag_upper + diag_lower
    diags = []
    for _ in range(t):
        diags.append(diag)
    diags = np.asarray(diags)
    if sparse:
        return tf.sparse.from_dense(times), tf.sparse.from_dense(diags)
    return times, diags


def optimal_point(points, type="roc"):
    def distance(p):
        if type == "roc":
            opt = [0, 1]
        else:
            opt = [1, 1]
        return np.sqrt(np.sum(np.square(p - opt)))

    d = []
    for p in points:
        d.append(distance(p))
    return np.argsort(d)[0]


def sample_from_logits(logits, n_samples, sparse_input, numpy=True):
    a_t = zero_inflated_lognormal(logits).sample(n_samples)
    a_mean = tf.reduce_mean(a_t, 0)
    a_sd = tf.math.reduce_variance(a_t, 0)
    if sparse_input:
        a_t = tf.squeeze(a_t)
    else:
        a_t = tf.squeeze(a_t[0])
    if numpy:
        a_t, a_mean, a_sd = a_t.numpy(), a_mean.numpy(), a_sd.numpy()
    return a_t, a_mean, a_sd


def mean_accuracy_score(A_true, A_pred):
    abs_squared_error = np.sqrt(np.mean(np.abs(A_true - A_pred), axis=(0, 1)))
    abs_squared = np.sqrt(np.mean(np.abs(A_true), axis=(0, 1)))
    return 1 - abs_squared_error / abs_squared


def load_dataset(baci=True, r=6, r2=10, n=175, model_pred=True, features=False, log=True):
    if r>r2:
        raise ValueError(f"r: {r} cannot be greater than r2: {r2}")
    if baci:
        with open(os.path.join(os.getcwd(), "Data", "baci_sparse_price.pkl"), "rb") as file:
            t = 24
            data = pkl.load(file)
            data = tf.sparse.reorder(data)
            data_slice = tf.sparse.slice(data, (0, 0, 0, r), (t, n, n, r2))
            data_slice = tf.sparse.transpose(data_slice, perm=(0, 3, 1, 2))
            data_dense = tf.sparse.reduce_sum(data_slice, 1).numpy()
        if features:
            with open(os.path.join(os.getcwd(), "Data", "X_input_baci.pkl"), "rb") as file:
                x_f = pkl.load(file)[:, :n, :n]
                x_f = np.expand_dims(x_f, 1)
                x_f = x_f[:-2]
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
        t = 57
        data_slice = tf.sparse.slice(data_sp, (1, r, 0, 0), (t, r2, n, n))
        data_dense = tf.sparse.reduce_sum(data_slice, 1).numpy()
        if features:
            with open(os.path.join(os.getcwd(), "Data", "X_input_not_baci.pkl"), "rb") as file:
                x_f = pkl.load(file)[:-1, :n, :n]
            x_f = np.expand_dims(x_f, 1)
    xt = np.expand_dims([np.eye(n)] * t, 1)
    if features:
        if log:
            x_f = np.log(x_f)
            x_f = np.where(np.isinf(x_f) | np.isnan(x_f), 0, x_f)
        xt = np.concatenate([xt, x_f], -1)
    if model_pred:
        data_dense = tf.cast(tf.expand_dims(data_dense, 1), tf.float32).numpy()
        return data_dense, xt, n, t
    return data_dense, n, t


'''def compute_statistics(a_true, a_pred):

    for t in range(1, a_true.shape[0]):
        a_score = tf.clip_by_value(tf.math.log(a_train[t + 1][0]), 0, 1e10).numpy()
        sample_score = tf.clip_by_value(tf.math.log(sample), 0, 1e10).numpy()
        ma = np.clip(mean_accuracy_score(a_score, sample_score), -10, 1)
        mean_acc.append(ma)
        msq = np.clip(mean_squared_error(a_score.reshape(-1), sample_score.reshape(-1)), 0, 100)
        mean_rmse.append(msq)
        mabs = np.clip(mean_absolute_error(a_score.reshape(-1), sample_score.reshape(-1)), 0,100)
        mean_abs.append(mabs)
        evar = explained_variance_score(a_score.reshape(-1), sample_score.reshape(-1))
        exp_var.append(evar)
    t_samples = np.asarray(t_samples)'''
