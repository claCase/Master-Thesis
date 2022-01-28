import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import tensorflow.keras.backend as K

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


def mask_sparse(A_sparse, p=0.5):
    mask = np.random.choice((True, False), p=(p, 1 - p), size=len(A_sparse.values))
    A0_ind = np.asarray(A_sparse.indices)
    A0_values = np.asarray(A_sparse.values)
    A0_ind = A0_ind[mask]
    A0_values = A0_values[mask]
    A0 = tf.sparse.SparseTensor(A0_ind, A0_values, np.max(A0_ind, 0))
    return A0, mask


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


def make_data(years, relations, nodes):
    tmp_adj_list = []
    for t in range(years):
        for r in range(relations):
            for importer in range(nodes):
                for exporter in range(nodes):
                    if importer != exporter:
                        link = np.random.choice((0, 1))
                        if link:
                            value = np.random.lognormal(0, 1)
                            # value = np.cast["float32"](value)
                            tmp_adj_list.append((t, r, importer, exporter, value))
    tmp_adj_list = np.asarray(tmp_adj_list, dtype=np.float32)
    sparse = tf.sparse.SparseTensor(
        tmp_adj_list[:, :-1], tmp_adj_list[:, -1], (years, relations, nodes, nodes)
    )
    return sparse


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
        sigma = tf.math.maximum(
            K.softplus(sigma),
            tf.math.sqrt(K.epsilon()))
    perm_axis = tf.concat((tf.range(len(p.shape))+1, (0,)), axis=0)
    p_mix = tf.transpose([1 - p, p], perm=perm_axis)
    a = tfd.Mixture(
        cat=tfd.Categorical(probs=p_mix),
        components=[
            tfd.Deterministic(loc=tf.zeros_like(p)),
            tfd.LogNormal(loc=mu, scale=sigma),
        ])
    return a
