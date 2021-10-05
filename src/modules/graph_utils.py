import tensorflow as tf
import numpy as np


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
    i = np.unique(edges[:, 0]).tolist()
    j = np.unique(edges[:, 1]).tolist()
    indices = set()
    for idx in i:
        indices.add(idx)
    for idx in j:
        indices.add(idx)
    indices = list(indices)
    mesh = np.meshgrid(indices, indices)
    p1 = mesh[0].reshape(-1)
    p2 = mesh[1].reshape(-1)
    tot_edges = np.concatenate([p1[:, np.newaxis], p2[:, np.newaxis]], 1)
    zero_edges = []
    for edge in tot_edges:
        if tuple(edge) not in edge_set:
            if np.random.uniform(0, 1) < p:
                zero_edges.append(edge)
    sparse_zero = tf.sparse.SparseTensor(zero_edges, np.zeros(len(zero_edges), dtype=np.float32), A.shape)
    return tf.sparse.reorder(tf.sparse.add(A, sparse_zero))


def mask_sparse(A_sparse, p=0.5):
    mask = np.random.choice((True, False), p=(p, 1 - p), size=len(A_sparse.values))
    A0_ind = np.asarray(A_sparse.indices)
    A0_values = np.asarray(A_sparse.values)
    A0_ind = A0_ind[mask]
    A0_values = A0_values[mask]
    A0 = tf.sparse.SparseTensor(A0_ind, A0_values, np.max(A0_ind, 0))
    return A0, mask

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
    sparse = tf.sparse.SparseTensor(tmp_adj_list[:, :-1], tmp_adj_list[:, -1], (years, relations, nodes, nodes))
    return sparse

