import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as l

# from math import pi


@tf.function
def square_loss(
    y_true, y_pred, test_mask=None, no_diag=False, random_mask=False, zero_mask=False
):
    if isinstance(y_true, tf.sparse.SparseTensor) and isinstance(
        y_pred, tf.sparse.SparseTensor
    ):
        diff = tf.square(y_true.values - y_pred.values)
        return tf.reduce_mean(diff)

    if isinstance(y_true, tf.Tensor) and isinstance(y_pred, tf.Tensor):
        """if zero_mask:
        no_zero_edges = tf.where(y_true == 0., 0., 1.)
        no_zero_edges = tf.cast(no_zero_edges, tf.float32)
        #diff *= no_zero_edges
        y_pred *= no_zero_edges"""

        if no_diag:
            no_diag = tf.ones_like(y_pred) - tf.linalg.diag(tf.ones(y_pred.shape[0]))
            y_pred = y_pred * no_diag
        diff = tf.square(y_true - y_pred)

        if test_mask is not None:
            diff *= test_mask

        if random_mask:
            random_mask = tf.constant(
                np.random.choice((0, 1), size=(y_true.shape)), dtype=tf.float32
            )
            diff *= random_mask
        return tf.reduce_mean(diff, (0, 1))

    if isinstance(y_true, tf.sparse.SparseTensor) and isinstance(y_pred, tf.Tensor):
        y_true = tf.sparse.to_dense(y_true)
        diff = tf.square(y_true - y_pred)
        # print(f"DIFF SHAPE {diff.shape}")
        diff = tf.reshape(diff, [-1])
        return tf.reduce_mean(diff)

    raise Exception(
        f"Inputs and Predictions are of different types: {type(y_true)}, {type(y_pred)}"
    )


@tf.function
def nll(y_true, mu, sigma):
    """
    y_true: SparseTensor
    mu: SparseTensor
    sigma: SparseTensor
    """
    assert isinstance(y_true, tf.sparse.SparseTensor) and isinstance(
        mu, tf.sparse.SparseTensor
    )
    nll = (
        -tf.reduce_mean(
            tf.math.square(y_true.values - mu.values) / tf.math.square(sigma.values)
            + tf.math.log(tf.math.square(sigma.values))
        )
        * 0.5
    )
    # print(f"loss shape: {nll.shape}")
    return nll


@tf.function
def binary_cross_entropy(y_true, p):
    assert isinstance(y_true, tf.sparse.SparseTensor) and isinstance(
        p, tf.sparse.SparseTensor
    )
    loss = tf.reduce_mean(
        y_true.values * p.values + (1 - y_true.values) * (1 - p.values)
    )
    return loss


# @tf.function
def mixed_discrete_continuous_nll_sparse(y_true, mu, p):
    assert (
        isinstance(y_true, tf.sparse.SparseTensor)
        and isinstance(mu, tf.sparse.SparseTensor)
        and isinstance(p, tf.sparse.SparseTensor)
    )

    non_zero_edges = tf.where(y_true.values != 0)
    y_true_non_zero_sparse = tf.sparse.SparseTensor(
        tf.squeeze(tf.gather(y_true.indices, non_zero_edges)),
        tf.squeeze(tf.gather(y_true.values, non_zero_edges)),
        y_true.shape,
    )
    mu_non_zero_sparse = tf.sparse.SparseTensor(
        tf.squeeze(tf.gather(y_true.indices, non_zero_edges)),
        tf.squeeze(tf.gather(mu.values, non_zero_edges)),
        y_true.shape,
    )
    """sigma_non_zero_sparse = tf.sparse.SparseTensor(tf.squeeze(tf.gather(y_true.indices, non_zero_edges)),
                                                tf.squeeze(tf.gather(sigma.values, non_zero_edges)),
                                                y_true.shape)"""
    y_true_binary_values = tf.where(y_true.values == 0, 0.0, 1.0)
    y_true_binary_sparse = tf.sparse.SparseTensor(
        y_true.indices, y_true_binary_values, y_true.shape
    )
    loss_nll = square_loss(y_true_non_zero_sparse, mu_non_zero_sparse)
    loss_ce = binary_cross_entropy(y_true_binary_sparse, p)
    return loss_nll + loss_ce


@tf.function
def embedding_smoothness(X, A, square=True):
    if isinstance(A, tf.Tensor):
        A = tf.sparse.from_dense(A)
    i, j = A.indices[:, 0], A.indices[:, 1]
    x1 = tf.gather(X, i)
    x2 = tf.gather(X, j)
    if square:
        d = x1 - x2
        loss = tf.einsum("ij,ij->i", d, d)
        return tf.reduce_mean(loss, 0)
    else:
        d = x1 - x2
        loss = tf.reduce_sum(d, 1)
        return tf.abs(tf.reduce_sum(loss, 0))


@tf.function
def temporal_embedding_smoothness(Xt0, Xt1, l=1):
    inner = tf.einsum("ij,ij->i", Xt0, Xt1)
    smoothness = 1 - inner
    tot_smoothness = tf.reduce_sum(smoothness)
    return tot_smoothness * l


class SparsityRegularizerLayer(l.Layer):
    def __init__(self, rate, **kwargs):
        super(SparsityRegularizerLayer, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, **kwargs):
        if isinstance(inputs, tf.sparse.SparseTensor):
            loss = self.rate * tf.reduce_sum(tf.square(inputs.values))
            self.add_loss(loss)
        if isinstance(inputs, tf.Tensor):
            loss = self.rate * tf.math.sqrt(tf.reduce_sum(tf.square(inputs)), (0, 1))
            self.add_loss(loss)
        return loss


class EmbeddingSmoothnessRegularizer(l.Layer):
    def __init__(self, rate, **kwargs):
        super(EmbeddingSmoothnessRegularizer, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, **kwargs):
        loss = self.rate * embedding_smoothness(*inputs)
        self.add_loss(loss)
        return loss


class SparsePenalizedMSE(l.Layer):
    def __init__(self):
        super(SparsePenalizedMSE, self).__init__()

    def build(self, input_shape):
        y, y_pred = input_shape
        self.penalty = self.add_variable(
            name="loss_penalty", shape=y_pred, initializer="glorot_normal"
        )

    def call(self, inputs, *args, **kwargs):
        y_pred, y = inputs
        penalty_mask = tf.sparse.SparseTensor(
            y_pred.indices, tf.ones_like(y_pred.values), y_pred.shape
        )
        penalty_mask = tf.sparse.to_dense(penalty_mask)
        penalty = self.penalty * penalty_mask
        penalty = tf.sparse.from_dense(penalty)
        diff = tf.math.square((y.values - y_pred.values) * penalty.values)
        loss = tf.reduce_mean(diff)
        return loss


class DensePenalizedMSE(l.Layer):
    def __init__(self):
        super(DensePenalizedMSE, self).__init__()

    def build(self, input_shape):
        y_true, y_pred = input_shape
        self.B = self.add_variable(
            name="loss_B", shape=(y_true), initializer="glorot_normal"
        )

    def call(self, inputs, *args, **kwargs):
        y_true, y_pred = inputs
        diff = y_true - y_pred
        select_non_zero = tf.where(y_true > 0, tf.abs(self.B) + 1, 1.0)
        diff_penalized = tf.multiply(diff, select_non_zero)
        diff_penalized = tf.reduce_mean(tf.square(diff_penalized), (0, 1))
        return diff_penalized


@tf.function
def mse_uncertainty(y_true, mu, sigma):
    return tf.reduce_mean(
        tf.square(y_true - mu) * tf.exp(-sigma) * 0.5 + 0.5 * sigma, (0, 1)
    )
