import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as l
from tensorflow.keras.losses import binary_crossentropy
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import tensorflow.keras.backend as K
from src.modules.utils import LogGamma

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


def zero_inflated_lognormal_loss(labels: tf.Tensor,
                                 logits: tf.Tensor,
                                 reduce_axis=(-4, -3, -2, -1),
                                 weights=[0.5,0.5],
                                 smooth=0.0) -> tf.Tensor:
    """Computes the zero inflated lognormal loss.
    Usage with tf.keras API:
    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=zero_inflated_lognormal)
    ```
    Arguments:
    labels: True targets, tensor of shape [batch_size, 1].
    logits: Logits of output layer, tensor of shape [batch_size, 3].
    Returns:
    Zero inflated lognormal loss value.
    """
    # logits = tf.expand_dims(logits, 0)
    # labels = tf.expand_dims(labels, 0)
    weights = tf.constant(weights) / tf.reduce_sum(weights)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    positive = tf.cast(labels > 0, tf.float32)
    logits = tf.convert_to_tensor(logits, dtype=tf.float32)
    logits.shape.assert_is_compatible_with(
        tf.TensorShape(labels.shape[:-1].as_list() + [3]))

    positive_logits = logits[..., :1]
    positive_logits = tf.nn.sigmoid(positive_logits)
    classification_loss = tf.keras.losses.binary_crossentropy(
        y_true=positive, y_pred=positive_logits, from_logits=False, axis=reduce_axis, label_smoothing=smooth)
    '''
    ce_weights = tf.where(positive==0, weights[0], weights[1])
    classification_loss = tf.expand_dims(classification_loss, axis=-1)
    classification_loss = classification_loss * ce_weights
    classification_loss = tf.reduce_mean(classification_loss, axis=reduce_axis)'''

    loc = logits[..., 1:2]
    # loc = tf.math.maximum(tf.nn.relu(loc), tf.math.sqrt(K.epsilon()))
    scale = tf.math.maximum(K.softplus(logits[..., 2:]),tf.math.sqrt(K.epsilon()))
    safe_labels = positive * labels + (
            1 - positive) * K.ones_like(labels)
    regression_loss = -K.mean(
        positive * (tfd.LogNormal(loc=loc, scale=scale)).log_prob(safe_labels),
        axis=reduce_axis)
    return 0.5 * classification_loss + 0.5 * regression_loss




@tf.function
def embedding_smoothness_sparse(X, A, square=True):
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


class EmbeddingSmoothnessRegularizerSparse(l.Layer):
    def __init__(self, rate, **kwargs):
        super(EmbeddingSmoothnessRegularizerSparse, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, **kwargs):
        loss = self.rate * embedding_smoothness_sparse(*inputs)
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


class TemporalSmoothness(l.Layer):
    """
    Parameters:
        - l: lambda regularizer coefficient that weights previous with forward embedding
        - distance: distance metric to use, can be: "euclidian", "rotation", "angle"
    """

    def __init__(self, l=0.1, distance="euclidian"):
        super(TemporalSmoothness, self).__init__()
        allowed_distances = {"euclidian", "rotation", "angle"}
        self.l = l
        self.distance = distance
        if self.distance not in allowed_distances:
            raise ValueError(f"{self.distance} not in {allowed_distances}")

    def build(self, input_shape):
        """
        Representation Learning for Dynamic Graphs - A Survey

        Inputs:
            x: Temporal embeddings of shape TxNxd where T is the time dimension, N is the number of nodes and d is the
               latent embedding dimension
            a: Temporal adjacency matrix of shape TxNxN
        Parameters:
            R: if distance=="rotation" then R is the Rotation Matrix Learnable Parameter of shape Txdxd
        """
        x = input_shape
        if self.distance == "rotation":
            self.R = self.add_weight(name="Rotation_weight", shape=(x[0] - 1, x[-1], x[-1]))
            self.identity = tf.constant(tf.eye(x[-1], x[-1]))
        #self.l = self.add_weight(name="Regularizer_weight", shape=(1,))

    def call(self, inputs, *args, **kwargs):
        """
        Inputs:
            x: Temporal embeddings of shape TxNxd where T is the time dimension, N is the number of nodes and d is the
               latent embedding dimension
            a: Temporal adjacency matrix of shape TxNxN
        """
        x = inputs
        xt = x[:-1]  # Xt
        xt_1 = x[1:]  # Xt+1
        if self.distance == "rotation":
            xt_1 = tf.einsum("tnd,tdx->tnx", xt_1, self.R)
        temporal_diff = xt_1 - xt
        distance = tf.einsum("tnd,tnd->tn", temporal_diff, temporal_diff)  # Euclidian distance ||xt+1 - xt||
        if self.distance == "euclidian":
            return self.l * tf.reduce_sum(distance, (0, 1))  # sum over T and N
        elif self.distance == "rotation":
            """
            Node Embedding over Temporal Graphs
            """
            rotation_inner = tf.matmul(self.R, tf.transpose(self.R, perm=(0, 2, 1)))  # TxRxR
            sq_diff = tf.math.square(rotation_inner - self.identity)  # TxRxR
            rotation_constraint = tf.reduce_sum(sq_diff, (0, 1, 2))
            return self.l * tf.reduce_sum(distance, (0, 1)) + rotation_constraint  # sum over T and N
        elif self.distance == "angle":
            """
            Scalable Temporal Latent Space Inference for Link Prediction in Dynamic Social Networks
            """
            angle = 1.0 - tf.einsum("tnd,tnd->tn", xt_1, xt)
            return self.l * tf.reduce_sum(angle, (0, 1))
