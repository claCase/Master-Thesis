import tensorflow as tf
import tensorflow.keras.layers as l


@tf.function
def square_loss(y_true, y_pred):
    if isinstance(y_true, tf.sparse.SparseTensor) and isinstance(
        y_pred, tf.sparse.SparseTensor
    ):
        diff = tf.square(y_true.values - y_pred.values)
        return tf.sqrt(tf.reduce_sum(diff))

    if isinstance(y_true, tf.Tensor) and isinstance(y_pred, tf.Tensor):
        diff = tf.square(y_true - y_pred)
        diff = tf.reshape(diff, [-1])
        return tf.sqrt(tf.reduce_sum(diff))

    if isinstance(y_true, tf.sparse.SparseTensor) and isinstance(y_pred, tf.Tensor):
        y_true = tf.sparse.to_dense(y_true)
        diff = tf.square(y_true - y_pred)
        print(f"DIFF SHAPE {diff.shape}")
        diff = tf.reshape(diff, [-1])
        return tf.reduce_sum(diff)

    raise Exception(
        f"Inputs and Predictions are of different types: {type(y_true)}, {type(y_pred)}"
    )


@tf.function
def log_normal_likelihood_loss(y_true, mu, sigma):
    likelihood = -(
        tf.reduce_sum(
            tf.math.square(tf.math.log(y_true.values) - mu)
            / (2 * tf.math.square(sigma))
        )
        + tf.reduce_sum(tf.math.log(sigma))
    )
    nll = -likelihood
    print(f"loss shape: {nll.shape}")
    return nll


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
        return tf.sqrt(tf.reduce_sum(loss, 0))
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
            loss = self.rate * tf.math.sqrt(tf.reduce_sum(tf.square(inputs)), (0,1))
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

