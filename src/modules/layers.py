import numpy as np
import scipy
import networkx as ntx
import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.keras import activations
from tensorflow.keras import initializers
import tensorflow.keras.backend as k
from tensorflow.keras.losses import mean_squared_error, sparse_categorical_crossentropy
from scipy.sparse.csgraph import laplacian
import scipy.sparse.linalg
from networkx.linalg import directed_laplacian_matrix
from src.modules.utils import add_self_loop
from src.modules.utils import generate_list_lower_triang


# physical_devices = tf.config.list_physical_devices('GPU')

# tf.config.experimental.set_memory_growth(physical_devices[0], True)


class BilinearLayer(l.Layer):
    """ "
    Implementation of Bilinear Layer:
    formula for single vector -> bilinear_r(e1, e2) = e^t*R_r*e where e:R^f and R_r: R^(fxf)
    formula for matrix -> ExRxE^T where E:R^(Nxf) and R: R^(fxf)
    """

    def __init__(self, sum_relations=False, *args, **kwargs):
        super(BilinearLayer, self).__init__(*args, kwargs)
        self.Bs = []
        self.sum_relations = sum_relations

    def build(self, input_shape):
        print(f"BUILD LAYER {input_shape}")
        self.batch_size = input_shape[0]
        self.relations = input_shape[1]
        self.n_nodes = input_shape[2]
        self.features = input_shape[3]
        with tf.name_scope("Bilinear"):
            # for each relation add trainable weight matrix of size fxf
            for i in range(self.relations):
                self.Bs.append(
                    self.add_weight(
                        shape=(self.features, self.features),
                        trainable=True,
                        name=f"Relation{i}",
                    )
                )

    def call(self, inputs, *args, **kwargs):
        result = []
        # for each relation calculate bilinear product of each node
        for r in range(self.relations):
            e = k.dot(inputs[:, r], self.Bs[r])
            bilinear_prod = k.batch_dot(e, tf.transpose(inputs[:, r], perm=(0, 2, 1)))
            bilinear_prod = tf.nn.sigmoid(bilinear_prod)
            result.append(bilinear_prod)
        # bilinear_prod = tf.stack(bilinear_prod)
        # print(bilinear_prod.shape)
        result = tf.transpose(result, perm=(1, 0, 2, 3))

        if self.sum_relations:
            print(f"sum_relations")
            result = k.sum(k.concatenate(result, axis=0), axis=1)

        return result

    def linear(self, input, output, **kwargs):
        W = tf.Variable(shape=(input.shape[-1], output), **kwargs)
        return tf.matmul(input, W)


class Bilinear(l.Layer):
    def __init__(
            self, hidden_dim=5, activation=None, dropout_rate=0.5, qr=True, **kwargs
    ):
        super(Bilinear, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.qr = qr
        self.initializer = tf.keras.initializers.GlorotNormal()

    def build(self, input_shape):
        self.R = tf.Variable(
            initial_value=self.initializer(shape=(self.hidden_dim, self.hidden_dim))
        )
        self.X = tf.Variable(
            initial_value=self.initializer(shape=(input_shape[0], self.hidden_dim))
        )
        self.dense = tf.keras.layers.Dense(self.hidden_dim)
        if self.dropout_rate:
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, **kwargs):
        if self.qr:
            Q, W = tf.linalg.qr(self.X, full_matrices=False)
            Z = tf.matmul(tf.matmul(W, self.R), W, transpose_b=True)
            if self.dropout_rate:
                Z = self.dropout(Z)
            A = tf.matmul(tf.matmul(Q, Z), Q, transpose_b=True)
        else:
            if self.dropout_rate:
                X = self.dropout(self.X)

            x_left = tf.matmul(X, self.R)
            A = tf.matmul(x_left, X, transpose_b=True)

        A = activations.get(self.activation)(A)
        # self.add_loss(tf.matmul(self.Q, self.Q, transpose_b=True) - tf.eye(self.Q.shape[0]))
        if self.qr:
            return tf.matmul(Q, W), A
        else:
            return self.X, A


class BilinearSparse(l.Layer):
    def __init__(self, hidden_dim, activation="relu", **kwargs):
        super(BilinearSparse, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.initializer = initializers.GlorotNormal()
        self.activation = activation

    def build(self, input_shape):
        self.X = tf.Variable(
            initial_value=self.initializer(shape=(input_shape[0], self.hidden_dim))
        )
        self.R = tf.Variable(
            initial_value=self.initializer(shape=(self.hidden_dim, self.hidden_dim))
        )
        # self.X1 = tf.Variable(initial_value=self.initializer(shape=(self.hidden_dim, self.hidden_dim)))
        # self.X2 = tf.Variable(initial_value=self.initializer(shape=(self.hidden_dim, self.hidden_dim)))

    def call(self, inputs, **kwargs):
        edgelist, values = inputs.indices, inputs.values
        # x1 = tf.matmul(self.X, self.X1)
        # x2 = tf.matmul(self.X, self.X2)
        e1 = tf.gather(self.X, edgelist[:, 0])
        e2 = tf.gather(self.X, edgelist[:, 1])
        left = tf.einsum("ij,jk->ik", e1, self.R)
        right = tf.einsum("ij,ij->i", left, e2)
        if self.activation:
            right = activations.get(self.activation)(right)
        A = tf.sparse.SparseTensor(edgelist, right, inputs.shape)
        return self.X, A


class BilinearDecoderDense(l.Layer):
    def __init__(self, activation="relu", diagonal=False, qr=False, **kwargs):
        super(BilinearDecoderDense, self).__init__(**kwargs)
        self.initializer = initializers.GlorotNormal()
        self.activation = activation
        self.diagonal = diagonal
        self.qr = qr

    def build(self, input_shape):
        X_shape = input_shape
        if self.diagonal:
            self.R = tf.Variable(
                initial_value=self.initializer(shape=(X_shape[-1],)), name="R_bilinear"
            )
            self.R = tf.linalg.diag(self.R)
        else:
            self.R = tf.Variable(
                initial_value=self.initializer(shape=(X_shape[-1], X_shape[-1])),
                name="R_bilinear",
            )

    def call(self, inputs, **kwargs):
        X = inputs
        if not self.qr:
            x_left = tf.matmul(X, self.R)
            A = tf.matmul(x_left, X, transpose_b=True)
            A = activations.get(self.activation)(A)
            return X, A
        else:
            Q, W = tf.linalg.qr(X, full_matrices=False)
            Z = tf.matmul(tf.matmul(W, self.R), W, transpose_b=True)
            A = tf.matmul(tf.matmul(Q, Z), Q, transpose_b=True)
            A = activations.get(self.activation)(A)
            return tf.matmul(Q, W), A


class BatchBilinearDecoderDense(l.Layer):
    """
    inputs:
        - X of shape batch x N x d
        - A of shape batch x N x N
    outputs: A of shape batch x N x N
    """

    def __init__(self, activation="relu", qr=False, regularizer="l2", zero_diag=True):
        super(BatchBilinearDecoderDense, self).__init__()
        self.activation = activation
        self.regularizer = regularizer
        self.qr = qr
        self.zero_diag = zero_diag

    def build(self, input_shape):
        x = input_shape
        self.R = self.add_weight(
            shape=(x[-1], x[-1]),
            initializer="glorot_normal",
            regularizer=self.regularizer,
            name="bilinear_matrix",
        )
        self.diag =tf.constant(1 - tf.linalg.diag([tf.ones(x[-2])]))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        if self.qr:
            Q, W = tf.linalg.qr(x, full_matrices=False)
            W_t = tf.einsum("...jk->...kj", W)
            Q_t = tf.einsum("...jk->...kj", Q)
            Z = tf.matmul(tf.matmul(W, self.R), W_t)
            A = tf.matmul(tf.matmul(Q, Z), Q_t)
            A = activations.get(self.activation)(A)
        else:
            x_t = tf.einsum("...jk->...kj", x)
            mat_left = tf.matmul(x, self.R)
            A = activations.get(self.activation)(tf.matmul(mat_left, x_t))
        if self.zero_diag:
            return A * self.diag
        return A


class BilinearDecoderSparse(l.Layer):
    def __init__(self, activation="relu", diagonal=False, qr=False, **kwargs):
        super(BilinearDecoderSparse, self).__init__(**kwargs)
        self.initializer = initializers.GlorotNormal()
        self.diagonal = diagonal
        self.activation = activation
        self.qr = qr

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        if self.diagonal:
            self.R_kernel = tf.Variable(
                initial_value=self.initializer(shape=(X_shape[-1]))
            )
            self.R_kernel = tf.linalg.diag(self.R_kernel)
        else:
            self.R_kernel = tf.Variable(
                initial_value=self.initializer(shape=(X_shape[-1], X_shape[-1]))
            )

    def call(self, inputs, **kwargs):
        X, A = inputs
        """if self.qr:
            Q, W = tf.linalg.qr(X, full_matrices=False)
            Z = tf.matmul(tf.matmul(W, self.R), W, transpose_b=True)
            A = tf.matmul(tf.matmul(Q, Z), Q, transpose_b=True)"""
        i, j = A.indices[:, 0], A.indices[:, 1]
        e1 = tf.gather(X, i)
        e2 = tf.gather(X, j)
        left = tf.einsum("ij,jk->ik", e1, self.R_kernel)
        right = tf.einsum("ij,ij->i", left, e2)
        if self.activation:
            A_pred = activations.get(self.activation)(right)
        A_pred = tf.sparse.SparseTensor(A.indices, A_pred, A.shape)
        return X, A_pred


class SelfAttention(l.Layer):
    def __init__(
            self,
            channels=10,
            attn_heads=5,
            dropout_rate=0.5,
            concat_heads=False,
            return_attn=False,
            renormalize=False,
            initializer="GlorotNormal"
    ):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.attn_heads = attn_heads
        self.dropout_rate = dropout_rate
        self.concat_heads = concat_heads
        self.return_attn = return_attn
        self.renormalize = renormalize
        self.initializer = initializers.get(initializer)

    def build(self, input_shape):
        """
        Inputs: X, A
            - X: shape(NxTxd)
            #- A: shape(TxNxN)
            - A: shape(NxTxT)
        """
        x, a = input_shape
        self.q_w = self.add_weight(name="query", shape=(self.attn_heads, x[-1], self.channels),
                                   initializer=self.initializer)
        self.k_w = self.add_weight(name="key", shape=(self.attn_heads, x[-1], self.channels),
                                   initializer=self.initializer)
        self.v_w = self.add_weight(name="value", shape=(self.attn_heads, x[-1], self.channels),
                                   initializer=self.initializer)
        if self.dropout_rate:
            self.drop = l.Dropout(self.dropout_rate)

    def call(self, inputs, *args, **kwargs):
        """
        query=key=value:
            - n: nodes if time series or batch size
            - t: time dim if time series or number of nodes if n=batch size
            - d: input embedding dimension
            - o: output embedding dimension
            - h: number of heads
        x=input embedding of shape NxTxd
        a=input adjacency matrix of shape NxTxT
            -
        """
        x, a = inputs
        query = tf.einsum("ntd,hdo->ntho", x, self.q_w)
        key = tf.einsum("ntd,hdo->ntho", x, self.k_w)
        value = tf.einsum("ntd,hdo->ntho", x, self.v_w)
        qk = tf.einsum("ntho,nzho->nhtz", query, key)  # NxHxTxT
        qk /= tf.sqrt(tf.cast(self.channels, tf.float32))
        qk += tf.transpose([tf.where(a == 0.0, -1e10, 0.0)] * self.attn_heads, perm=(1, 0, 2, 3))  # NxHxTxT
        soft_qk = tf.nn.softmax(qk, axis=-1)
        if self.dropout_rate:
            soft_qk = self.drop(soft_qk)
            if self.renormalize:
                soft_qk = tf.nn.softmax(soft_qk, axis=-1)
        x_prime = tf.einsum("nhtz,nzho->nhto", soft_qk, value)
        if self.concat_heads:
            x_prime = tf.transpose(x_prime, (0, 2, 1, 3))  # NxTxHxO
            x_prime = tf.reshape(x_prime, (*tf.shape(x_prime)[:-2], -1))  # NxTxHO
        else:
            x_prime = tf.reduce_mean(x_prime, axis=1)
            x_prime = tf.squeeze(x_prime)  # NxTxO
        if self.return_attn:
            return x_prime, soft_qk
        return x_prime
