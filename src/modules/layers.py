import numpy as np
import scipy
import networkx as ntx
import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.keras import activations
from tensorflow.keras import initializers
import tensorflow.keras.backend as k
from src.modules.utils import make_data
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
    """def __init__(self, hidden_dim=5, activation=None, **kwargs):
        super(Bilinear, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.initializer = initializers.GlorotNormal()

    def build(self, input_shape):
        self.R = tf.Variable(
            initial_value=self.initializer(shape=(self.hidden_dim, self.hidden_dim))
        )
        self.X = tf.Variable(
            initial_value=self.initializer(shape=(input_shape[0], self.hidden_dim))
        )

    def call(self, inputs, **kwargs):
        x_left = tf.matmul(self.X, self.R)
        A = tf.matmul(x_left, x_left, transpose_b=True)
        if self.activation is not None:
            A = activations.get(self.activation)(A)
        return self.X, A"""

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
        self.act = tf.keras.layers.LeakyReLU(alpha=0.3)
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
        X_shape, A_shape = input_shape
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
        X, A = inputs
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

    def __init__(self, activation="relu", qr=True, regularizer="l2"):
        super(BatchBilinearDecoderDense, self).__init__()
        self.activation = activation
        self.regularizer = regularizer
        self.qr = qr

    def build(self, input_shape):
        x, a = input_shape
        self.R = self.add_weight(
            shape=(x[-1], x[-1]),
            initializer="glorot_normal",
            regularizer=self.regularizer,
            name="bilinear_matrix",
        )

    def call(self, inputs, *args, **kwargs):
        x, a = inputs
        if self.qr:
            Q, W = tf.linalg.qr(x, full_matrices=False)
            W_t = tf.einsum("...jk->...kj", W)
            Q_t = tf.einsum("...jk->...kj", Q)
            Z = tf.matmul(tf.matmul(W, self.R), W_t)
            A = tf.matmul(tf.matmul(Q, Z), Q_t)
            A = activations.get(self.activation)(A)
            return tf.matmul(Q, W), A
        else:
            x_t = tf.einsum("...jk->...kj", x)
            mat_left = tf.matmul(x, self.R)
            A = activations.get(self.activation)(tf.matmul(mat_left, x_t))
            return x, A


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
            lags=10,
            concat_heads=False,
            return_attn=False
    ):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.attn_heads = attn_heads
        self.dropout_rate = dropout_rate
        self.lags = lags
        self.concat_heads = concat_heads
        self.return_attn = return_attn

    def build(self, input_shape):
        """
        Inputs: X, A
            - X: shape(NxTxd)
            #- A: shape(TxNxN)
            - A: shape(NxTxT)
        """
        x, a = input_shape
        self.q_w = self.add_weight(name="query", shape=(self.attn_heads, x[-1], self.channels))
        self.k_w = self.add_weight(name="key", shape=(self.attn_heads, x[-1], self.channels))
        self.v_w = self.add_weight(name="value", shape=(self.attn_heads, x[-1], self.channels))
        '''self.temp_masking = tf.transpose(tf.where(tf.constant(
            [generate_list_lower_triang(a[0], a[-1], self.lags)] * self.attn_heads
        ) == 0.0, -1e10, 0), perm=(1, 0, 2, 3))'''
        if self.dropout_rate:
            self.drop = l.Dropout(self.dropout_rate)

    def call(self, inputs, *args, **kwargs):
        """
        query=key=value:
            - n: nodes if time series or batch size
            - t: time dim if time series or number of nodes
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
        qk = tf.einsum("ntho,nzho->nhtz", query, key)
        qk /= tf.sqrt(tf.cast(self.channels, tf.float32))
        qk += tf.transpose([tf.where(a == 0, -1e-10, 0.0)] * self.attn_heads, perm=(1, 0, 2, 3))  # NxHxTxT
        soft_qk = tf.nn.softmax(qk, axis=-1)
        if self.dropout_rate:
            soft_qk = self.drop(soft_qk)
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


class FlatDecoderSparse(l.Layer):
    def __init__(self, activation="relu", **kwargs):
        super(FlatDecoderSparse, self).__init__(**kwargs)
        self.initializer = initializers.GlorotNormal()
        self.activation = activation

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        self.kernel1 = tf.Variable(
            initial_value=self.initializer(shape=(X_shape[-1], 1))
        )
        self.kernel2 = tf.Variable(
            initial_value=self.initializer(shape=(X_shape[-1], 1))
        )

    def call(self, inputs, *args, **kwargs):
        X, A = inputs
        i, j = A.indices[:, 0], A.indices[:, 1]
        X1 = tf.matmul(X, self.kernel1)
        X2 = tf.matmul(X, self.kernel2)
        e1 = tf.gather(X1, i)
        e2 = tf.gather(X2, j)
        score = e1 + e2
        if self.activation is not None:
            score = activations.get(self.activation)(score)
        return X, tf.sparse.reorder(tf.sparse.SparseTensor(A.indices, score, A.shape))


class InnerProductDenseDecoder(l.Layer):
    def __init__(self, activation=None, **kwargs):
        super(InnerProductDenseDecoder, self).__init__(**kwargs)
        self.activation = activation

    def call(self, inputs, **kwargs):
        X, A = inputs
        inner = tf.matmul(X, X, transpose_b=True)
        if self.activation is not None:
            inner = activations.get(self.activation)(inner)
        return X, inner


class InnerProductSparseDecoder(l.Layer):
    def __init__(self, activation=None, dropout_rate=0.5, **kwargs):
        super(InnerProductSparseDecoder, self).__init__(**kwargs)
        self.activation = activation
        self.dropout_rate = dropout_rate

    def call(self, inputs, **kwargs):
        X, A = inputs
        i, j = A.indices[:, 0], A.indices[:, 1]
        e1 = tf.gather(X, i)
        e2 = tf.gather(X, j)
        inner = tf.einsum("ij,ij->i", e1, e2)
        if self.dropout_rate:
            inner = l.Dropout(self.dropout_rate)(inner)
        if self.activation is not None:
            inner = activations.get(self.activation)(inner)

        A_pred = tf.sparse.SparseTensor(A.indices, inner, A.shape)
        return X, A_pred


def three_vectors_outer(nodes_source, nodes_target, relations):
    """
    Computes tensor product between three vectors: XxYxZ
        Parameters:
            nodes_source: Nxfn where N is number of nodes and fn is the feature size
            nodes_target: Nxfn where N is number of nodes and fn is the feature size
            relations: Rxfr where R is the numer of relations and fr is the feature size
        Returns:
            tf.Tensor: Tensor of size N x N x R x fn x fn x fr
    """
    ein1 = tf.einsum("ij,kl->ikjl", nodes_source, nodes_target)
    ein2 = tf.einsum("ijkl,bc->ijbklc", ein1, relations)
    return ein2


class TrippletProductDense(l.Layer):
    def __init__(self, **kwargs):
        super(TrippletProductDense, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        R, S, T = inputs
        return three_vectors_outer(S, T, R)


def sparse_three_vector_outer(X, R, A: tf.sparse.SparseTensor):
    """
    Computes sparse version of three_vector_outer
        Parameters:
            X: list of source nodes of size Nxfn where N is number of nodes and fn is the feature size
            R: Rxfr where R is the numer of relations and fr is the feature size
            A: SparseTensor of dense_shape RxNxN
        Returns:
            E: list of edges where each edge is the result of the tensor product of the three vectors of size Exfrxfnxfn
    """
    S = tf.gather(X, A.indices[:, 1])
    T = tf.gather(X, A.indices[:, 2])
    R = tf.gather(R, A.indices[:, 0])

    ein1 = tf.einsum("ij,il->ijl", R, S)  # [start_idx:end_idx], S[start_idx:end_idx])
    ein2 = tf.einsum("ijl,in->ijln", ein1, T)  # T[start_idx:end_idx])
    return ein2


class TrippletProductSparse(l.Layer):
    def __init__(self, **kwargs):
        super(TrippletProductSparse, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        X, R, A = inputs
        return sparse_three_vector_outer(X, R, A)


def outer_kernel_product(outer, kernel):
    """
    Elementwise product between three dimensional tensor and kernel
        Parameters:
            outer: Tensor of size R x N x N x fr x fn x fn
            kernel: Tensor of size fr x fn x fn
        Returns:
             Tensor of size N x N x R
    """
    return tf.einsum("...ijk,ijk->...", outer, kernel)


class TrippletScoreOuter(l.Layer):
    def __init__(self, identity_kernel=False, **kwargs):
        super(TrippletScoreOuter, self).__init__(**kwargs)
        self.initializer = initializers.GlorotNormal()
        self.identity_kernel = identity_kernel

    def build(self, input_shape):
        kernel_shape = input_shape[-3:]
        if self.identity_kernel:
            self.kernel = tf.ones(kernel_shape)
        else:
            self.kernel = tf.Variable(
                initial_value=self.initializer(shape=kernel_shape)
            )

    def call(self, inputs, **kwargs):
        return outer_kernel_product(inputs, self.kernel)


class TrippletDecoderOuterSparse(l.Layer):
    def __init__(self, identity_kernel=False, **kwargs):
        super(TrippletDecoderOuterSparse, self).__init__(**kwargs)
        self.identity_kernel = identity_kernel
        self.dec1 = TrippletProductSparse()
        self.dec2 = TrippletScoreOuter(self.identity_kernel)

    def call(self, inputs, **kwargs):
        sparse_outer = self.dec1(inputs)
        return self.dec2(sparse_outer)


class TensorDecompositionLayer(l.Layer):
    def __init__(self, k, **kwargs):
        super(TensorDecompositionLayer, self).__init__(**kwargs)
        self.k = k
        self.initializer = initializers.GlorotNormal()

    def build(self, input_shape):
        A_shape = input_shape
        self.kernels = []
        for i in range(self.k):
            x = tf.Variable(
                initial_value=self.initializer(shape=(A_shape[1],)), trainable=True
            )
            r = tf.Variable(
                initial_value=self.initializer(shape=(A_shape[0],)), trainable=True
            )
            self.kernels.append({"x": x, "r": r})

    def call(self, inputs, **kwargs):
        modes = []
        for i in range(self.k):
            res = tf.einsum("i,j->ij", self.kernels[i]["r"], self.kernels[i]["x"])
            res2 = tf.einsum("ij,k->ijk", res, self.kernels[i]["x"])  # RxNxN
            res2 = tf.expand_dims(res2, axis=0)
            modes.append(res2)
        modes = tf.concat(modes, 0)
        A_pred = tf.reduce_sum(modes, 0)
        if isinstance(inputs, tf.sparse.SparseTensor):
            A_pred = tf.sparse.from_dense(A_pred)
        return A_pred


class NTN(l.Layer):
    """
    Implementation of Reasoning With Neural Tensor Networks for Knowledge Base Completion
    https://proceedings.neurips.cc/paper/2013/file/b337e84de8752b27eda3a12363109e80-Paper.pdf
    """

    def __init__(self, activation="relu", output_activation="relu", k=5, **kwargs):
        """
        Parameters:
            activation: type of activation to be used
            k: number of kernels for each type of relation
        """
        super(NTN, self).__init__()
        self.activation = activations.get(activation)
        self.output_activation = activations.get(output_activation)
        self.k = k

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        self.initializer = initializers.GlorotNormal()
        # Relation Specific Parameters
        r = A_shape[0]
        self.V1r = tf.Variable(
            initial_value=self.initializer(shape=(r, X_shape[-1], self.k))
        )
        self.V2r = tf.Variable(
            initial_value=self.initializer(shape=(r, X_shape[-1], self.k))
        )
        self.ur = tf.Variable(initial_value=self.initializer(shape=(r, self.k, 1)))
        self.br = tf.Variable(
            initial_value=self.initializer(
                shape=(
                    r,
                    self.k,
                )
            )
        )
        self.Wr = tf.Variable(
            initial_value=self.initializer(shape=(r, X_shape[-1], X_shape[-1], self.k))
        )

    def call(self, inputs, **kwargs):
        X, A = inputs
        if isinstance(A, tf.Tensor):
            A = tf.sparse.from_dense(A)
        return self.ntn(X, A)

    def ntn(self, X, A):
        a_pred = tf.sparse.from_dense(tf.zeros(shape=A.shape))
        for i in range(A.shape[0]):
            Ar = tf.sparse.reduce_sum(
                tf.sparse.slice(A, (i, 0, 0), (1, A.shape[1], A.shape[2])),
                0,
                output_is_sparse=True,
            )
            wi = self.Wr[i]
            ui = self.ur[i]
            bi = self.br[i]
            v1i = self.V1r[i]
            v2i = self.V2r[i]
            i, j, k = (
                Ar.indices[:, 0],
                Ar.indices[:, 1],
                tf.expand_dims(tf.ones(len(Ar.indices), dtype=tf.int64) * int(i), -1),
            )
            e1 = tf.gather(X, i)
            e2 = tf.gather(X, j)
            bilinear = tf.einsum("ij,jnk->ink", e1, wi)
            bilinear = tf.einsum("ink,in->ik", bilinear, e2)
            v1 = tf.einsum("ij,jk->ik", e1, v1i)
            v2 = tf.einsum("ij,jk->ik", e2, v2i)
            vr = v1 + v2
            activated = self.activation(bilinear + vr + bi)
            score = tf.einsum("ik,kj->ij", activated, ui)
            score = tf.squeeze(score)
            score = self.output_activation(score)
            i, j = tf.expand_dims(i, -1), tf.expand_dims(j, -1)
            indices = tf.concat([k, i, j], -1)
            a = tf.sparse.SparseTensor(indices, score, A.shape)
            a_pred = tf.sparse.add(a_pred, a)
        a_pred = tf.sparse.to_dense(a_pred)
        return X, a_pred


class RGHAT(l.Layer):
    """
    Implementation of Relational Graph Neural Network with Hierarchical Attention
    https://ojs.aaai.org/index.php/AAAI/article/view/6508
    """

    def __init__(
            self,
            **kwargs,
    ):
        super(RGHAT, self).__init__(**kwargs)
        pass


class CrossAggregation(l.Layer):
    """
    Implementation of Cross Aggregation from Deep & Cross Network for Ad Click Predictions: https://arxiv.org/pdf/1708.05123
    """

    def __init__(self, output_dim=None, dropout_rate=0.5, **kwargs):
        super(CrossAggregation, self).__init__(**kwargs)
        self.initializer = initializers.GlorotNormal()
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        X_s, X_t, A = input_shape
        if self.output_dim is None:
            self.output_dim = X_s[-1]
        self.kernel = tf.Variable(
            initial_value=self.initializer(shape=(X_s[-1], self.output_dim))
        )

    def call(self, inputs, **kwargs):
        X_s, X_t, A = inputs
        edgelist = A.indices
        h1 = tf.gather(X_s, edgelist[:, 0])
        h2 = tf.gather(X_t, edgelist[:, 1])
        outer = tf.einsum("ij,ik->ijk", h1, h2)
        attn_score = tf.matmul(outer, self.kernel)
        if self.dropout_rate:
            attn_score = l.Dropout(self.dropout_rate)(attn_score)
        attention = unsorted_segment_softmax(attn_score, edgelist[:, 1])
        attention = tf.sparse.SparseTensor(edgelist, attention, A.shape)
        """attended_embeddings = X_s * attention
        update_embeddings = tf.math.unsorted_segment_sum(
            attended_embeddings, edgelist[:, 1]
        )"""
        update_embeddings = tf.sparse.sparse_dense_matmul(attention, X)
        return update_embeddings


class GAIN(l.Layer):
    """
    Implementation of GAIN: Graph Attention & Interaction Network for Inductive Semi-Supervised Learning over
    Large-scale Graphs
    https://arxiv.org/pdf/2011.01393.pdf
    """

    pass


class GAT(l.Layer):
    def __init__(
            self,
            hidden_dim,
            dropout_rate=0.5,
            activation="relu",
            add_identity=False,
            add_bias=True,
            return_attention=False,
            **kwargs,
    ):
        """
        Implementation of Graph Atttention Networks: https://arxiv.org/pdf/2102.07200.pdf
        """
        super(GAT, self).__init__(**kwargs)
        self.initializer = initializers.GlorotNormal()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.return_attention = return_attention
        self.activation = activation
        self.add_bias = add_bias
        self.add_identity = add_identity

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        self.kernel1 = tf.Variable(
            initial_value=self.initializer(shape=(self.hidden_dim, 1))
        )
        self.kernel2 = tf.Variable(
            initial_value=self.initializer(shape=(self.hidden_dim, 1))
        )
        self.message_kernel_self = tf.Variable(
            initial_value=self.initializer(shape=(X_shape[-1], self.hidden_dim))
        )
        # self.message_kernel_ngb = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], self.hidden_dim)))
        if self.add_bias:
            self.bias = tf.Variable(
                initial_value=self.initializer(shape=(self.hidden_dim,))
            )

    def message_self(self, X):
        return tf.matmul(X, self.message_kernel_self)

    def message_ngb(self, X):
        return tf.matmul(X, self.message_kernel_ngb)

    def call(self, inputs, **kwargs):
        X, A = inputs
        if self.add_identity:
            A = add_self_loop(A)
        edgelist = A.indices
        x1 = self.message_self(X)
        e1 = tf.gather(x1, edgelist[:, 1])
        e1_att = tf.matmul(e1, self.kernel1)
        # x2 = self.message_ngb(X)
        e2 = tf.gather(x1, edgelist[:, 0])
        e2_att = tf.matmul(e2, self.kernel2)
        att_score = e1_att + e2_att
        att_score = activations.get(self.activation)(att_score)
        if self.dropout_rate:
            att_score = l.Dropout(self.dropout_rate)(att_score)
            x1 = l.Dropout(self.dropout_rate)(x1)
        att_score = tf.sparse.SparseTensor(edgelist, tf.squeeze(att_score), A.shape)
        attention = tf.sparse.softmax(att_score)
        update_embeddings = tf.sparse.sparse_dense_matmul(attention, x1)
        if self.add_bias:
            update_embeddings += self.bias
        """if self.dropout_rate:
            update_embeddings = l.Dropout(self.dropout_rate)(update_embeddings)"""
        if self.return_attention:
            return attention, update_embeddings, A
        else:
            return update_embeddings, A


class MultiHeadGAT(l.Layer):
    def __init__(self, heads, hidden_dim, return_attention=False, **kwargs):
        super(MultiHeadGAT, self).__init__(**kwargs)
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.initializer = initializers.GlorotNormal()
        self.return_attention = return_attention
        self.attention = []

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        self.attention_heads = []
        for _ in range(self.heads):
            self.attention_heads.append(
                GAT(self.hidden_dim, return_attention=self.return_attention)
            )
        self.heads_kernel = tf.Variable(
            initial_value=self.initializer(shape=(self.heads, 1))
        )

    def call(self, inputs, **kwargs):
        heads_emb = []
        for i in range(self.heads):
            if self.return_attention:
                att, emb, A = self.attention_heads[i](inputs)
                self.attention.append(att)
            else:
                emb, A = self.attention_heads[i](inputs)
            emb = tf.expand_dims(emb, 0)
            heads_emb.append(emb)
        heads_emb = tf.concat(heads_emb, 0)
        heads_kernel = l.Dropout(0.5)(self.heads_kernel)
        heads_comb = activations.get("softmax")(heads_kernel)
        combined_heads = tf.einsum("hnd,hk->knd", heads_emb, heads_comb)
        combined_heads = tf.squeeze(combined_heads, 0)
        if self.return_attention:
            return self.attention, combined_heads, A
        else:
            return combined_heads, A


class GCN(l.Layer):
    """
    Implementation of Graph Convolutional Neural Netwroks: https://arxiv.org/pdf/1609.02907.pdf
    """

    def __init__(self, hidden_dim, activation="relu", dropout_rate=0.5, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.initializer = initializers.GlorotNormal()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.activation = activation

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        self.W1 = tf.Variable(
            initial_value=self.initializer(shape=(X_shape[-1], self.hidden_dim))
        )
        # self.W2 = tf.Variable(initial_value=self.initializer(shape=(self.hidden_dim, self.output_dim)))

    def call(self, inputs, **kwargs):
        X, A = inputs
        A_aug = self.add_identity(A)
        D_inv_sqrt = self.inverse_degree(A_aug)
        A_norm = tf.matmul(D_inv_sqrt, A_aug)
        A_norm = tf.matmul(A_norm, D_inv_sqrt)
        Z = tf.matmul(A_norm, tf.matmul(X, self.W1))
        act = activations.get(self.activation)(Z)
        if self.dropout_rate:
            act = l.Dropout(self.dropout_rate)(Z)
        return act, A

    def add_identity(self, A):
        I = tf.eye(*A.shape)
        return A + I

    def inverse_degree(self, A):
        D = tf.math.reduce_sum(A, axis=-1)
        D_inv_sq = tf.math.sqrt(1 / D)
        D_inv_sq = tf.linalg.diag([D_inv_sq])[0]
        return D_inv_sq


class GCNDirected(l.Layer):
    """
    Implementation of Spectral-based Graph Convolutional Network for Directed Graphs
    https://arxiv.org/abs/1907.08990
    """

    def __init__(
            self, hidden_dim, activation="relu", dropout_rate=0.5, layer=0, **kwargs
    ):
        super(GCNDirected, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.initializer = initializers.GlorotNormal()
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.layer = layer

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        self.kernel1 = tf.Variable(
            initial_value=self.initializer(shape=(X_shape[-1], self.hidden_dim)),
            name=f"kernel l{self.layer}",
        )

    def call(self, inputs, **kwargs):
        X, A = inputs
        """#A = self.add_identity(A)
        D_out = tf.reduce_sum(A, axis=1)
        D_out_inv = 1 / D_out
        D_out_inv_mat = tf.linalg.diag([D_out_inv])[0]
        D_out_mat = tf.linalg.diag([D_out])[0]
        D_out_sqrt = tf.math.sqrt(D_out)
        D_out_sqrt_mat = tf.linalg.diag([D_out_sqrt])[0]
        D_out_inv_sqrt = 1 / D_out_sqrt
        D_out_inv_sqrt_mat = tf.linalg.diag([D_out_inv_sqrt])[0]
        print(f"D_out_inv_mat {D_out_inv_mat}")
        P = tf.matmul(D_out_inv_mat, A)
        print(f"P {P}")
        left_u, left_v = tf.linalg.eig(tf.transpose(P))  # get left eigenvectors by transposing matrix
        left_u_first = tf.argsort(tf.math.real(left_u), direction="DESCENDING")[0]
        print(f"Left U {left_u_first}")
        left_v = tf.math.real(tf.transpose(tf.gather(left_v, left_u_first, axis=1)))
        print(f"left v {left_v}")
        left_v_den = tf.reduce_sum(left_v)
        left_v_norm = left_v / left_v_den
        print(f"Left V {left_v_norm}")
        phi = tf.linalg.diag([left_v_norm])[0]
        phi_sqrt = tf.math.sqrt(left_v_norm)
        print(f"phi_sqrt {phi_sqrt}")
        phi_sqrt_inv = tf.math.sqrt(1 / phi_sqrt)
        phi_sqrt_mat = tf.linalg.diag([phi_sqrt])[0]
        print(f"phi_sqrt_inv {phi_sqrt_mat}")
        phi_sqrt_inv_mat = tf.linalg.diag([phi_sqrt_inv])[0]
        print(f"phi_sqrt_inv {phi_sqrt_inv_mat}")
        L = 0.5 * (tf.matmul(tf.matmul(phi_sqrt_mat, P), phi_sqrt_inv_mat) +
                   tf.matmul(tf.matmul(phi_sqrt_mat, tf.transpose(P)), phi_sqrt_inv_mat))
        L2 = 0.5 * (tf.matmul(tf.matmul(D_out_sqrt_mat, P), D_out_inv_sqrt_mat) +
                    tf.matmul(tf.matmul(D_out_sqrt_mat, tf.transpose(P)), D_out_inv_sqrt_mat))
        """
        # L = tf.constant(L)
        # L = laplacian(A, use_out_degree=True)
        G = ntx.from_numpy_matrix(A.numpy(), create_using=ntx.DiGraph)
        L = directed_laplacian_matrix(G, walk_type="pagerank")
        L = -L + tf.eye(A.shape[0])
        xt = tf.matmul(X, self.kernel1)
        # print(f"t {xt}")
        Z = tf.matmul(L, xt)
        # print(f"Z {Z}")
        act = activations.get(self.activation)(Z)
        if self.dropout_rate:
            act = l.Dropout(self.dropout_rate)(act)
        return act, A

    def add_identity(self, A):
        I = tf.eye(*A.shape)
        return A + I


class RelationalAwareAttention(l.Layer):
    def __init__(self, **kwargs):
        super(RelationalAwareAttention, self).__init__(**kwargs)


class GRAMME(l.Layer):
    def __init__(self, **kwargs):
        super(GRAMME, self).__init__(**kwargs)


class OrbitModel(l.Layer):
    """
    Implementation of From One Point to A Manifold: Orbit Models for Precise and Efficient Knowledge Graph Embedding
    https://arxiv.org/pdf/1512.04792v2.pdf
    """

    pass


def unsorted_segment_softmax(x, indices, n_nodes=None):
    """
    # From spektral/layers/ops/scatter.py
    Applies softmax along the segments of a Tensor. This operator is similar
    to the tf.math.segment_* operators, which apply a certain reduction to the
    segments. In this case, the output tensor is not reduced and maintains the
    same shape as the input.
    :param x: a Tensor. The softmax is applied along the first dimension.
    :param indices: a Tensor, indices to the segments.
    :param n_nodes: the number of unique segments in the indices. If `None`,
    n_nodes is calculated as the maximum entry in the indices plus 1.
    :return: a Tensor with the same shape as the input.
    """
    n_nodes = tf.reduce_max(indices) + 1 if n_nodes is None else n_nodes
    e_x = tf.exp(
        x - tf.gather(tf.math.unsorted_segment_max(x, indices, n_nodes), indices)
    )
    e_x /= tf.gather(
        tf.math.unsorted_segment_sum(e_x, indices, n_nodes) + 1e-9, indices
    )
    return e_x


def stable_softmax_3d(attention, A: tf.sparse.SparseTensor):
    """
    Compute softmax over 2d slice of 3d tensor:
        z(s,r,t) = sm(s,r,t) - max(sm(:,:,t))
        softmax(s,r,t) = exp(z(s,r,t))/sum_(s,r)[exp(z(s,r,t))]
        Parameters:
            x: list of values corresponding to the flatten 3d tensor
            A: sparse tensor of shape RxNxN where R is the number of relations and N is the number of nodes
    """
    sparse_attention = tf.sparse.SparseTensor(A.indices, tf.squeeze(attention), A.shape)
    dense_attention = tf.sparse.to_dense(sparse_attention)
    z = dense_attention - tf.math.reduce_max(dense_attention, axis=(-3, -2))
    expz = tf.exp(z)
    expz /= tf.reduce_sum(expz, axis=(-3, -2), keepdims=True)
    return expz


class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name="Time2VecLayer")
        self.k = kernel_size

    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(
            name="wb", shape=(input_shape[1],), initializer="uniform", trainable=True
        )
        self.bb = self.add_weight(
            name="bb", shape=(input_shape[1],), initializer="uniform", trainable=True
        )
        # periodic
        self.wa = self.add_weight(
            name="wa",
            shape=(1, input_shape[1], self.k),
            initializer="uniform",
            trainable=True,
        )
        self.ba = self.add_weight(
            name="ba",
            shape=(1, input_shape[1], self.k),
            initializer="uniform",
            trainable=True,
        )
        super(Time2Vec, self).build(input_shape)

    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = k.dot(inputs, self.wa) + self.ba
        wgts = k.sin(dp)  # or K.cos(.)

        ret = k.concatenate([k.expand_dims(bias, -1), wgts], -1)
        ret = k.reshape(ret, (-1, inputs.shape[1] * (self.k + 1)))
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] * (self.k + 1)


class RGAT(l.Layer):
    """
    Learning Attention-based Embeddings for Relation Prediction in Knowledge Graph (ARKG)
    Knowledge Graph Embedding using Graph Convolutional Networks with Relation-Aware Attention (AAKG)
    KGAT - Knowledge Graph Attention Network for Recommendation (KGAT)
    r-GAT: Relational Graph Attention Network for Multi-Relational Graphs (RGAT)
    """

    def __init__(
            self,
            hidden_dim=10,
            input_transform_activation=None,
            scoring_activation="sigmoid",
            attention_scoring_type="bilinear",
            ego_aggregation_type="bi_interaction",
            architecture="arkg",
            **kwargs,
    ):
        super(RGAT, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.input_transform_activation = input_transform_activation
        self.scoring_activation = scoring_activation
        self.ego_aggregation_type = ego_aggregation_type
        self.attention_scoring_type = attention_scoring_type
        self.architecture = architecture

        self.x_emb = l.Dense(self.hidden_dim, self.input_transform_activation)
        self.r_embed = l.Dense(self.hidden_dim, self.input_transform_activation)

        if self.attention_scoring_type == "bilinear":
            self.scoring = BilinearRelationalScoringSparse(self.scoring_activation)
        elif self.attention_scoring_type == "linear":
            self.scoring = LinearScoringSparse(self.scoring_activation)
        elif self.attention_scoring_type == "kgat":
            self.scoring = KgatScoring()

        self.score2adj = ScoreSoftmax()
        self.aggregate = AggregateEmbedding()

    def call(self, inputs, *args, **kwargs):
        x, r, a = inputs
        assert isinstance(a, tf.sparse.SparseTensor)
        # map inputs to embeddings
        x1 = self.x_emb(x)
        r1 = self.r_embed(r)
        # get embeddings scores
        scores = self.scoring(x1, r1, a)
        # get normalized adj scores
        adj = self.score2softmax(scores, a)
        h = self.aggregate()


class AttentionBasedRelationPrediction(l.Layer):
    def __init__(
            self,
            nodes_embedding_dim,
            relations_embedding_dim,
            edge_embedding_dim,
            edge_activation="relu",
    ):
        super(AttentionBasedRelationPrediction, self).__init__()
        self.nodes_embedding_dim = nodes_embedding_dim
        self.relations_embedding_dim = relations_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.edge_activation = edge_activation
        self.edge_embedding_func = LinearScoringSparse(
            self.edge_activation, self.edge_embedding_dim
        )
        self.score2adj = ScoreSoftmax()
        self.aggregate = AggregateEmbedding()

    def build(self, input_shape):
        x, r, a = input_shape
        self.w_e = self.add_weight(
            "W_edge_score",
            shape=(self.edge_embedding_dim, 1),
            initializer="glorot_normal",
        )
        self.w_x = self.add_weight(
            "W_nodes_embedding",
            shape=(x[-1], self.nodes_embedding_dim),
            initializer="glorot_normal",
        )
        self.w_r = self.add_weight(
            "W_nodes_embedding",
            shape=(r[-1], self.relations_embedding_dim),
            initializer="glorot_normal",
        )

    def call(self, inputs, *args, **kwargs):
        x, r, a = inputs
        x_emb = tf.matmul(x, self.w_x)
        r_emb = tf.matmul(r, self.w_r)
        e_emb = self.edge_embedding_func([x_emb, r_emb, a])  # Exd
        scores = tf.matmul(e_emb, self.w_e)  # Ex1
        scores = tf.reshape(scores, [-1])  # E
        adj = self.score2adj([scores, a])
        h = self.aggregate([e_emb, adj])
        return h, r_emb


class LinearScoringSparse(l.Layer):
    """
    Implements a linear scoring function:
    Formula: (x_i||x_j||r_ij).dot(W) : W € R^(2f+f_r x 1)
    """

    def __init__(self, activation="sigmoid", units=1):
        super(LinearScoringSparse, self).__init__()
        self.activation = activation
        self.units = units

    def build(self, input_shape):
        x, r, a = input_shape
        self.w = self.add_weight(
            name="linear_scoring_weight", shape=(2 * x[-1] + r[-1], self.units)
        )

    def call(self, inputs, *args, **kwargs):
        x, r, a = inputs
        assert isinstance(a, tf.sparse.SparseTensor)
        k, i, j = a.indices[:, 0], a.indices[:, 1], a.indices[:, 2]
        x_source = tf.gather(x, i)
        x_target = tf.gather(x, j)
        r_emb = tf.gather(r, k)
        concat = tf.concat([x_source, x_target, r_emb], -1)
        scores = tf.matmul(concat, self.w)
        scores = tf.reshape(scores, [-1, self.units])
        scores = activations.get(self.activation)(scores)
        return scores


class BilinearRelationalScoringSparse(l.Layer):
    """
    Implements a bilinear score between source-relation, target-relation and source-target nodes
    Formula: x_i.dot(W1).dot(r_i) + x_j.dot(W2).dot(r_i) + x_i.dot(W3).dot(x_j)
    """

    def __init__(self, activation="sigmoid", dropout_rate=0.5, regularize=False):
        super(BilinearRelationalScoringSparse, self).__init__()
        self.activation = activation
        self.regularize = regularize
        self.dropout_rate = dropout_rate
        self.dropout_x = l.Dropout(self.dropout_rate)
        self.dropout_r = l.Dropout(self.dropout_rate)

    def build(self, input_shape):
        x, r, a = input_shape
        if self.regularize:
            regularizer = "l2"
        else:
            regularizer = None
        self.x_source_r_kernel = self.add_weight(
            name="bilinear_source2rel",
            shape=(x[-1], r[-1]),
            initializer="glorot_normal",
            regularizer=regularizer,
        )
        self.x_target_r_kernel = self.add_weight(
            name="bilinear_target2rel",
            shape=(x[-1], r[-1]),
            initializer="glorot_normal",
            regularizer=regularizer,
        )
        self.x_source_target_kernel = self.add_weight(
            name="bilinear_source2target",
            shape=(x[-1], x[-1]),
            initializer="glorot_normal",
            regularizer=regularizer,
        )

    def call(self, inputs, *args, **kwargs):
        x, r, a = inputs
        if self.dropout_rate:
            x = self.dropout_x(x)
            r = self.dropout_r(r)
        assert isinstance(a, tf.sparse.SparseTensor)
        k, i, j = a.indices[:, 0], a.indices[:, 1], a.indices[:, 2]
        x_source = tf.gather(x, i)
        x_target = tf.gather(x, j)
        r_emb = tf.gather(r, k)
        x_source2r_score = tf.einsum(
            "ij,jk,ik->i", x_source, self.x_source_r_kernel, r_emb
        )
        x_target2r_score = tf.einsum(
            "ij,jk,ik->i", x_target, self.x_target_r_kernel, r_emb
        )
        x_source2target_score = tf.einsum(
            "ij,jk,ik->i", x_source, self.x_source_target_kernel, x_target
        )
        score = x_source2r_score + x_target2r_score + x_source2target_score
        score = activations.get(self.activation)(score)
        return score


# TODO
class LinearScoringDense(l.Layer):
    def __init__(self, activation=None, regularizer=None):
        super(LinearScoringDense, self).__init__()
        self.activation = activation
        self.regularizer = regularizer

    def build(self, input_shape):
        x, r, a = input_shape
        self.x_source_kernel = self.add_weight(
            name="w_xs",
            shape=(x[-1], 1),
            initializer="glorot_normal",
            regularizer=self.regularizer,
        )
        self.x_target_kernel = self.add_weight(
            name="w_xt",
            shape=(x[-1], 1),
            initializer="glorot_normal",
            regularizer=self.regularizer,
        )
        self.r_kernel = self.add_weight(
            name="w_r",
            shape=(r[-1], 1),
            initializer="glorot_normal",
            regularizer=self.regularizer,
        )

    def call(self, inputs, *args, **kwargs):
        x, r, a = inputs
        x_s_score = tf.matmul(x, self.x_source_kernel)
        x_t_score = tf.matmul(x, self.x_target_kernel)
        r_score = tf.matmul(r, self.r_kernel)
        attn_s_t = x_s_score + tf.transpose(x_t_score)  # NxN
        attn = tf.expand_dims(attn_s_t, 1) + r_score  # NxNxR
        attn = tf.transpose(attn, (1, 0, 2))  # RxNxN
        attn = activations.get(self.activation)(attn)
        return attn


# TODO
class BilinearRelationalScoringDense(l.Layer):
    def __init__(self):
        super(BilinearRelationalScoringDense, self).__init__()
        self.x_source_r = BilinearDecoderDense()
        self.x_target_r = BilinearDecoderDense()
        self.x_source_target = BilinearDecoderDense()

    def call(self, inputs, *args, **kwargs):
        x, r, a = inputs
        x_s_r = self.x_source_r()


class KgatScoring(l.Layer):
    def __init__(self):
        super(KgatScoring, self).__init__()
        self.Wr = []

    def build(self, input_shape):
        x, r, a = input_shape

        for i in range(a[0]):
            wr = self.add_weight(f"Wr_scoring_{i}", shape=(x[-1], r[-1]))
            self.Wr.append(wr)

    def call(self, inputs, *args, **kwargs):
        x, r, a = inputs
        assert isinstance(a, tf.sparse.SparseTensor)
        k, i, j = a.indices[:, 0], a.indices[:, 1], a.indices[:, 2]
        x_source = tf.gather(x, i)
        x_target = tf.gather(x, j)
        r_emb = tf.gather(r, k)
        wr = tf.gather(self.Wr, k)
        score1 = tf.einsum("ij,ijk->ik", x_target, wr)
        score2 = tf.einsum("ij,ijk->ik", x_source, wr)
        score_tanh = tf.nn.tanh(score2 + r_emb)
        score = tf.einsum("ij,ij->i", score1, score_tanh)
        return score


class ScoreSoftmax(l.Layer):
    def __init__(self):
        super(ScoreSoftmax, self).__init__()

    def call(self, inputs, *args, **kwargs):
        score, a = inputs
        assert isinstance(a, tf.sparse.SparseTensor) and len(score) == len(a.indices)
        adj = tf.sparse.SparseTensor(a.indices, score, a.shape)
        adj = tf.sparse.transpose(adj, (1, 2, 0))  # Ni X Nj x R
        adj_reshaped = tf.sparse.reshape(adj, (a.shape[-1], -1))  # Ni x (Nj*R)
        soft = tf.sparse.softmax(adj_reshaped)
        soft = tf.sparse.reshape(soft, adj.shape)  # Ni X Nj x R
        soft = tf.sparse.transpose(soft, (2, 0, 1))  # R x Ni x Nj
        return soft


class AggregateEmbedding(l.Layer):
    def __init__(self):
        super(AggregateEmbedding, self).__init__()

    def call(self, inputs, *args, **kwargs):
        """
        Aggregation based on 'Learning Attention-based Embeddings for Relation Prediction in Knowledge Graph'
        Inputs:
            x: Initial Node Embeddings of shape Nxf where N is the number of nodes and f is the hidden dimension
            e_kij: List of vector embedding of edges e_kij from linear scoring of shape N_e x f where N_e is the number
                   of edges and f in the hidden dimension
            a: adjacency sparse tensor of shape R x N x N
        """
        e_kij, a = inputs
        d = e_kij.shape[-1]
        r, n, n = a.shape[0], a.shape[1], a.shape[2]
        if isinstance(a, tf.Tensor):
            a = tf.sparse.from_dense(a)
        k, i, j = a.indices[:, 0], a.indices[:, 1], a.indices[:, 2]
        h_list = tf.expand_dims(a.values, -1) * e_kij
        h = tf.tensor_scatter_nd_add(
            tf.zeros((n, d), dtype=a.values.dtype), tf.expand_dims(i, -1), h_list
        )
        return h
