import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.keras import activations
from tensorflow.keras import initializers
import tensorflow.keras.backend as k
from tensorflow.keras.losses import mean_squared_error, sparse_categorical_crossentropy

physical_devices = tf.config.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(physical_devices[0], True)


class BilinearLayer(l.Layer):
    """"
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
                    self.add_weight(shape=(self.features, self.features), trainable=True, name=f"Relation{i}"))

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

    ein1 = tf.einsum('ij,il->ilj', S, R)
    ein2 = tf.einsum('ijl,in->iljn', ein1, T)
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


class TrippletScore(l.Layer):
    def __init__(self):
        super(TrippletScore, self).__init__()
        self.initializer = initializers.GlorotNormal()

    def build(self, input_shape):
        kernel_shape = input_shape[-3:]
        self.kernel = tf.Variable(initial_value=self.initializer(shape=kernel_shape))

    def call(self, inputs, **kwargs):
        return outer_kernel_product(inputs, self.kernel)


class TrippletDecoder(l.Layer):
    def __init__(self, **kwargs):
        super(TrippletDecoder, self).__init__(**kwargs)
        self.dec1 = TrippletProductSparse()
        self.dec2 = TrippletScore()

    def call(self, inputs, **kwargs):
        sparse_outer = self.dec1(inputs)
        return self.dec2(sparse_outer)


class NTN(l.Layer):
    """
    Implementation of Reasoning With Neural Tensor Networks for Knowledge Base Completion
    https://proceedings.neurips.cc/paper/2013/file/b337e84de8752b27eda3a12363109e80-Paper.pdf
    """

    def __init__(self, activation="tanh", k=5, **kwargs):
        """
        Parameters:
            activation: type of activation to be used
            k: number of kernels for each type of relation
        """
        super(NTN, self).__init__(**kwargs)
        self.activation = activation
        if self.activation == "relu":
            self.activation = activations.relu
        if self.activation == "tanh":
            self.activation = activations.tanh
        if self.activation == "sigmoid":
            self.activation = activations.sigmoid
        self.k = k

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        self.initializer = initializers.GlorotNormal()
        # Relation Specific Parameters
        self.Wr = []
        self.ur = []
        self.br = []
        self.V1r = []
        self.V2r = []
        for _ in range(A_shape[0]):
            self.V1r.append(tf.Variable(self.initializer(shape=(X_shape[-1], self.k))))
            self.V2r.append(tf.Variable(self.initializer(shape=(X_shape[-1], self.k))))
            self.ur.append(tf.Variable(self.initializer(shape=(self.k, 1))))
            self.br.append(tf.Variable(self.initializer(shape=(self.k,))))
            self.Wr.append(tf.Variable(self.initializer(shape=(X_shape[-1], X_shape[-1], self.k))))

    def call(self, inputs, **kwargs):
        X, A = inputs
        return self.ntn(X, A)

    def ntn(self, X, A: tf.sparse.SparseTensor):
        e1 = tf.gather(X, A.indices[:, 1])
        e2 = tf.gather(X, A.indices[:, 2])
        wr = tf.gather(self.Wr, A.indices[:, 0])
        ur = tf.gather(self.ur, A.indices[:, 0])
        br = tf.gather(self.br, A.indices[:, 0])
        v1 = tf.gather(self.V1r, A.indices[:, 0])
        v2 = tf.gather(self.V2r, A.indices[:, 0])
        bilinear = tf.einsum("ni,nijk->njk", e1, wr)
        bilinear = tf.einsum("njk,ij->nk", bilinear, e2)
        v1 = tf.einsum("nd,ndk->nk", e1, v1)  # nxd, nxdxk -> nxk
        v2 = tf.einsum("nd,ndk->nk", e2, v2)  # nxd, nxdxk -> nxk
        vt = v1 + v2  # nxk+nxk -> nxk
        activated = self.activation(bilinear + vt + br)  # nxk
        score = tf.einsum("nk,nkj->nj", activated, ur)  # nxk,nxkx1 -> nx1
        return score


class RGHAT(l.Layer):
    """
    Implementation of Relational Graph Neural Network with Hierarchical Attention
    https://ojs.aaai.org/index.php/AAAI/article/view/6508
    """

    def __init__(self,
                 nodes_features_dim,
                 relations_features_dim,
                 heads=5,
                 embedding_combination="additive",
                 dropout_rate=None,
                 attention_activation="relu",
                 include_adj_values=False,
                 message_activation=None,
                 **kwargs
                 ):
        super(RGHAT, self).__init__(**kwargs)
        self.nodes_features_dim = nodes_features_dim
        self.relations_features_dim = relations_features_dim
        self.heads = heads
        self.embedding_combination = embedding_combination
        assert self.embedding_combination in ["additive", "multiplicative", "bi-interaction"]
        self.dropout_rate = dropout_rate
        self.attention_activation = attention_activation
        self.initializer = initializers.GlorotNormal()
        self.include_adj_values = include_adj_values
        self.message_activation = message_activation
        assert self.message_activation in ["relu", "sigmoid", "tanh", "leaky_relu", None]
        if self.message_activation == "relu":
            self.message_activation = activations.relu
        elif self.message_activation == "sigmoid":
            self.message_activation = activations.sigmoid
        elif self.message_activation == "leaky_relu":
            self.message_activation = l.LeakyReLU(0.2)
        elif self.message_activation == "tanh":
            self.message_activation = activations.tanh

    def build(self, input_shape):
        """
        Build the model parameters
            Parameters:
                input_shape: input shape array [X_shape, R_shape, A_shape] where X is the nodes features matrix, R is
                             the relations features matrix and A is the adjacency tensor of size RxNxN
        """
        X_shape, R_shape, A_shape = input_shape

        #for _ in range(self.heads):
        self.s_m_kernel = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], self.nodes_features_dim)),
                                      dtype=tf.float32)
        self.t_m_kernel = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], self.nodes_features_dim)),
                                      dtype=tf.float32)
        self.r_m_kernel = tf.Variable(initial_value=self.initializer(shape=(R_shape[-1], self.relations_features_dim)),
                                      dtype=tf.float32)
        self.S_sr_kenrel = tf.Variable(
            initial_value=self.initializer(shape=(self.nodes_features_dim, self.nodes_features_dim)), dtype=tf.float32)
        self.R_sr_kernel = tf.Variable(
            initial_value=self.initializer(shape=(self.nodes_features_dim, self.nodes_features_dim)), dtype=tf.float32)
        self.T_srt_kenrel = tf.Variable(
            initial_value=self.initializer(shape=(self.nodes_features_dim, self.nodes_features_dim)), dtype=tf.float32)
        self.SR_srt_kenrel = tf.Variable(
            initial_value=self.initializer(shape=(self.nodes_features_dim, self.nodes_features_dim)), dtype=tf.float32)
        self.sr_attention_kernel = tf.Variable(initial_value=self.initializer(shape=(self.nodes_features_dim, 1)),
                                               dtype=tf.float32)
        self.srt_attention_kernel = tf.Variable(initial_value=self.initializer(shape=(self.nodes_features_dim, 1)),
                                                dtype=tf.float32)

    def source_message(self, X):
        """
        Linear transformation of sources to make them messages
            Parameters:
                X: Nodes features matrix of dimensions Nxf where N is the number of nodes and f is the feature dimension
        """
        f = tf.matmul(X, self.s_m_kernel)
        if self.message_activation is None:
            return f
        else:
            return self.message_activation(f)

    def target_message(self, X):
        f = tf.matmul(X, self.t_m_kernel)
        if self.message_activation is None:
            return f
        else:
            return self.message_activation(f)

    def relation_message(self, R):
        f = tf.matmul(R, self.r_m_kernel)
        if self.message_activation is None:
            return f
        else:
            return self.message_activation(f)

    def additive(self, embeddings, update_embeddings):
        """
        Additive update of node embeddings: activation((emd + update_emb)*W) W_shape = dxd
            Params:
                embeddings: list of embeddings of each edge, shape Exd
                update_embeddings: list of new embeddings of each edge, shape Exd
        """
        self.W = tf.Variable(initial_value=self.initializer(shape=(self.nodes_features_dim, self.nodes_features_dim)))
        sum_emb = embeddings + update_embeddings  # shape: Exd
        return activations.relu(tf.matmul(sum_emb, self.W))

    def multiplicative(self, embeddings, update_embeddings):
        """
        Multiplicative update of node embeddings: activation((emd * update_emb)*W)
            Params:
                    embeddings: list of embeddings of each edge, shape Exd
                    update_embeddings: list of new embeddings of each edge, shape Exd
        """
        self.W = tf.Variable(initial_value=self.initializer(shape=(self.nodes_features_dim, self.nodes_features_dim)))
        mult_emb = embeddings * update_embeddings
        return activations.relu(tf.matmul(mult_emb, self.W))

    def bi_interaction(self, embeddings, update_embeddings):
        """
        Bi-interaction update of node embeddings: 0.5*[additive(emd, update_emb) + multiplicative(emb, update_emb)]
            Params:
                    embeddings: list of embeddings of each edge, shape Exd
                    update_embeddings: list of new embeddings of each edge, shape Exd
        """
        return 0.5 * (self.additive(embeddings, update_embeddings) + self.multiplicative(embeddings, update_embeddings))

    def call(self, inputs, **kwargs):
        """
        Computes new embeddins based on Relational-level attention and Entity-level attention.
        Relational-level attention:
            a_sr = [source_embedding||relation_embedding]*W1 : W1 shape is 2dxd
            attention_sr = softmax(a_sr) = exp(activation(a_sr*sr_kernel))/sum_r[exp(activation(a_sr*sr_kernel))]
            sr_kernel shape is dx1
            W1 can be decomposed by W_s and W_r with shapes dxd, so that the multiplication becomes:
                a_s = source_embedding * W_s
                a_r = relation_embedding * W_r
                a_sr = a_s + a_r
        Entity-level attention:
            a_srt = [a_sr||target_embedding]*W2 : W2 shape is 2dxd
            attention_sr_t = softmax(a_srt) = exp(activation(a_srt*srt_kernel))/sum_t[exp(activation(a_srt*srt_kernel))]
            srt_kernel shape is dx1
            W1 can be decomposed by W_sr and W_t with shapes dxd, so that the multiplication becomes:
                a_sr = source_embedding * W_sr
                a_t = relation_embedding * W_t
                a_srt = a_sr + a_t
        """
        X, R, A = inputs
        message_s = self.source_message(X)
        message_t = self.target_message(X)
        message_r = self.relation_message(R)

        # Relational level attention s->r
        a_s = tf.gather(message_s, A.indices[:, 1])
        a_s = tf.matmul(a_s, self.S_sr_kenrel)
        a_r = tf.gather(message_r, A.indices[:, 0])
        a_r = tf.matmul(a_r, self.R_sr_kernel)
        a_sr = a_s + a_r
        a_sr_attention = tf.matmul(a_sr, self.sr_attention_kernel)
        attention_sr = unsorted_segment_softmax(a_sr_attention, A.indices[:, 0])
        if self.dropout_rate:
            attention_sr = l.Dropout(self.dropout_rate)(attention_sr)

        # Entity level attention sr->t
        a_sr_ = tf.matmul(a_sr, self.SR_srt_kenrel)  # shape: Exd
        a_t = tf.gather(message_t, A.indices[:, 2])
        a_t = tf.matmul(a_t, self.T_srt_kenrel)  # shape: Exd
        a_srt = a_sr_ + a_t  # shape: Exd
        a_sr_t_attention = tf.matmul(a_srt, self.srt_attention_kernel)
        attention_sr_t = unsorted_segment_softmax(a_sr_t_attention, A.indices[:, 2])  # shape: Ex1
        if self.dropout_rate:
            attention_sr_t = l.Dropout(self.dropout_rate)(attention_sr_t)  # shape: Ex1

        attention_srt = attention_sr * attention_sr_t
        update_t = attention_srt * tf.gather(a_sr, A.indices[:, 2])
        update_embeddings = tf.math.unsorted_segment_sum(update_t, A.indices[:, 2], A.shape[2])

        if self.embedding_combination == "additive":
            new_embeddings = self.additive(message_t, update_embeddings)
        if self.embedding_combination == "multiplicative":
            new_embeddings = self.multiplicative(message_t, update_embeddings)
        if self.embedding_combination == "bi-interaction":
            new_embeddings = self.bi_interaction(message_t, update_embeddings)

        return new_embeddings, R, A


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
        X1_shape, X2_shape, edgelist_shape = input_shape
        if self.output_dim is None:
            self.output_dim = X1_shape[-1]
        self.kernel = tf.Variable(initial_value=self.initializer(shape=(X1_shape[-1], self.nodes_features_dim)))

    def call(self, inputs, **kwargs):
        X1, X2, edgelist = inputs
        h1 = tf.gather(X1, edgelist[:, 0])
        h2 = tf.gather(X2, edgelist[:, 1])
        outer = tf.einsum('ij,ik->ijk', h1, h2)
        new_emb_list = tf.matmul(outer, self.kernel)
        attention = unsorted_segment_softmax(new_emb_list, edgelist[:, 1])
        if self.dropout_rate:
            attention = l.Dropout(self.dropout_rate)(attention)
        attended_embeddings = new_emb_list * attention
        update_embeddings = tf.math.unsorted_segment_sum(attended_embeddings, edgelist[:, 1])
        return update_embeddings


class GAIN(l.Layer):
    """
    Implementation of GAIN: Graph Attention & Interaction Network for Inductive Semi-Supervised Learning over
    Large-scale Graphs
    https://arxiv.org/abs/2011.01393
    """

    def __init__(self, aggregators_list, output_dim, **kwargs):
        super(GAIN, self).__init__(**kwargs)
        self.aggregators_list = aggregators_list
        self.output_dim = output_dim
        self.initializer = initializers.GlorotNormal()

    def build(self, input_shape):
        X_shape, R_shape, A_shape = input_shape
        self.aggregator_kernel_self = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], 1)))
        self.aggregator_kernel2_aggregator = None


class GAT(l.Layer):
    def __init__(self, **kwargs):
        """
        Implementation of Graph Atttention Networks: https://arxiv.org/pdf/2102.07200.pdf
        """
        super(GAT, self).__init__(**kwargs)
        self.initializer = initializers.GlorotNormal()

    def build(self, input_shape):
        X_shape, edgelist = input_shape
        self.kernel1 = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], 1)))
        self.kernel2 = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], 1)))

    def call(self, inputs, **kwargs):
        X, edgelist = inputs
        x1 = tf.gather(X, edgelist[:, 0])
        x1 = tf.matmul(x1, self.kernel1)
        x2 = tf.gather(X, edgelist[:, 1])
        x2 = tf.matmul(x2, self.kernel2)
        att_score = x1 + x2
        attention = unsorted_segment_softmax(att_score, edgelist[:, 1])
        attended_embeddings = X * attention
        update_embeddings = tf.math.unsorted_segment_sum(attended_embeddings, edgelist[:, 1])
        return update_embeddings


class GCN(l.Layer):
    """
    Implementation of Graph Convolutional Neural Netwroks: https://arxiv.org/pdf/1609.02907.pdf
    """

    def __init__(self, hidden_dim, output_dim, dropout_rate, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.initializer = initializers.GlorotNormal()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        self.W1 = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], self.hidden_dim)))
        self.W2 = tf.Variable(initial_value=self.initializer(shape=(self.hidden_dim, self.output_dim)))

    def call(self, inputs, **kwargs):
        X, A = inputs
        A_aug = self.add_identity(A)
        D_inv_sqrt = self.inverse_degree(A_aug)
        A_norm = tf.matmul(D_inv_sqrt, A_aug)
        A_norm = tf.matmul(A_norm, D_inv_sqrt)
        t1 = tf.matmul(A_norm, X)
        t2 = tf.matmul(t1, self.W1)
        t3 = activations.relu(t2)
        if self.dropout_rate:
            t3 = l.Dropout(self.dropout_rate)(t3)
        t4 = tf.matmul(A_norm, t3)
        t5 = tf.matmul(t4, self.W2)
        return activations.softmax(t5)

    def add_identity(self, A):
        I = tf.eye(*A.shape)
        return A + I

    def inverse_degree(self, A):
        D = tf.math.reduce_sum(A, axis=-1)
        D_inv_sq = tf.math.sqrt(1 / D)
        D_inv_sq = tf.linalg.diag([D_inv_sq])[0]
        return D_inv_sq


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


'''def attention3d(self, X: tf.Tensor, R, A: tf.sparse.SparseTensor):
    self_attention = tf.matmul(message_s, self.source_attention_kernel)
    self_attention = tf.gather(self_attention, A.indices[:, 1])
    target_attention = tf.matmul(message_t, self.target_attention_kernel)
    target_attention = tf.gather(target_attention, A.indices[:, 2])
    relation_attention = tf.matmul(message_r, self.relation_attention_kernel)
    relation_attention = tf.gather(relation_attention, A.indices[:, 0])
    source_relation_attention = self_attention + relation_attention

    if self.attention_activation == "relu":
        source_relation_attention = activations.relu(source_relation_attention)
    if self.attention_activation == "leaky_relu":
        attention = l.LeakyReLU(0.2)(source_relation_attention)

    #attention = attention * A.values
    #attention = stable_softmax_3d(attention, A)
    #attention = tf.sparse.from_dense(attention, A.indices, A.shape)
    if self.dropout_rate is not None:
        attention = l.Dropout(self.dropout_rate)(attention)
    return attention'''


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


if __name__ == '__main__':
    nodes = 20
    ft = 10
    r = 15

    A = make_data(1, 15, 20)
    R = tf.Variable(np.random.normal(size=(r, ft)), dtype=tf.float32)
    X = tf.Variable(np.random.normal(size=(nodes, ft)), dtype=tf.float32)

    rghat = RGHAT(15, 15, "additive", .5)
    A0 = tf.sparse.slice(A, (0, 0, 0, 0), (1, r, nodes, nodes))
    A0_dense = tf.sparse.to_dense(A0)
    A0_squeeze = tf.squeeze(A0_dense, axis=0)
    A0_sparse = tf.sparse.from_dense(A0_squeeze)
    new_emb, R, A = rghat([X, R, A0_sparse])
    # print(new_emb)
