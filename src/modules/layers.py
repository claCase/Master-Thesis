import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.keras import activations
from tensorflow.keras import initializers
import tensorflow.keras.backend as k
from src.modules.graph_utils import make_data
from tensorflow.keras.losses import mean_squared_error, sparse_categorical_crossentropy


# physical_devices = tf.config.list_physical_devices('GPU')

# tf.config.experimental.set_memory_growth(physical_devices[0], True)


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


class Bilinear(l.Layer):
    def __init__(self, hidden_dim=5, **kwargs):
        super(Bilinear, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.initializer = initializers.GlorotNormal()

    def build(self, input_shape):
        print("Building model")
        self.R = tf.Variable(initial_value=self.initializer(shape=(self.hidden_dim, self.hidden_dim)))
        self.X = tf.Variable(initial_value=self.initializer(shape=(input_shape[0], self.hidden_dim)))

    def call(self, inputs, **kwargs):
        x_left = tf.matmul(self.X, self.R)
        A = tf.matmul(x_left, x_left, transpose_b=True)
        return A


class BilinearSparse(l.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(BilinearSparse, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.initializer = initializers.GlorotNormal()

    def build(self, input_shape):
        self.X = tf.Variable(initial_value=self.initializer(shape=(input_shape[0], self.hidden_dim)))
        self.R = tf.Variable(initial_value=self.initializer(shape=(self.hidden_dim, self.hidden_dim)))

    def call(self, inputs, **kwargs):
        edgelist, values = inputs.indices, inputs.values
        e1 = tf.gather(self.X, edgelist[:, 0])
        e2 = tf.gather(self.X, edgelist[:, 1])
        left = tf.matmul(e1, self.R)  # ei,ii->ei
        right = tf.multiply(left, e2)
        right = tf.reduce_sum(right, 1)
        # right = tf.einsum("ei,ei->e", left, e2)
        A = tf.sparse.SparseTensor(edgelist, right, inputs.shape)
        return A


class BilinearDecoderDense(l.Layer):
    def __init__(self, **kwargs):
        super(BilinearDecoderDense, self).__init__(**kwargs)
        self.initializer = initializers.GlorotNormal()

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        self.R = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], X_shape[-1])), name="R_bilinear")

    def call(self, inputs, **kwargs):
        X, A = inputs
        x_left = tf.matmul(X, self.R)
        A = tf.matmul(x_left, X, transpose_b=True)
        return X, A


class BilinearDecoderSparse(l.Layer):
    def __init__(self, activation="relu", **kwargs):
        super(BilinearDecoderSparse, self).__init__(**kwargs)
        self.initializer = initializers.GlorotNormal()
        self.activation = activation

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        self.R_kernel = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], X_shape[-1])))

    def call(self, inputs, **kwargs):
        X, A = inputs
        i, j = A.indices[:, 0], A.indices[:, 1]
        '''i = tf.range(A.shape[0])
        j = tf.range(A.shape[1])
        c = tf.stack(tf.meshgrid(i, j, indexing='ij'), axis=-1)
        c = tf.reshape(c, (-1, 2)).numpy()
        i = c[:,0]
        j = c[:,1]'''
        e1 = tf.gather(X, i)
        e2 = tf.gather(X, j)
        left = tf.einsum("ij,jj->ij", e1, self.R_kernel)
        right = tf.einsum("ij,ij->i", left, e2)
        right = activations.get(self.activation)(right)
        A_pred = tf.sparse.SparseTensor(A.indices, right, A.shape)
        #A_pred = tf.sparse.to_dense(A_pred)
        return [X, A_pred]


class InnerProductDenseDecoder(l.Layer):
    def __init__(self, **kwargs):
        super(InnerProductDenseDecoder, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        X, A = inputs
        return X, tf.matmul(X, X, transpose_b=True)


class InnerProductSparseDecoder(l.Layer):
    def __init__(self, **kwargs):
        super(InnerProductSparseDecoder, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        X, A = inputs
        i, j = A.indices[:, 0], A.indices[:, 1]
        e1 = tf.gather(X, i)
        e2 = tf.gather(X, j)
        inner = tf.einsum("ij,ij->i", e1, e2)
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

    ein1 = tf.einsum('ij,il->ijl', R, S)  # [start_idx:end_idx], S[start_idx:end_idx])
    ein2 = tf.einsum('ijl,in->ijln', ein1, T)  # T[start_idx:end_idx])
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
            self.kernel = tf.Variable(initial_value=self.initializer(shape=kernel_shape))

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


class TrippletScoreFlatSparse(l.Layer):
    def __init__(self):
        super(TrippletScoreFlatSparse, self).__init__()
        self.initializer = initializers.GlorotNormal()

    def build(self, input_shape):
        X_shape, R_shape, A_shape = input_shape
        self.kernel_x1 = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1])))
        self.kernel_x2 = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1])))
        self.kernel_r = tf.Variable(initial_value=self.initializer(shape=(R_shape[-1])))

    def call(self, inputs, **kwargs):
        X, R, A = inputs

        r = tf.gather(R, A.indices[:0])
        x1 = tf.gather(X, A.indices[:1])
        x2 = tf.gather(X, A.indices[:2])
        x1_k = tf.matmul(x1, self.kernel_x1)
        x2_k = tf.matmul(x2, self.kernel_x2)
        r_k = tf.matmul(r, self.kernel_r)
        tot = x1_k + x2_k + r_k
        return tot


class TensorDecompositionLayer(l.Layer):
    def __init__(self, k, **kwargs):
        super(TensorDecompositionLayer, self).__init__(**kwargs)
        self.k = k
        self.initializer = initializers.GlorotNormal()

    def build(self, input_shape):
        A_shape = input_shape
        self.kernels = []
        for i in range(self.k):
            x = tf.Variable(initial_value=self.initializer(shape=(A_shape[1],)), trainable=True)
            r = tf.Variable(initial_value=self.initializer(shape=(A_shape[0],)), trainable=True)
            self.kernels.append(
                {"x": x,
                 "r": r}
            )

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
        print("Building NTN")
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
        # Relational specific parameters
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
        score = tf.squeeze(score)
        A_score = tf.sparse.SparseTensor(A.indices, score, A.shape)
        return X, A_score


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
                 attention_activation="sigmoid",
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
        self.message_activation = activations.get(self.message_activation)

    def build(self, input_shape):
        """
        Build the model parameters
            Parameters:
                input_shape: input shape array [X_shape, R_shape, A_shape] where X is the nodes features matrix, R is
                             the relations features matrix and A is the adjacency tensor of size RxNxN
        """
        X_shape, R_shape, A_shape = input_shape
        self.heads_variables = []
        for _ in range(self.heads):
            self.s_m_kernel = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], self.nodes_features_dim)),
                                          dtype=tf.float32)
            self.t_m_kernel = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], self.nodes_features_dim)),
                                          dtype=tf.float32)
            self.r_m_kernel = tf.Variable(
                initial_value=self.initializer(shape=(R_shape[-1], self.relations_features_dim)),
                dtype=tf.float32)
            S_sr_kernel = tf.Variable(
                initial_value=self.initializer(shape=(self.nodes_features_dim, self.nodes_features_dim)),
                dtype=tf.float32)
            R_sr_kernel = tf.Variable(
                initial_value=self.initializer(shape=(self.nodes_features_dim, self.nodes_features_dim)),
                dtype=tf.float32)
            T_srt_kernel = tf.Variable(
                initial_value=self.initializer(shape=(self.nodes_features_dim, self.nodes_features_dim)),
                dtype=tf.float32)
            SR_srt_kernel = tf.Variable(
                initial_value=self.initializer(shape=(self.nodes_features_dim, self.nodes_features_dim)),
                dtype=tf.float32)
            sr_attention_kernel = tf.Variable(initial_value=self.initializer(shape=(self.nodes_features_dim, 1)),
                                              dtype=tf.float32)
            srt_attention_kernel = tf.Variable(initial_value=self.initializer(shape=(self.nodes_features_dim, 1)),
                                               dtype=tf.float32)
            self.heads_kernel = tf.Variable(initial_value=self.initializer(shape=(1, self.heads)))

            self.heads_variables.append(
                {
                    "S_sr_kernel": S_sr_kernel,
                    "R_sr_kernel": R_sr_kernel,
                    "T_srt_kernel": T_srt_kernel,
                    "SR_srt_kernel": SR_srt_kernel,
                    "sr_attention_kernel": sr_attention_kernel,
                    "srt_attention_kernel": srt_attention_kernel
                }
            )

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
        self.W = self.add_weight(initializer=self.initializer, shape=(self.nodes_features_dim, self.nodes_features_dim),
                                 trainable=True)
        sum_emb = embeddings + update_embeddings  # shape: Exd
        return activations.relu(tf.matmul(sum_emb, self.W))

    def multiplicative(self, embeddings, update_embeddings):
        """
        Multiplicative update of node embeddings: activation((emd * update_emb)*W)
            Params:
                    embeddings: list of embeddings of each edge, shape Exd
                    update_embeddings: list of new embeddings of each edge, shape Exd
        """
        self.W = self.add_weight(initializer=self.initializer, shape=(self.nodes_features_dim, self.nodes_features_dim),
                                 trainable=True)
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
        outputs_head = []
        edges = A.indices
        batch = len(edges)
        for i in range(self.heads):
            '''
            # Relational level attention s->r
            batch_emb_head = []
            for b in range(len(edges)//batch):
                star_idx = b * batch
                if not b%(len(edges)//batch):
                    end_idx = None
                else:
                    end_idx = b*(batch+1)'''
            a_s = tf.gather(message_s, edges[:, 1])  # [star_idx:end_idx, 1])
            a_s = tf.matmul(a_s, self.heads_variables[i]["S_sr_kernel"])
            a_r = tf.gather(message_r, edges[:, 0])  # star_idx:end_idx, 0])
            a_r = tf.matmul(a_r, self.heads_variables[i]["R_sr_kernel"])
            a_sr = a_s + a_r
            a_sr_attention = tf.matmul(a_sr, self.heads_variables[i]["sr_attention_kernel"])
            attention_sr = unsorted_segment_softmax(a_sr_attention, edges[:, 0])  # star_idx:end_idx, 0])
            if self.dropout_rate:
                attention_sr = l.Dropout(self.dropout_rate)(attention_sr)

            # Entity level attention sr->t
            a_sr_ = tf.matmul(a_sr, self.heads_variables[i]["SR_srt_kernel"])  # shape: Exd
            a_t = tf.gather(message_t, edges[:, 2])  # star_idx:end_idx, 2])
            a_t = tf.matmul(a_t, self.heads_variables[i]["T_srt_kernel"])  # shape: Exd
            a_srt = a_sr_ + a_t  # shape: Exd
            a_sr_t_attention = tf.matmul(a_srt, self.heads_variables[i]["srt_attention_kernel"])
            attention_sr_t = unsorted_segment_softmax(a_sr_t_attention,
                                                      edges[:, 2])  # star_idx:end_idx, 2])  # shape: Ex1
            if self.dropout_rate:
                attention_sr_t = l.Dropout(self.dropout_rate)(attention_sr_t)  # shape: Ex1

            attention_srt = attention_sr * attention_sr_t
            update_t = attention_srt * tf.gather(a_sr, edges[:, 2])  # star_idx:end_idx, 2])
            update_embeddings = tf.math.unsorted_segment_sum(update_t, edges[:, 2],
                                                             A.shape[2])  # star_idx:end_idx, 2], A.shape[2])

            if self.embedding_combination == "additive":
                new_embeddings = self.additive(message_t, update_embeddings)
            if self.embedding_combination == "multiplicative":
                new_embeddings = self.multiplicative(message_t, update_embeddings)
            if self.embedding_combination == "bi-interaction":
                new_embeddings = self.bi_interaction(message_t, update_embeddings)
            new_embeddings = tf.expand_dims(new_embeddings, 0)
            # batch_emb_head.append(new_embeddings)
            # batch_emb_head = tf.concat(batch_emb_head, 0)
            # batch_emb_head = tf.expand_dims(batch_emb_head, 0)
            outputs_head.append(new_embeddings)  # batch_emb_head)
        outputs_head = tf.concat(outputs_head, 0)
        # Combine heads
        head_weights = activations.softmax(self.heads_kernel)
        head_weights = l.Dropout(0.5)(head_weights)
        # Aggregate batch heads embeddings
        new_embeddings = tf.einsum("hnd,kh->knd", outputs_head, head_weights)
        new_embeddings = tf.squeeze(new_embeddings, 0)
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
        X1_shape, X2_shape, A = input_shape
        edgelist = A.indices
        if self.output_dim is None:
            self.output_dim = X1_shape[-1]
        self.kernel = tf.Variable(initial_value=self.initializer(shape=(X1_shape[-1], self.nodes_features_dim)))

    def call(self, inputs, **kwargs):
        X1, X2, A = inputs
        edgelist = A.indices
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
    def __init__(self, hidden_dim, dropout_rate=0.5, **kwargs):
        """
        Implementation of Graph Atttention Networks: https://arxiv.org/pdf/2102.07200.pdf
        """
        super(GAT, self).__init__(**kwargs)
        self.initializer = initializers.GlorotNormal()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        self.kernel1 = tf.Variable(initial_value=self.initializer(shape=(self.hidden_dim, 1)))
        self.kernel2 = tf.Variable(initial_value=self.initializer(shape=(self.hidden_dim, 1)))
        self.message_kernel_self = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], self.hidden_dim)))
        self.message_kernel_ngb = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], self.hidden_dim)))

    def message_self(self, X):
        return tf.matmul(X, self.message_kernel_self)

    def message_ngb(self, X):
        return tf.matmul(X, self.message_kernel_ngb)

    def call(self, inputs, **kwargs):
        X, A = inputs
        edgelist = A.indices
        x1 = self.message_self(X)
        x1 = tf.gather(x1, edgelist[:, 0])
        x1_att = tf.matmul(x1, self.kernel1)
        x2 = self.message_ngb(X)
        x2 = tf.gather(x2, edgelist[:, 1])
        x2_att = tf.matmul(x2, self.kernel2)
        att_score = x1_att + x2_att
        attention = unsorted_segment_softmax(att_score, edgelist[:, 1])
        if self.dropout_rate:
            attention = l.Dropout(self.dropout_rate)(attention)
        attended_embeddings = tf.gather(x1, edgelist[:, 0]) * attention
        update_embeddings = tf.math.unsorted_segment_sum(attended_embeddings, edgelist[:, 1], A.shape[-1])
        if self.dropout_rate:
            update_embeddings = l.Dropout(self.dropout_rate)(update_embeddings)
        return update_embeddings, A


class MultiHeadGAT(l.Layer):
    def __init__(self, heads, hidden_dim, **kwargs):
        super(MultiHeadGAT, self).__init__(**kwargs)
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.initializer = initializers.GlorotNormal()

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        self.attention_heads = []
        for _ in range(self.heads):
            self.attention_heads.append(GAT(self.hidden_dim))
        self.heads_kernel = tf.Variable(initial_value=self.initializer(shape=(self.heads, 1)))

    def call(self, inputs, **kwargs):
        heads_emb = []
        for i in range(self.heads):
            emb, A = self.attention_heads[i](inputs)
            emb = tf.expand_dims(emb, 0)
            heads_emb.append(emb)
        heads_emb = tf.concat(heads_emb, 0)
        heads_comb = activations.get("softmax")(self.heads_kernel)
        heads_comb = l.Dropout(0.5)(heads_comb)
        combined_heads = tf.einsum("hnd,hk->knd", heads_emb, heads_comb)
        combined_heads = tf.squeeze(combined_heads, 0)
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
        self.W1 = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], self.hidden_dim)))
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

    def __init__(self, hidden_dim, activation="relu", dropout_rate=0, layer=0, **kwargs):
        super(GCNDirected, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.initializer = initializers.GlorotNormal()
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.layer = layer

    def build(self, input_shape):
        X_shape, A_shape = input_shape
        self.kernel1 = tf.Variable(initial_value=self.initializer(shape=(X_shape[-1], self.hidden_dim)),
                                   name=f"kernel l{self.layer}")

    def call(self, inputs, **kwargs):
        X, A = inputs
        A = self.add_identity(A)
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
        # L = tf.constant(L)
        t = tf.matmul(X, self.kernel1)
        print(f"t {t}")
        Z = tf.matmul(L, t)
        print(f"Z {Z}")
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
