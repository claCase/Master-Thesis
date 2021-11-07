import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.keras import models as m
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import activations
from tensorflow.keras import initializers
import tensorflow.keras.backend as k
from src.modules import layers
import tensorflow_probability as tfp
from src.modules import losses
from src.modules.utils import sample_zero_edges, mask_sparse, predict_all_sparse, add_self_loop
from spektral.layers import GATConv, DiffPool


class GraphRNN(m.Model):
    def __init__(self, encoder, decoder, nodes_embedding_dim, **kwargs):
        super(GraphRNN, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.mu_activation = "tanh"
        self.sigma_activation = "relu"
        self.dropout_rate = kwargs.get("dropout_rate")
        self.embedding_size = kwargs.get("embedding_size")
        self.nodes_embedding_dim = nodes_embedding_dim

        if self.encoder == "NTN":
            self.encoder = layers.NTN

        # self.prior_mean =
        self.encoder_mean = self.encoder(10, self.mu_activation)
        self.encoder_sigma = self.encoder(10, self.sigma_activation)

    def call(self, inputs, **kwargs):
        X, R, A = inputs


class GAHRT_Outer_Probabilistic(m.Model):
    def __init__(self, nodes_features_dim, relations_features_dim, **kwargs):
        super(GAHRT_Outer_Probabilistic, self).__init__(**kwargs)
        self.encoder = layers.RGHAT(nodes_features_dim, relations_features_dim)
        self.decoder = layers.TrippletDecoder(False)

    def call(self, inputs, **kwargs):
        enc = self.encoder(inputs)
        dec_mu = activations.relu(self.decoder(enc))
        dec_sigma = activations.relu(self.decoder(enc)) + 1e-8
        return dec_mu, dec_sigma

    def sample(self, inputs):
        mu, sigma = self.call(inputs)
        logn = tfp.distributions.LogNormal(mu, sigma)
        return tf.squeeze(logn._sample_n(1))


class GAHRT_Outer_Deterministic(m.Model):
    def __init__(
            self, layers_nodes_features_dim, layers_relations_features_dim, **kwargs
    ):
        super(GAHRT_Outer_Deterministic, self).__init__(**kwargs)
        self.encoders = []
        self.layers_nodes_features_dim = layers_nodes_features_dim
        self.layers_relations_features_dim = layers_relations_features_dim
        assert len(self.layers_nodes_features_dim) == len(
            self.layers_relations_features_dim
        )
        for fn, fr in zip(
                self.layers_nodes_features_dim, self.layers_relations_features_dim
        ):
            self.encoders.append(layers.RGHAT(fn, fr))
        self.decoder = layers.TrippletDecoderOuterSparse(True)
        self.regularizer = layers.SparsityRegularizer(0.5)

    def call(self, inputs, **kwargs):
        for i in range(len(self.layers_nodes_features_dim)):
            inputs = self.encoders[i](inputs)
        print(inputs[-1].shape)
        print(inputs[-1].indices.shape)
        X, R, A = inputs
        A_dec = activations.softplus(self.decoder(inputs))
        print(A_dec.shape)
        A_dec = tf.sparse.SparseTensor(inputs[-1].indices, A_dec, inputs[-1].shape)
        self.regularizer(A_dec.values)
        return inputs[0], inputs[1], A_dec


class GAHRT_NTN_Deterministic(m.Model):
    def __init__(
            self, layers_nodes_features_dim, layers_relations_features_dim, **kwargs
    ):
        super(GAHRT_NTN_Deterministic, self).__init__(**kwargs)
        self.encoders = []
        self.layers_nodes_features_dim = layers_nodes_features_dim
        self.layers_relations_features_dim = layers_relations_features_dim
        for fn, fr in zip(
                self.layers_nodes_features_dim, self.layers_relations_features_dim
        ):
            self.encoders.append(layers.RGHAT(fn, fr))
        self.decoder = layers.NTN()
        self.regularizer = losses.SparsityRegularizerLayer(0.5)

    def call(self, inputs, **kwargs):
        for i in range(len(self.layers_nodes_features_dim)):
            inputs = self.encoders[i](inputs)
        dec_X, dec_A = self.decoder([inputs[0], inputs[2]])
        dec_A = activations.relu(dec_A.values)
        dec_A = tf.sparse.SparseTensor(inputs[-1].indices, dec_A.values)
        self.regularizer(dec_A.values)
        return dec_X, inputs[1], dec_A.values


class GAHRT_Tripplet_Falt_Deterministic(m.Model):
    def __init__(
            self, layers_nodes_features_dim, layers_relations_features_dim, **kwargs
    ):
        super(GAHRT_Tripplet_Falt_Deterministic, self).__init__(**kwargs)
        self.encoders = []
        self.layers_nodes_features_dim = layers_nodes_features_dim
        self.layers_relations_features_dim = layers_relations_features_dim
        assert len(self.layers_nodes_features_dim) == len(
            self.layers_relations_features_dim
        )
        for fn, fr in zip(
                self.layers_nodes_features_dim, self.layers_relations_features_dim
        ):
            self.encoders.append(layers.RGHAT(fn, fr))
        self.decoder = layers.TrippletScoreFlatSparse(2)
        self.regularizer = losses.SparsityRegularizerLayer(0.5)

    def call(self, inputs, **kwargs):
        for i in range(len(self.layers_nodes_features_dim)):
            inputs = self.encoders[i](inputs)
        X, R, A_dec = self.decoder(inputs)
        # A_dec = activations.softplus(A_dec.values)
        # self.regularizer(A_dec.values)
        return X, R, A_dec


class NTN(m.Model):
    def __init__(self, n_layers=1, activation="tanh", output_activation="relu"):
        super(NTN, self).__init__()
        self.n_layers = n_layers
        self.ntns = []
        for i in range(self.n_layers):
            self.ntns.append(layers.NTN(activation=activation, output_activation=output_activation))

    def call(self, inputs, **kwargs):
        for i in range(self.n_layers):
            inputs = self.ntns[i](inputs)
        return inputs


class TensorDecompositionModel(m.Model):
    def __init__(self, k, **kwargs):
        super(TensorDecompositionModel, self).__init__(**kwargs)
        self.k = k
        self.dec = layers.TensorDecompositionLayer(self.k)

    def call(self, inputs, **kwargs):
        return self.dec(inputs)


class Bilinear(m.Model):
    def __init__(self, hidden, activation="relu", use_mask=False, **kwargs):
        super(Bilinear, self).__init__(**kwargs)
        self.hidden = hidden
        self.use_mask = use_mask
        self.activation = activation

    def build(self, input_shape):
        self.bilinear = layers.Bilinear(self.hidden, self.activation)
        if self.use_mask:
            self.bilinear_mask = layers.Bilinear(self.hidden)
        # self.regularizer = losses.SparsityRegularizerLayer(0.5)

    def call(self, inputs, **kwargs):
        X, A = self.bilinear(inputs)
        if self.use_mask:
            X_mask, A_mask = self.bilinear_mask(inputs)
            A = tf.math.multiply(A, A_mask)
        A_flat = tf.reshape(A, [-1])
        # self.add_loss(self.regularizer(A_flat))
        return X, A


class BilinearSparse(m.Model):
    def __init__(self, hidden_dim, sparsity_rate=0, **kwargs):
        super(BilinearSparse, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.bilinear = layers.BilinearSparse(self.hidden_dim)
        self.bilinear_mask = layers.BilinearSparse(self.hidden_dim, "sigmoid")
        self.sparsity_rate = sparsity_rate
        self.regularizer = losses.SparsityRegularizerLayer(self.sparsity_rate)
        # self.max_thr = tf.Variable(initial_value=0.5, trainable=True)

    def call(self, inputs, **kwargs):
        X, A = self.bilinear(inputs)
        X_mask, A_mask = self.bilinear_mask(inputs)
        # max_thr = activations.get("sigmoid")(self.max_thr)
        pred = tf.math.multiply(A.values, A_mask.values)
        A_pred = tf.sparse.SparseTensor(inputs.indices, pred, inputs.shape)
        if self.sparsity_rate:
            self.add_loss(self.regularizer(A.values))
            self.add_loss(self.regularizer(mask.values))
        return A_mask, X_mask, A, X, A_pred


class GCN(m.Model):
    def __init__(self, hidden_dims, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.gcns = []
        for h in self.hidden_dims:
            self.gcns.append(layers.GCN(h))

    def call(self, inputs, **kwargs):
        for i in range(len(self.hidden_dims)):
            inputs = self.gcns[i](inputs)
        return inputs


class GCN_BIL(m.Model):
    def __init__(self, hidden_dims, regularize, dec_activation="sigmoid", **kwargs):
        super(GCN_BIL, self).__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.dec_activation = dec_activation
        self.enc = GCN(self.hidden_dims)
        self.dec = layers.BilinearDecoderDense(self.dec_activation)
        self.regularize = regularize
        if self.regularize:
            self.regularizer = losses.SparsityRegularizerLayer(0.5)

    def call(self, inputs, **kwargs):
        X, A = self.enc(inputs)
        X, A = self.dec([X, A])
        if self.regularize:
            self.add_loss(self.regularizer(A))
        return X, A


class GCN_Inner(m.Model):
    def __init__(self, hidden_dims, regularize, **kwargs):
        super(GCN_Inner, self).__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.enc = GCN(self.hidden_dims)
        self.dec = layers.InnerProductDenseDecoder()
        self.regularize = regularize
        if self.regularize:
            self.regularizer = losses.SparsityRegularizerLayer(0.5)

    def call(self, inputs, **kwargs):
        X, A = self.enc(inputs)
        X, A = self.dec([X, A])
        if self.regularize:
            self.add_loss(self.regularizer(A))
        return X, A


class GCN_Classifier(m.Model):
    def __init__(self, hidden_dims, **kwargs):
        super(GCN_Classifier, self).__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.enc = GCN(self.hidden_dims)
        self.classifier_layer = l.Dense(hidden_dims[-1], "softmax")

    def call(self, inputs, training=None, mask=None):
        X, A = self.enc(inputs)
        pred = self.classifier_layer(X)
        return pred


class GCNDirectedBIL(m.Model):
    def __init__(
            self, hidden_dims, sparse_regularize, emb_regularize, skip_connections, **kwargs
    ):
        super(GCNDirectedBIL, self).__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.encoders = []
        for i, h in enumerate(self.hidden_dims):
            self.encoders.append(layers.GCNDirected(self.hidden_dims[i], layer=i))
        self.dec = layers.BilinearDecoderDense()
        self.sparse_regularize = sparse_regularize
        self.emb_regularize = emb_regularize
        if self.sparse_regularize:
            self.sparse_regularizer = losses.SparsityRegularizerLayer(0.5)
        if self.emb_regularize:
            self.emb_regularizer = losses.embedding_smoothness
        self.skip_connections = skip_connections

    def call(self, inputs, **kwargs):
        for i in range(len(self.hidden_dims)):
            X, A = self.encoders[i](inputs)
            if self.skip_connections:
                if X.shape[-1] == inputs[0].shape[-1]:
                    inputs = [X + inputs[0], A]
                else:
                    raise Exception(
                        f"Skip Connection fails because of previous layer shape mismatch {X.shape[-1]} != {inputs[0].shape[-1]}"
                    )
            else:
                inputs = [X, A]
        X, A = self.dec(inputs)

        if self.sparse_regularize:
            self.add_loss(self.sparse_regularizer(A))
        if self.emb_regularize:
            self.add_loss(self.emb_regularizer(X, A))
        return X, A


class GCNDirectedClassifier(m.Model):
    def __init__(
            self, hidden_dims, sparse_regularize, emb_regularize, skip_connections, **kwargs
    ):
        super(GCNDirectedClassifier, self).__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.encoders = []
        for i, h in enumerate(self.hidden_dims):
            self.encoders.append(layers.GCNDirected(self.hidden_dims[i], layer=i))
        self.Classification_layer = l.Dense(self.hidden_dims[-1], activation="softmax")
        self.sparse_regularize = sparse_regularize
        self.emb_regularize = emb_regularize
        if self.sparse_regularize:
            self.sparse_regularizer = losses.SparsityRegularizerLayer(0.5)
        if self.emb_regularize:
            self.emb_regularizer = losses.embedding_smoothness
        self.skip_connections = skip_connections

    def call(self, inputs, **kwargs):
        for i in range(len(self.hidden_dims)):
            X, A = self.encoders[i](inputs)
            if self.skip_connections:
                if X.shape[-1] == inputs[0].shape[-1]:
                    inputs = [X + inputs[0], A]
                else:
                    raise Exception(
                        f"Skip Connection fails because of previous layer shape mismatch {X.shape[-1]} != {inputs[0].shape[-1]}"
                    )
            else:
                inputs = [X, A]
        pred = self.Classification_layer(inputs[0])

        if self.sparse_regularize:
            self.add_loss(self.sparse_regularizer(A))
        if self.emb_regularize:
            self.add_loss(self.emb_regularizer(X, A))
        return pred, A


class GAT(m.Model):
    def __init__(self, hidden_dims=(), **kwargs):
        super(GAT, self).__init__(**kwargs)
        self.gats = []
        for i in hidden_dims:
            self.gats.append(layers.GAT(i))

    def call(self, inputs, **kwargs):
        for i in range(len(self.gats)):
            inputs = self.gats[i](inputs)
        return inputs


class GAT_BIL(m.Model):
    def __init__(self, hidden_dims=(), return_attention=False, **kwargs):
        super(GAT_BIL, self).__init__(**kwargs)
        self.return_attention = return_attention
        if self.return_attention:
            self.attention = []
        self.gats = []
        for h in hidden_dims:
            self.gats.append(layers.GAT(h, return_attention=self.return_attention))
        self.decoder = layers.BilinearDecoderSparse()

    def call(self, inputs, **kwargs):
        for i in range(len(self.gats)):
            inputs = self.gats[i](inputs)
            if self.return_attention:
                self.attention.append(inputs[0])
                inputs = inputs[1:]
        pred = self.decoder(inputs)
        if self.return_attention:
            return [self.attention, *pred]
        else:
            return pred


class GAT_Inner(m.Model):
    def __init__(self, hidden_dims=(), return_attention=False, **kwargs):
        super(GAT_Inner, self).__init__(**kwargs)
        self.return_attention = return_attention
        if self.return_attention:
            self.attention = []
        self.gats = []
        for h in hidden_dims:
            self.gats.append(layers.GAT(h))
        self.decoder = layers.InnerProductSparseDecoder()

    def call(self, inputs, **kwargs):
        for i in range(len(self.gats)):
            inputs = self.gats[i](inputs)
        pred = self.decoder(inputs)
        return pred


class MultiHeadGAT_BIL(m.Model):
    def __init__(
            self,
            heads,
            hidden_dim,
            sparse_rate,
            embedding_smoothness_rate,
            return_attention=False,
            **kwargs,
    ):
        super(MultiHeadGAT_BIL, self).__init__(**kwargs)
        self.return_attention = return_attention
        if self.return_attention:
            self.attention = []
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.sparse_rate = sparse_rate
        self.embedding_smoothness_rate = embedding_smoothness_rate
        self.mgats = []
        for l_heads, l_hidden in zip(self.heads, self.hidden_dim):
            self.mgats.append(
                layers.MultiHeadGAT(
                    l_heads, l_hidden, return_attention=self.return_attention
                )
            )
        self.decoder = layers.BilinearDecoderSparse("softplus")

    def call(self, inputs, **kwargs):
        for i in range(len(self.hidden_dim)):
            if self.return_attention:
                inputs = self.mgats[i](inputs)
                self.attention.append(inputs[0])
                inputs = inputs[1:]
            else:
                inputs = self.mgats[i](inputs)
            if self.embedding_smoothness_rate:
                self.add_loss(
                    losses.EmbeddingSmoothnessRegularizer(
                        self.embedding_smoothness_rate
                    )(inputs)
                )
        pred = self.decoder(inputs)
        if self.sparse_rate:
            self.add_loss(losses.SparsityRegularizerLayer(self.sparse_rate)(pred[-1]))
        return pred


class MultiHeadGAT_Inner(m.Model):
    def __init__(
            self, heads, hidden_dim, sparse_rate, embedding_smoothness_rate, **kwargs
    ):
        super(MultiHeadGAT_Inner, self).__init__(**kwargs)
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.sparse_rate = sparse_rate
        self.embedding_smoothness_rate = embedding_smoothness_rate
        self.mgats = []
        for l_heads, l_hidden in zip(self.heads, self.hidden_dim):
            self.mgats.append(layers.MultiHeadGAT(l_heads, l_hidden))
        self.decoder = layers.InnerProductSparseDecoder()

    def call(self, inputs, **kwargs):
        for i in range(len(self.hidden_dim)):
            inputs = self.mgats[i](inputs)
            if self.embedding_smoothness_rate:
                self.add_loss(
                    losses.EmbeddingSmoothnessRegularizer(
                        self.embedding_smoothness_rate
                    )(inputs)
                )
        pred = self.decoder(inputs)
        if self.sparse_rate:
            self.add_loss(losses.SparsityRegularizerLayer(self.sparse_rate)(pred[-1]))
        return pred


class MultiHeadGAT_Flat(m.Model):
    def __init__(
            self, heads, hidden_dim, sparse_rate, embedding_smoothness_rate, **kwargs
    ):
        super(MultiHeadGAT_Flat, self).__init__(**kwargs)
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.sparse_rate = sparse_rate
        self.embedding_smoothness_rate = embedding_smoothness_rate
        self.mgats = []
        for l_heads, l_hidden in zip(self.heads, self.hidden_dim):
            self.mgats.append(layers.MultiHeadGAT(l_heads, l_hidden))
        self.decoder = layers.FlatDecoderSparse("softplus")

    def call(self, inputs, **kwargs):
        for i in range(len(self.hidden_dim)):
            inputs = self.mgats[i](inputs)
            if self.embedding_smoothness_rate:
                self.add_loss(
                    losses.EmbeddingSmoothnessRegularizer(
                        self.embedding_smoothness_rate
                    )(inputs)
                )
        pred = self.decoder(inputs)
        if self.sparse_rate:
            self.add_loss(losses.SparsityRegularizerLayer(self.sparse_rate)(pred[-1]))
        return pred


class MultiHeadGAT_Classifier(m.Model):
    def __init__(
            self, heads, hidden_dim, sparse_rate, embedding_smoothness_rate, **kwargs
    ):
        super(MultiHeadGAT_Classifier, self).__init__(**kwargs)
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.sparse_rate = sparse_rate
        self.embedding_smoothness_rate = embedding_smoothness_rate
        self.mgats = []
        for l_heads, l_hidden in zip(self.heads, self.hidden_dim):
            self.mgats.append(layers.MultiHeadGAT(l_heads, l_hidden))
        self.classifier_layer = l.Dense(hidden_dim[-1], activation="softmax")

    def call(self, inputs, **kwargs):
        for i in range(len(self.hidden_dim)):
            inputs = self.mgats[i](inputs)
            if self.embedding_smoothness_rate:
                self.add_loss(
                    losses.EmbeddingSmoothnessRegularizer(
                        self.embedding_smoothness_rate
                    )(inputs)
                )
        pred = self.classifier_layer(inputs[0])
        return pred


class GAT_Inner_spektral(m.Model):
    def __init__(self, **kwargs):
        super(GAT_Inner_spektral, self).__init__()
        self.gat = GATConv(**kwargs)
        self.bil_w = layers.InnerProductSparseDecoder("relu")
        self.bil_mask = layers.InnerProductSparseDecoder("sigmoid")

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        x = self.gat(inputs)
        _, a_w = self.bil_w([x, a])
        _, a_mask = self.bil_mask([x, a])
        a_final_values = tf.math.multiply(a_w.values, a_mask.values)
        a_final = tf.sparse.SparseTensor(a_w.indices, a_final_values, a_w.shape)
        return x, a_final


class GAT_Inner_spektral_dense(m.Model):
    def __init__(self, **kwargs):
        super(GAT_Inner_spektral_dense, self).__init__()
        self.gat = GATConv(**kwargs)
        self.bil_w = layers.InnerProductDenseDecoder("relu")
        self.bil_mask = layers.InnerProductDenseDecoder("sigmoid")

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        x = self.gat(inputs)
        _, a_w = self.bil_w([x, a])
        _, a_mask = self.bil_mask([x, a])
        a_final = tf.math.multiply(a_w, a_mask)
        return x, a_final


class GAT_BIL_spektral(m.Model):
    def __init__(self, channels=10,
                 attn_heads=10,
                 concat_heads=False,
                 dropout_rate=0.5,
                 output_activation="relu",
                 use_mask=False,
                 sparsity=0.,
                 **kwargs):
        super(GAT_BIL_spektral, self).__init__()

        self.channles = channels
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.use_mask = use_mask
        self.sparsity = sparsity

        self.gat = GATConv(channels=self.channles,
                           attn_heads=self.attn_heads,
                           concat_heads=self.concat_heads,
                           dropout_rate=self.dropout_rate,
                           add_self_loop=False,
                           **kwargs)
        self.bil_w = layers.BilinearDecoderSparse(self.output_activation)
        # self.bil_mask = layers.BilinearDecoderSparse("sigmoid")

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        x = self.gat(inputs)
        _, a_w = self.bil_w([x, a])
        # _, a_mask = self.bil_mask([x, a])
        # a_final_values = tf.math.multiply(a_w.values, a_mask.values)
        # a_final = tf.sparse.SparseTensor(a_w.indices, a_final_values, a_w.shape)
        return x, a_w


class GAT_BIL_spektral_dense(m.Model):
    def __init__(self, channels=10,
                 attn_heads=10,
                 concat_heads=True,
                 dropout_rate=0.60,
                 return_attn_coef=False,
                 output_activation="relu",
                 use_mask=False,
                 sparsity=0.,
                 **kwargs):
        super(GAT_BIL_spektral_dense, self).__init__()
        self.channles = channels
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.return_attn_coef = return_attn_coef,
        self.output_activation = output_activation
        self.use_mask = use_mask
        self.sparsity = sparsity

        self.gat = GATConv(channels=self.channles,
                           attn_heads=self.attn_heads,
                           concat_heads=self.concat_heads,
                           dropout_rate=self.dropout_rate,
                           add_self_loop=False,
                           return_attn_coef=self.return_attn_coef,
                           **kwargs)
        self.gat_bin = GATConv(channels=self.channles,
                           attn_heads=self.attn_heads,
                           concat_heads=self.concat_heads,
                           dropout_rate=self.dropout_rate,
                           add_self_loop=False,
                           return_attn_coef=self.return_attn_coef,
                           **kwargs)
        self.bil_w = layers.BilinearDecoderDense(activation=self.output_activation)
        if self.use_mask:
            self.bil_mask = layers.BilinearDecoderDense("sigmoid")

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        zero_diag = tf.linalg.diag(tf.ones(a.shape[0]))
        zero_diag = tf.ones(a.shape) - zero_diag
        if self.return_attn_coef:
            xw, attn = self.gat(inputs)
            xb, attnb = self.gat_bin(inputs)
        else:
            xw = self.gat(inputs)
        _, a_w = self.bil_w([xw, a])
        a_w *= zero_diag
        if self.sparsity:
            self.add_loss(losses.SparsityRegularizerLayer(self.sparsity)(a_w))
        if self.use_mask:
            _, a_mask = self.bil_mask([xb, a])
            a_mask *= zero_diag
            mask = tf.where(a_mask > 0.5, 1.0, 0.0)
            if self.sparsity:
                self.add_loss(losses.SparsityRegularizerLayer(self.sparsity)(a_mask))
            #a_final = tf.math.multiply(a_w, mask)
            if self.return_attn_coef:
                return xw, a_w, a_mask, attn
            else:
                return xw, a_w, a_mask
        else:
            if self.return_attn_coef:
                return xw, a_w, attn
            else:
                return xw, a_w


class GraphSamplerSparse(tf.keras.models.Model):
    def __init__(self, hidden_dim, temperature):
        super(GraphSamplerSparse, self).__init__()
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def build(self, input_shape):
        self.bil_mask = layers.BilinearSparse(self.hidden_dim, activation="sigmoid")
        self.bil_weight = layers.BilinearSparse(self.hidden_dim, activation="relu")

    def call(self, inputs):
        x, a = self.bil_mask(inputs)
        x_m, a_m = self.bil_weight(inputs)
        probs = tf.sparse.softmax(a_m)
        probs_dense = tf.sparse.to_dense(probs)
        # bernulli_sampler = tfp.distributions.RelaxedOneHotCategorical(probs=probs_dense, temperature=0.5)
        prior = tfp.distributions.Independent(
            tfp.distributions.RelaxedOneHotCategorical(0.1, 0.5 * tf.ones_like(inputs)))
        distr_layer = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda x: tfp.distributions.RelaxedOneHotCategorical(probs=x,
                                                                                      temperature=self.temperature),
            convert_to_tensor_fn=lambda x: x.sample(),
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1)
        )
        # bernulli_sampler = tfp.distributions.Bernoulli(probs=probs_dense)
        '''a_m_sample = bernulli_sampler.sample()
        a_m_sample = tf.cast(a_m_sample, tf.float32)'''
        # a_m_sample_sp = tf.sparse.from_dense(a_m_sample)
        a_m_sample = distr_layer(probs_dense)
        a_masked = a.__mul__(a_m_sample)
        return x, a, x_m, a_m, a_masked


class GraphSamplerDense(tf.keras.models.Model):
    def __init__(self, hidden_dim, temperature):
        super(GraphSamplerDense, self).__init__()
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def build(self, input_shape):
        self.bil_mask = layers.Bilinear(self.hidden_dim, activation="sigmoid")
        self.bil_weight = layers.Bilinear(self.hidden_dim, activation="relu")

    def call(self, inputs):
        x, a = self.bil_weight(inputs)
        x_m, a_m = self.bil_mask(inputs)
        probs = tf.nn.softmax(a_m)
        prior = tfp.distributions.Independent(
            tfp.distributions.RelaxedOneHotCategorical(0.1, 0.5 * tf.ones_like(inputs)))
        distr_layer = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda x: tfp.distributions.RelaxedOneHotCategorical(probs=x,
                                                                                      temperature=self.temperature),
            convert_to_tensor_fn=lambda x: x.sample(),
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1)
        )
        a_m_sample = distr_layer(probs)
        a_masked = tf.math.multiply(a, a_m_sample)
        return x, a, x_m, a_m, a_masked


class Relational_GAT_Bil(m.Model):
    def __init__(self, **kwargs):
        super(Relational_GAT_Bil, self).__init__()
        # self.


def grad_step(inputs, model, loss_fn, optimizer, loss_history=[]):
    with tf.GradientTape() as tape:
        pred = model(inputs)
        loss = loss_fn(inputs[-1], pred[-1])
        if model.losses:
            loss += model.losses
        loss_history.append(loss.numpy())
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return model, loss_history, pred


def slice_data(data_sp, t, X, sparse=True, r=None, log=False, simmetrize=False):
    A = tf.sparse.slice(
        data_sp, (t, 0, 0, 0), (1, data_sp.shape[1], data_sp.shape[2], data_sp.shape[3])
    )
    A = tf.sparse.to_dense(A)
    A = tf.squeeze(A, 0)
    if r is not None and isinstance(r, tuple):
        A = A[r[0]: r[1]]
    elif r is not None and isinstance(r, int):
        A = A[r]
    if log:
        A = tf.math.log(A)
        A = tf.clip_by_value(A, 0, 1e12)
    if sparse:
        A = tf.sparse.from_dense(A)
    if X:
        return [*X, A]
    else:
        return A


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pickle as pkl

    # from src.graph_data_loader import relational_graph_plotter
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-model")
    parser.add_argument("--dataset", default="comtrade")
    args = parser.parse_args()
    model = args.model
    dataset = args.dataset
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    CUDA_VISIBLE_DEVICES = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    """nodes = 174
    ft = 10
    r = 60
    """

    # R = tf.Variable(np.random.normal(size=(r, ft)), dtype=tf.float32)
    # X = tf.Variable(np.random.normal(size=(nodes, ft)), dtype=tf.float32)
    if dataset == "comtrade":
        with open(
                "A:\\Users\\Claudio\\Documents\\PROJECTS\\Master-Thesis\\Data\\complete_data_final_transformed_no_duplicate.pkl",
                "rb",
        ) as file:
            data_np = pkl.load(file)
        data_sp = tf.sparse.SparseTensor(
            data_np[:, :4], data_np[:, 4], np.max(data_np, 0)[:4]
        )
        data_sp = tf.sparse.reorder(data_sp)

        years = data_sp.shape[0]

        X = tf.eye(data_sp.shape[2])
        R = tf.eye(data_sp.shape[1])

    elif dataset == "cora":
        from spektral.datasets import Cora

        data = Cora()
        G = data.read()[0]
        A = G.a.tocoo()
        edgelist = np.concatenate([A.row[:, None], A.col[:, None]], -1)
        A = tf.sparse.SparseTensor(edgelist, A.data, A.shape)
        A = tf.sparse.reorder(A)
        X = G.x
        Y = G.y
        # R = np.eye(10)
    else:
        raise Exception("Dataset not available")

    if model == "gahrt_outer_det":
        model = GAHRT_Outer_Deterministic([5], [5])
        r = None
        sparse = True
        inputs = [X, R]
    elif model == "gahrt_outer_prob":
        model = GAHRT_Outer_Probabilistic([10], [10])
        r = None
        sparse = True
        inputs = [X, R]
    elif model == "gahrt_flat_det":
        model = GAHRT_Tripplet_Falt_Deterministic([10], [10])
        r = None
        sparse = True
        inputs = [X, R]
    elif model == "ntn":
        model = NTN(5)
        r = None
        sparse = True
        inputs = [X, R]
    elif model == "bilinear":
        model = Bilinear(10)
        r = 10
        sparse = False
        inputs = [X]
    elif model == "gcn":
        model = GCN([5, 5, 2])
        sparse = False
        r = 10
        inputs = [X]
    elif model == "gat_bilinear":
        model = GAT_BIL([25, 25, 15])
        sparse = True
        r = 10
        inputs = [X]
    elif model == "gat_inner":
        model = GAT_Inner([5, 5, 2])
        sparse = True
        r = 10
        inputs = [X]
    elif model == "mgat_bilinear":
        model = MultiHeadGAT_BIL([5, 4, 3], [25, 25, 20], False, False)
        sparse = True
        r = 10
        inputs = [X]
    elif model == "gat_bil_spk":
        model = GAT_BIL_spektral(channels=10,
                                 attn_heads=10,
                                 concat_heads=False,
                                 dropout_rate=0.5)
        sparse = True
        r = 10
        inputs = [X]
    elif model == "mgat_inner":
        model = MultiHeadGAT_Inner([5, 4, 3], [25, 25, 20], False, False)
        sparse = True
        r = 10
        inputs = [X]
    elif model == "gcn_bilinear":
        model = GCN_BIL([25, 25, 20], False)
        sparse = False
        r = 10
        inputs = [X]
    elif model == "gcn_inner":
        model = GCN_Inner([5, 5, 2], False)
        sparse = False
        r = 10
        inputs = [X]
    elif model == "gcn_directed_bilinear":
        model = GCNDirectedBIL([25, 30, 15, 15], False, False, False)
        sparse = False
        r = 10
        inputs = [X]
    elif model == "gcn_directed_classifier":
        model = GCNDirectedClassifier([25, 25, 20], False, False, False)
        sparse = False
    elif model == "bilinear_sparse":
        print("Bilinear model")
        model = BilinearSparse(30, 0.01)
        sparse = True
        inputs = []
        r = 10

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_epochs_history = []
    epochs = 150
    years = 1
    batches = 1
    t = 50
    """train_size = .7 * len(A.indices)
    idx_train = np.random.choice(np.arange(len(A.indices)), size=int(train_size), replace=False)
    train_indices = tf.gather(A.indices, idx_train)
    train_values = tf.gather(A.values, idx_train)
    A_train = tf.sparse.SparseTensor(train_indices, train_values, A.shape)
    inputs_t = [*inputs, A]
    """
    if dataset == "comtrade":
        print("Comtrade Dataset")
        inputs_t = slice_data(
            data_sp, t, inputs, sparse=sparse, r=r, log=True, simmetrize=True
        )
    else:
        if not sparse:
            A = tf.sparse.to_dense(A)
            inputs_t = [X, A]
    for e in range(epochs):
        # selected_years = np.random.choice(np.arange(years), size=years, replace=False)
        loss_batch_history = []
        batch_grads = []
        for batch in range(batches):
            """noise = np.random.lognormal(0, 1, size=(len(inputs_t[-1].values)))
            vals = inputs_t[-1].values
            vals += noise
            inputs_t[-1] = tf.sparse.SparseTensor(inputs_t[-1].indices, vals, inputs_t[-1].shape)

            A = mask_sparse(A)
            inputs_t = [X, A]"""
            # model, loss, outputs = grad_step(inputs_t, model, losses.square_loss, optimizer, [])
            if sparse:
                if model == "bilinear_sparse":
                    A_true = sample_zero_edges(inputs_t, 0.4)
                    # A_true = add_self_loop(A_true)
                else:
                    A_true = inputs_t[-1]
                    # A_true = sample_zero_edges(inputs_t[-1], .4)
                    # A_true = add_self_loop(A_true)
                # A_true_binary = tf.sparse.SparseTensor(A_true.indices, tf.math.maximum(A_true.values, 0.), A_true.shape)
                # print("Sampling Edges")
            with tf.GradientTape() as tape:
                if model == "bilinear_sparse":
                    A_pred = model(A_true)  #
                else:
                    if not sparse:
                        A_true = inputs_t[-1]
                        A_true = A_true * np.random.choice((0, 1), size=A_true.shape)
                    if len(inputs_t) == 3:
                        i = [X, R, A_true]
                    else:
                        i = [X, A_true]
                    pred = model(i)

                if dataset == "comtrade":
                    if model == "bilinear_sparse":
                        loss = losses.square_loss(A_true, A_pred[-1])
                        # loss += losses.square_loss(A_true_binary, A_pred[0])
                        # loss += losses.square_loss(inputs_t[0], pred[0])
                    else:
                        loss = losses.square_loss(A_true, pred[-1])
                elif dataset == "cora":
                    loss = categorical_crossentropy(Y, A_pred[0])
                    loss = tf.reduce_sum(loss, 0)
                if model.losses:
                    loss += model.losses
                loss_batch_history.append(loss.numpy())
            grads = tape.gradient(loss, model.trainable_weights)
            if batch_grads:
                sum_grads = []
                prev_grads = batch_grads.pop()
                for g, g_prev in zip(grads, prev_grads):
                    sum = tf.reduce_sum([g, g_prev], 0)
                    sum_grads.append(sum)
                batch_grads.append(sum_grads)
            else:
                batch_grads.append(grads)
            batch_grads.append(grads)
            print(f"epoch {e} batch {batch} | loss: {loss}")
        optimizer.apply_gradients(zip(batch_grads[0], model.trainable_weights))
        avg = np.sum(loss_batch_history)
        loss_epochs_history.append(avg)

    if model != "bilinear_sparse":
        A_true = inputs_t[-1]
    """if sparse:
        A_pred = predict_all_sparse(A_true)
    """
    if model == "bilinear_sparse":
        mask, A, X, A_pred = model(A_pred)
        X = X.numpy()
        A = tf.sparse.to_dense(tf.sparse.reorder(A))
        mask = tf.sparse.to_dense(tf.sparse.reorder(mask))
        A_pred = tf.sparse.to_dense(tf.sparse.reorder(A_pred))

        plt.subplot(2, 2, 1)
        plt.title("Predicted Weighted")
        plt.imshow(A_pred)
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.title("True")
        plt.imshow(A_true)
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.title("A weighted")
        plt.imshow(A)
        plt.colorbar()
        plt.subplot(2, 2, 4)
        plt.title("A mask")
        plt.imshow(mask)
        plt.colorbar()
        with open(
                "A:/Users/Claudio/Documents/PROJECTS/Master-Thesis/Data/idx_to_countries.pkl",
                "rb",
        ) as file:
            idx2country = pkl.load(file)
        with open(
                "A:/Users/Claudio/Documents/PROJECTS/Master-Thesis/Data/iso3_to_name.pkl",
                "rb",
        ) as file:
            iso3_to_name = pkl.load(file)

        iso3 = [idx2country[i] for i in range(len(X))]
        names = [iso3_to_name[int(i)] for i in iso3]
        from sklearn.manifold import TSNE

        X = TSNE(n_components=2).fit_transform(X)
        fig = plt.figure()
        ax = fig.add_subplot()  # projection="3d")
        ax.scatter(X[:, 0], X[:, 1])  # , X[:, 2])
        for c, (x, y) in zip(names, X[:, :2]):
            ax.text(x, y, f"{c}", size=7)
        plt.show()

    if len(inputs_t) == 2:
        if sparse:
            X_pred, A_pred = model([X, A_true])
            A_pred = tf.sparse.reorder(A_pred)
            A_pred = tf.sparse.to_dense(A_pred)
            A_true = tf.sparse.to_dense(A_true)
    elif len(inputs_t) == 3:
        X, R, A = inputs_t
        A = tf.sparse.to_dense(tf.sparse.reorder(A))
        X_pred, R_pred, A_pred = model([X, R, A_pred])
        A_pred = tf.sparse.to_dense(A_pred)
    plt.figure()
    plt.title("Embeddings")
    plt.scatter(X_pred[:, 0], X_pred[:, 1])
    """if isinstance(A, tf.sparse.SparseTensor):
        A = tf.sparse.to_dense(A)"""
    """if isinstance(A_pred, tf.sparse.SparseTensor):
        A_pred = tf.sparse.to_dense(A_pred)"""
    if len(A_pred.shape) == 2:
        plt.figure()
        plt.title("True Adj")
        plt.imshow(A_true)
        plt.colorbar()
        plt.figure()
        plt.title("Pred Adj")
        plt.imshow(A_pred)
        plt.colorbar()
        try:
            w = model.decoder.trainable_weights[0]
            plt.figure()
            plt.title("Bilinear Matrix")
            plt.imshow(w.numpy())
            plt.colorbar()
        except:
            pass
    elif len(A_pred.shape) == 3:
        r = A_pred.shape[0]
        r = r if r < 10 else 10
        f, axes = plt.subplots(r, 3)
        for i in range(r):
            axes[i, 0].imshow(A[i, :, :], cmap="winter")
            axes[i, 1].imshow(A_pred[i, :, :], cmap="winter")
            img = axes[i, 2].imshow(A[i, :, :] - A_pred[i, :, :], cmap="winter")
            f.colorbar(img, ax=axes[i, 2])
        axes[0, 0].title.set_text("True Values")
        axes[0, 1].title.set_text("Predicted Values")
        axes[0, 2].title.set_text("Difference")
        from sklearn.manifold import TSNE

        X = TSNE(n_components=2).fit_transform(X_pred)
        R = TSNE(n_components=2).fit_transform(R_pred)
        plt.figure()
        plt.title("Node Embeddings")
        plt.scatter(X[:, 0], X[:, 1])
        plt.figure()
        plt.title("Product Embeddings")
        plt.scatter(R[:, 0], R[:, 1])
    plt.figure()
    plt.title("Losses")
    """loss_k = []
    for e in range(epochs):
        t_loss = np.empty(years)
        for k in range(years):
            val = loss_history[e][k]
            # print(val)
            t_loss[k] = val
        loss_k.append(t_loss)"""
    plt.plot(loss_epochs_history)
    plt.show()
