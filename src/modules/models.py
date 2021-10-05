import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.keras import models as m
from tensorflow.keras import activations
from tensorflow.keras import initializers
import tensorflow.keras.backend as k
from src.modules import layers
import tensorflow_probability as tfp
from src.modules import losses
from src.modules.graph_utils import sample_zero_edges, mask_sparse


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


class GAHRT_Model_Probabilistic(m.Model):
    def __init__(self, nodes_features_dim, relations_features_dim, **kwargs):
        super(GAHRT_Model_Probabilistic, self).__init__(**kwargs)
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


class GAHRTTrippleDeterministic(m.Model):
    def __init__(self, layers_nodes_features_dim, layers_relations_features_dim, **kwargs):
        super(GAHRTTrippleDeterministic, self).__init__(**kwargs)
        self.encoders = []
        self.layers_nodes_features_dim = layers_nodes_features_dim
        self.layers_relations_features_dim = layers_relations_features_dim
        assert len(self.layers_nodes_features_dim) == len(self.layers_relations_features_dim)
        for fn, fr in zip(self.layers_nodes_features_dim, self.layers_relations_features_dim):
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
    def __init__(self, layers_nodes_features_dim, layers_relations_features_dim, **kwargs):
        super(GAHRT_NTN_Deterministic, self).__init__(**kwargs)
        self.encoders = []
        self.layers_nodes_features_dim = layers_nodes_features_dim
        self.layers_relations_features_dim = layers_relations_features_dim
        for fn, fr in zip(self.layers_nodes_features_dim, self.layers_relations_features_dim):
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


class NTN_Model(m.Model):
    def __init__(self, n_layers, **kwargs):
        super(NTN_Model, self).__init__(**kwargs)
        self.ntn = layers.NTN()
        self.n_layers = n_layers

    def call(self, inputs, **kwargs):
        # for n in range(self.n_layers):
        out = self.ntn(inputs)
        return out


class TensorDecompositionModel(m.Model):
    def __init__(self, k, **kwargs):
        super(TensorDecompositionModel, self).__init__(**kwargs)
        self.k = k
        self.dec = layers.TensorDecompositionLayer(self.k)

    def call(self, inputs, **kwargs):
        return self.dec(inputs)


class Bilinear(m.Model):
    def __init__(self, hidden, **kwargs):
        super(Bilinear, self).__init__(**kwargs)
        self.hidden = hidden

    def build(self, input_shape):
        self.bilinear = layers.Bilinear(self.hidden)
        self.regularizer = losses.SparsityRegularizerLayer(0.5)

    def call(self, inputs, **kwargs):
        out = self.bilinear(inputs)
        out_flat = tf.reshape(out, [-1])
        self.add_loss(self.regularizer(out_flat))
        return out


class BilinearSparse(m.Model):
    def __init__(self, hidden_dim, **kwargs):
        super(BilinearSparse, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.bilinear = layers.BilinearSparse(self.hidden_dim)
        self.regularizer = losses.SparsityRegularizerLayer(0.5)

    def call(self, inputs, **kwargs):
        out = self.bilinear(inputs)
        self.add_loss(self.regularizer(out.values))
        return out


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
    def __init__(self, hidden_dims, regularize, **kwargs):
        super(GCN_BIL, self).__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.enc = GCN(self.hidden_dims)
        self.dec = layers.BilinearDecoderDense()
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


class GCNDirectedBIL(m.Model):
    def __init__(self, hidden_dims, sparse_regularize, emb_regularize, skip_connections, **kwargs):
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
                        f"Skip Connection fails because of previous layer shape mismatch {X.shape[-1]} != {inputs[0].shape[-1]}")
            else:
                inputs = [X, A]
        X, A = self.dec(inputs)

        if self.sparse_regularize:
            self.add_loss(self.sparse_regularizer(A))
        if self.emb_regularize:
            self.add_loss(self.emb_regularizer(X, A))
        return X, A


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
    def __init__(self, hidden_dims=(), **kwargs):
        super(GAT_BIL, self).__init__(**kwargs)
        self.gats = []
        for h in hidden_dims:
            self.gats.append(layers.GAT(h))
        self.decoder = layers.BilinearDecoderSparse()

    def call(self, inputs, **kwargs):
        for i in range(len(self.gats)):
            inputs = self.gats[i](inputs)
        pred = self.decoder(inputs)
        return pred


class GAT_Inner(m.Model):
    def __init__(self, hidden_dims=(), **kwargs):
        super(GAT_Inner, self).__init__(**kwargs)
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
    def __init__(self, heads, hidden_dim, sparse_rate, embedding_smoothness_rate, **kwargs):
        super(MultiHeadGAT_BIL, self).__init__(**kwargs)
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.sparse_rate = sparse_rate
        self.embedding_smoothness_rate = embedding_smoothness_rate
        self.mgats = []
        for l_heads, l_hidden in zip(self.heads, self.hidden_dim):
            self.mgats.append(layers.MultiHeadGAT(l_heads, l_hidden))
        self.decoder = layers.BilinearDecoderSparse()

    def call(self, inputs, **kwargs):
        for i in range(len(self.hidden_dim)):
            inputs = self.mgats[i](inputs)
            if self.embedding_smoothness_rate:
                self.add_loss(losses.EmbeddingSmoothnessRegularizer(self.embedding_smoothness_rate)(inputs))
        pred = self.decoder(inputs)
        if self.sparse_rate:
            self.add_loss(losses.SparsityRegularizerLayer(self.sparse_rate)(pred[-1]))
        return pred


class MultiHeadGAT_Inner(m.Model):
    def __init__(self, heads, hidden_dim, sparse_rate, embedding_smoothness_rate, **kwargs):
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
                self.add_loss(losses.EmbeddingSmoothnessRegularizer(self.embedding_smoothness_rate)(inputs))
        pred = self.decoder(inputs)
        if self.sparse_rate:
            self.add_loss(losses.SparsityRegularizerLayer(self.sparse_rate)(pred[-1]))
        return pred


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





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pickle as pkl
    from src.graph_data_loader import relational_graph_plotter
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-model")
    args = parser.parse_args()
    model = args.model

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # CUDA_VISIBLE_DEVICES = ""
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    '''nodes = 174
    ft = 10
    r = 60
    '''

    # R = tf.Variable(np.random.normal(size=(r, ft)), dtype=tf.float32)
    # X = tf.Variable(np.random.normal(size=(nodes, ft)), dtype=tf.float32)
    with open(
            "A:\\Users\\Claudio\\Documents\\PROJECTS\\Master-Thesis\\Data\\complete_data_final_transformed_no_duplicate.pkl",
            "rb") as file:
        data_np = pkl.load(file)
    data_sp = tf.sparse.SparseTensor(data_np[:, :4], data_np[:, 4], np.max(data_np, 0)[:4])
    data_sp = tf.sparse.reorder(data_sp)

    years = data_sp.shape[0]


    X = tf.eye(data_sp.shape[2])
    R = tf.eye(data_sp.shape[1])
    if model == "gahrt_tripplet_det":
        model = GAHRTTrippleDeterministic([5], [5])
        r = None
        sparse = True
        inputs = [X, R]
    elif model == "gahrt_tripplet_prob":
        model = GAHRT_Model_Probabilistic(10, 10)
        r = None
        sparse = True
        inputs = [X, R]
    elif model == "ntn":
        model = NTN_Model(5)
        r = None
        sparse = True
        inputs = [X, R]
    elif model == "bilinear":
        model = Bilinear(10)
        r = 10
        sparse = False
        inputs = [X]
    elif model == "bilinear_sparse":
        model = BilinearSparse(10)
        r = 10
        sparse = True
        inputs = [X]
    elif model == "gcn":
        model = GCN([5, 5, 2])
        sparse = False
        r = 10
        inputs = [X]
    elif model == "gat_bilinear":
        model = GAT_BIL([5, 5, 5])
        sparse = True
        r = 10
        inputs = [X]
    elif model == "gat_inner":
        model = GAT_Inner([5, 5, 2])
        sparse = True
        r = 10
        inputs = [X]
    elif model == "mgat_bilinear":
        model = MultiHeadGAT_BIL([5, 4, 3], [5, 5, 2], False, False)
        sparse = True
        r = 10
        inputs = [X]
    elif model == "mgat_inner":
        model = MultiHeadGAT_Inner([5, 4, 3], [5, 5, 2], False, False)
        sparse = True
        r = 10
        inputs = [X]
    elif model == "gcn_bilinear":
        model = GCN_BIL([5, 5, 2], False)
        sparse = False
        r = 10
        inputs = [X]
    elif model == "gcn_inner":
        model = GCN_Inner([5, 5, 2], False)
        sparse = False
        r = 10
        inputs = [X]
    elif model == "gcn_directed_bil":
        model = GCNDirectedBIL([5], False, False, False)
        sparse = False
        r = 10
        inputs = [X]


    def slice_data(data_sp, t, X, sparse=True, r=None, log=False, simmetrize=False):
        A = tf.sparse.slice(data_sp, (t, 0, 0, 0), (1, data_sp.shape[1], data_sp.shape[2], data_sp.shape[3]))
        A = tf.sparse.to_dense(A)
        A = tf.squeeze(A, 0)
        if r is not None and isinstance(r, tuple):
            A = A[r[0]:r[1]]
        elif r is not None and isinstance(r, int):
            A = A[r]
        if log:
            A = tf.math.log(A)
            A = tf.clip_by_value(A, 0, 1e12)
        if sparse:
            A = tf.sparse.from_dense(A)

        return [*X, A]


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_epochs_history = []
    epochs = 30
    years = 1
    batches = 30
    t = 10
    grad_step = 0.5
    for e in range(epochs):
        selected_years = np.random.choice(np.arange(years), size=years, replace=False)
        loss_batch_history = []
        batch_grads = []
        for batch in range(batches):
            inputs_t = slice_data(data_sp, t, inputs, sparse=sparse, r=r, log=True, simmetrize=True)
            '''noise = np.random.lognormal(0, 1, size=(len(inputs_t[-1].values)))
            vals = inputs_t[-1].values
            vals += noise
            inputs_t[-1] = tf.sparse.SparseTensor(inputs_t[-1].indices, vals, inputs_t[-1].shape)
            
            A = mask_sparse(A)
            inputs_t = [X, A]'''
            #model, loss, outputs = grad_step(inputs_t, model, losses.square_loss, optimizer, [])
            inputs_t[-1] = sample_zero_edges(inputs_t[-1], .2)
            with tf.GradientTape() as tape:
                pred = model(inputs_t)
                loss = losses.square_loss(inputs_t[-1], pred[-1])
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

    X, A = inputs_t
    X_pred, A_pred = pred
    print(X_pred.shape)
    plt.figure()
    plt.title("Embeddings")
    plt.scatter(X_pred[:, 0], X_pred[:, 1])
    if isinstance(A, tf.sparse.SparseTensor):
        A = tf.sparse.to_dense(A)
    if isinstance(A_pred, tf.sparse.SparseTensor):
        A_pred = tf.sparse.to_dense(A_pred)
    plt.figure()
    plt.title("True Adj")
    plt.imshow(A)
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
    plt.figure()
    plt.title("Losses")
    '''loss_k = []
    for e in range(epochs):
        t_loss = np.empty(years)
        for k in range(years):
            val = loss_history[e][k]
            # print(val)
            t_loss[k] = val
        loss_k.append(t_loss)'''
    plt.plot(loss_epochs_history)
    '''X, R, A = model(inputs)
    plt.figure()
    plt.scatter(X[:,0], X[:,1])
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(A)
    plt.subplot(1,2,2)
    plt.imshow(inputs[-1])'''
    plt.show()

    '''A_dense = tf.sparse.to_dense(A)
    # A_pred_dense = tf.sparse.to_dense(tf.sparse.SparseTensor(A.indices, output, A.shape))
    # pred_prob = model_det.sample([X, R, A])
    pred_det = model_det([X, R, A])
    A_pred_dense = tf.sparse.to_dense(tf.sparse.SparseTensor(A.indices, pred_det, A.shape))
    f, axes = plt.subplots(A.shape[0], 3)
    for i in range(A.shape[0]):
        axes[i, 0].imshow(A_dense[i, :, :], cmap="winter")
        axes[i, 1].imshow(A_pred_dense[i, :, :], cmap="winter")
        img = axes[i, 2].imshow(A_dense[i, :, :] - A_pred_dense[i, :, :], cmap="winter")
        f.colorbar(img, ax=axes[i, 2])
    axes[0, 0].title.set_text("True Values")
    axes[0, 1].title.set_text("Predicted Values")
    axes[0, 2].title.set_text("Difference")
    plt.tight_layout()
    plt.show()
    '''
