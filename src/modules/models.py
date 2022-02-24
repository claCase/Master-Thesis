import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.keras import models as m
from tensorflow.keras.losses import categorical_crossentropy
from src.modules import layers
import tensorflow_probability as tfp
from src.modules import losses
from src.modules.utils import (
    sample_zero_edges,
    mask_sparse,
    predict_all_sparse,
    add_self_loop,
)
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, _config_for_enable_caching_device, \
    _caching_device
from spektral.layers import GATConv, GCNConv
from spektral.utils.convolution import gcn_filter
from src.modules.layers import BatchBilinearDecoderDense, GCNDirected
from src.modules.losses import TemporalSmoothness
from tensorflow import keras


class NestedCell(DropoutRNNCellMixin, keras.layers.Layer):
    def __init__(self, nodes, features, channels, attention_heads, hidden_size, dropout, recurrent_dropout, **kwargs):
        super(NestedCell, self).__init__(**kwargs)
        self.tot_nodes = nodes
        self.nodes_features = features
        self.hidden_size_in = channels * attention_heads
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.channels = channels
        self.state_size = [tf.TensorShape((self.tot_nodes, self.hidden_size)),
                           tf.TensorShape((self.tot_nodes, self.hidden_size)),
                           tf.TensorShape((self.tot_nodes, self.hidden_size))]
        self.output_size = [tf.TensorShape((self.tot_nodes, self.tot_nodes, 3))]
        self.gnn_u = GATConv(channels=self.channels, attn_heads=self.attention_heads, concat_heads=True, dropout_rate=0)
        self.decoder_mu = BatchBilinearDecoderDense(activation=None)
        self.decoder_sigma = BatchBilinearDecoderDense(activation=None)
        self.decoder_p = BatchBilinearDecoderDense(activation=None)
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self._enable_caching_device = kwargs.pop('enable_caching_device', True)

    def build(self, input_shapes):
        default_caching_device = _caching_device(self)
        self.b_u_p = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_u_p",
                                     caching_device=default_caching_device)
        self.b_r_p = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_r_p",
                                     caching_device=default_caching_device)
        self.b_c_p = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_c_p",
                                     caching_device=default_caching_device)
        self.W_u_p = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                     initializer="glorot_normal", name="W_u_p", caching_device=default_caching_device)
        self.W_r_p = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                     initializer="glorot_normal", name="W_r_p", caching_device=default_caching_device)
        self.W_c_p = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                     initializer="glorot_normal", name="W_c_p", caching_device=default_caching_device)

        self.b_u_mu = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_u_mu",
                                      caching_device=default_caching_device)
        self.b_r_mu = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_r_mu",
                                      caching_device=default_caching_device)
        self.b_c_mu = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_c_mu",
                                      caching_device=default_caching_device)
        self.W_u_mu = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                      initializer="glorot_normal", name="W_u_mu", caching_device=default_caching_device)
        self.W_r_mu = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                      initializer="glorot_normal", name="W_r_mu", caching_device=default_caching_device)
        self.W_c_mu = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                      initializer="glorot_normal", name="W_c_mu_sigma",
                                      caching_device=default_caching_device)

        self.b_u_sigma = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_u_sigma",
                                         caching_device=default_caching_device)
        self.b_r_sigma = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_r_sigma",
                                         caching_device=default_caching_device)
        self.b_c_sigma = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_c_sigma",
                                         caching_device=default_caching_device)
        self.W_u_sigma = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                         initializer="glorot_normal", name="W_u_sigma",
                                         caching_device=default_caching_device)
        self.W_r_sigma = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                         initializer="glorot_normal", name="W_r_sigma",
                                         caching_device=default_caching_device)
        self.W_c_sigma = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                         initializer="glorot_normal", name="W_c_sigma",
                                         caching_device=default_caching_device)

    def call(self, inputs, states, training):
        x, a = tf.nest.flatten(inputs)
        h_p, h_mu, h_sigma = tf.nest.flatten(states)
        if 0 < self.dropout < 1:
            inputs_mask = self.get_dropout_mask_for_cell(inputs=a, training=training, count=1)
            a = a * inputs_mask
        if 0 < self.recurrent_dropout < 1:
            h_mask = self.get_recurrent_dropout_mask_for_cell(inputs=states, training=training, count=1)
            h_p = h_p * h_mask[0]
            h_mu = h_mu * h_mask[1]
            h_sigma = h_sigma * h_mask[2]
        conv_u = self.gnn_u([x, a], training=training)  # B x N x d
        conv_r = conv_u
        conv_c = conv_u
        # Recurrence
        u_mu = tf.nn.sigmoid(self.b_u_mu + tf.concat([conv_u, h_mu], -1) @ self.W_u_mu)
        r_mu = tf.nn.sigmoid(self.b_r_mu + tf.concat([conv_r, h_mu], -1) @ self.W_r_mu)
        c_mu = tf.nn.tanh(self.b_c_mu + tf.concat([conv_c, r_mu * h_mu], -1) @ self.W_c_mu)
        h_prime_mu = u_mu * h_mu + (1 - u_mu) * c_mu

        u_sigma = tf.nn.sigmoid(self.b_u_sigma + tf.concat([conv_u, h_sigma], -1) @ self.W_u_sigma)
        r_sigma = tf.nn.sigmoid(self.b_r_sigma + tf.concat([conv_r, h_sigma], -1) @ self.W_r_sigma)
        c_sigma = tf.nn.tanh(self.b_c_sigma + tf.concat([conv_c, r_sigma * h_sigma], -1) @ self.W_c_sigma)
        h_prime_sigma = u_sigma * h_sigma + (1 - u_sigma) * c_sigma

        u_p = tf.nn.sigmoid(self.b_u_p + tf.concat([conv_u, h_p], -1) @ self.W_u_p)
        r_p = tf.nn.sigmoid(self.b_r_p + tf.concat([conv_r, h_p], -1) @ self.W_r_p)
        c_p = tf.nn.tanh(self.b_c_p + tf.concat([conv_c, r_p * h_p], -1) @ self.W_c_p)
        h_prime_p = u_p * h_p + (1 - u_p) * c_p

        p = self.decoder_p(h_prime_p)
        p = tf.expand_dims(p, -1)
        mu = self.decoder_mu(h_prime_mu)
        mu = tf.expand_dims(mu, -1)
        sigma = self.decoder_sigma(h_prime_sigma)
        sigma = tf.expand_dims(sigma, -1)
        logits = tf.concat([p, mu, sigma], axis=-1)
        return logits, (h_prime_p, h_prime_mu, h_prime_sigma)

    def get_config(self):
        config = {"nodes": self.tot_nodes,
                  "nodes_features": self.nodes_features,
                  "hidden_size": self.hidden_size,
                  "attention_heads": self.attention_heads,
                  "channels": self.channels,
                  "dropout": self.dropout,
                  "recurrent_dropout": self.recurrent_dropout}
        config.update(_config_for_enable_caching_device(self))
        base_config = super(NestedCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NestedCell2(keras.layers.Layer):
    def __init__(self, nodes, features, channels, attention_heads, hidden_size, dropout_rate=0.2, **kwargs):
        super(NestedCell2, self).__init__(**kwargs)
        self.tot_nodes = nodes
        self.nodes_features = features
        self.hidden_size_in = channels * attention_heads
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.channels = channels
        self.state_size = [tf.TensorShape((self.tot_nodes, self.hidden_size))]
        self.output_size = [tf.TensorShape((self.tot_nodes, self.tot_nodes, 3))]
        self.gnn = GATConv(channels=self.channels, attn_heads=self.attention_heads, concat_heads=True)
        self.decoder_mu = BatchBilinearDecoderDense(activation=None)
        self.decoder_sigma = BatchBilinearDecoderDense(activation=None)
        self.decoder_p = BatchBilinearDecoderDense(activation=None)
        self.drop = l.Dropout(dropout_rate)

    def build(self, input_shapes):
        self.b_u = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_u_sigma")
        self.b_r = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_r_sigma")
        self.b_c = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_c_sigma")
        self.W_u = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                   initializer="glorot_normal", name="W_u_sigma")
        self.W_r = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                   initializer="glorot_normal", name="W_r_sigma")
        self.W_c = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                   initializer="glorot_normal", name="W_c_sigma")

    def call(self, inputs, states, training):
        h = states[0]
        conv_u = self.gnn(inputs, training=training)  # B x N x d
        conv_r = conv_u
        conv_c = conv_u
        # Recurrence
        u = tf.nn.sigmoid(self.b_u + tf.concat([conv_u, h], -1) @ self.W_u)
        r = tf.nn.sigmoid(self.b_r + tf.concat([conv_r, h], -1) @ self.W_r)
        c = tf.nn.tanh(self.b_c + tf.concat([conv_c, r * h], -1) @ self.W_c)
        h_prime = u * h + (1 - u) * c
        h_prime = self.drop(h_prime, training=training)

        p = self.decoder_p(h_prime)
        p = tf.expand_dims(p, -1)
        mu = self.decoder_mu(h_prime)
        mu = tf.expand_dims(mu, -1)
        sigma = self.decoder_sigma(h_prime)
        sigma = tf.expand_dims(sigma, -1)
        logits = tf.concat([p, mu, sigma], axis=-1)
        return logits, h_prime

    def get_config(self):
        return {"hidden": self.hidden_size, "nodes": self.tot_nodes}


class NestedCell3(keras.layers.Layer):
    def __init__(self, nodes, features, channels, attention_heads, hidden_size, dropout_rate=0.2, **kwargs):
        super(NestedCell3, self).__init__(**kwargs)
        self.tot_nodes = nodes
        self.nodes_features = features
        self.hidden_size_in = channels * attention_heads
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.channels = channels
        self.state_size = [tf.TensorShape((self.tot_nodes, self.hidden_size))]
        self.output_size = [tf.TensorShape((self.tot_nodes, self.tot_nodes))]
        self.gnn = GATConv(channels=self.channels, attn_heads=self.attention_heads, concat_heads=True)
        self.decoder_mu = BatchBilinearDecoderDense(activation=None)
        self.decoder_sigma = BatchBilinearDecoderDense(activation=None)
        self.decoder_p = BatchBilinearDecoderDense(activation=None)
        self.drop = l.Dropout(dropout_rate)

    def build(self, input_shapes):
        self.b_u = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_u")
        self.b_r = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_r")
        self.b_c = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_c")
        self.W_u = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                   initializer="glorot_normal", name="W_u")
        self.W_r = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                   initializer="glorot_normal", name="W_r")
        self.W_c = self.add_weight(shape=(self.hidden_size + self.hidden_size_in, self.hidden_size),
                                   initializer="glorot_normal", name="W_c")

    def call(self, inputs, states, training):
        h = states[0]
        conv_u = self.gnn(inputs, training=training)  # B x N x d
        # Recurrence
        u = tf.nn.sigmoid(self.b_u + tf.concat([conv_u, h], -1) @ self.W_u)
        print(u.shape)
        r = tf.nn.sigmoid(self.b_r + tf.concat([conv_u, h], -1) @ self.W_r)
        c = tf.nn.tanh(self.b_c + tf.concat([conv_u, r * h], -1) @ self.W_c)
        h_prime = u * h + (1 - u) * c
        # h_prime = self.drop(h_prime, training=training)
        A = self.decoder_p(h_prime)
        return A, h_prime

    def get_config(self):
        return {"hidden": self.hidden_size, "nodes": self.tot_nodes}


class GRUGAT(l.Layer):
    def __init__(self, hidden_size=10, attn_heads=10, dropout=0.2, hidden_activation="relu", rc_gat=False,
                 temporal_smoothness=""):
        super(GRUGAT, self).__init__()
        self.gnn_u = GATConv(channels=hidden_size // 2, attn_heads=attn_heads, concat_heads=True,
                             activation=hidden_activation, dropout_rate=dropout, kernel_regularizer="l2")
        self.rc_gat = rc_gat
        if self.rc_gat:
            self.gnn_r = GATConv(channels=hidden_size // 2, attn_heads=attn_heads, concat_heads=True,
                                 activation=hidden_activation, dropout_rate=dropout, kernel_regularizer="l2")
            self.gnn_c = GATConv(channels=hidden_size // 2, attn_heads=attn_heads, concat_heads=True,
                                 activation=hidden_activation, dropout_rate=dropout, kernel_regularizer="l2")

        self.hidden_activation = hidden_activation
        self.hidden_size = (hidden_size // 2) * attn_heads
        self.drop = l.Dropout(dropout)
        self.state_size = self.hidden_size
        self.output_size = self.hidden_size
        self.temporal_smoothness = temporal_smoothness
        if self.temporal_smoothness:
            self.tmp_smooth = TemporalSmoothness(0.5, self.temporal_smoothness)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        x, a = inputs
        return tf.zeros(shape=(*x.shape[:-1], self.hidden_size))

    def build(self, input_shape):
        self.b_u = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal", name="b_u")
        self.b_r = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal", name="b_r")
        self.b_c = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal", name="b_c")
        self.W_u = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal",
                                   name="W_u")
        self.W_r = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal",
                                   name="W_r")
        self.W_c = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal",
                                   name="W_c")

    def call(self, inputs, state, training, *args, **kwargs):
        x, a = inputs
        # Encoding
        if state is None:
            h = self.get_initial_state(inputs)
        else:
            h = state

        conv_u = self.gnn_u(inputs, training=training)  # B x N x d
        if self.rc_gat:
            conv_r = self.gnn_r(inputs, training=training)  # B x N x d
            conv_c = self.gnn_c(inputs, training=training)  # B x N x d
        else:
            conv_r = conv_u
            conv_c = conv_u
        # Recurrence
        u = tf.nn.sigmoid(self.b_u + tf.concat([conv_u, h], -1) @ self.W_u)
        r = tf.nn.sigmoid(self.b_r + tf.concat([conv_r, h], -1) @ self.W_r)
        c = tf.nn.tanh(self.b_c + tf.concat([conv_c, r * h], -1) @ self.W_c)
        h_prime = u * h + (1 - u) * c
        h_prime = self.drop(h_prime, training=training)
        return h_prime


class RNNGAT(l.Layer):
    def __init__(self, hidden_size=10, attn_heads=10, dropout=0.2, hidden_activation="relu", rc_gat=False):
        super(RNNGAT, self).__init__()
        self.gnn = GATConv(channels=hidden_size // 2, attn_heads=attn_heads, concat_heads=True,
                           activation=hidden_activation, dropout_rate=dropout, kernel_regularizer="l2")

        self.hidden_activation = hidden_activation
        self.hidden_size = (hidden_size // 2) * attn_heads
        self.drop = l.Dropout(dropout)
        # self.state_size = self.hidden_size
        self.state_size = self.hidden_size
        self.output_size = self.hidden_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        x, a = inputs
        return tf.zeros(shape=(*x.shape[:-1], self.hidden_size))

    def build(self, input_shape):
        self.b = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal", name="b_u")
        self.W_h = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal",
                                   name="W_h")

    def call(self, inputs, state, training, *args, **kwargs):
        x, a = inputs
        # Encoding
        if state is None:
            h = self.get_initial_state(inputs)
        else:
            h = state

        conv_u = self.gnn(inputs, training=training)  # B x N x d
        # Recurrence
        h = tf.nn.sigmoid(self.b + tf.concat([conv_u, h], -1) @ self.W_h)
        h_prime = self.drop(h, training=training)
        return h_prime


class GRUGCN(l.Layer):
    def __init__(self, hidden_size=10, dropout=0.2, hidden_activation="relu", temporal_smoothness=""):
        super(GRUGCN, self).__init__()
        self.gnn_u = GCNConv(channels=hidden_size, concat_heads=True,
                             activation=hidden_activation, dropout_rate=dropout)
        self.gnn_r = GCNConv(channels=hidden_size, concat_heads=True,
                             activation=hidden_activation, dropout_rate=dropout)
        self.gnn_c = GCNConv(channels=hidden_size, concat_heads=True,
                             activation=hidden_activation, dropout_rate=dropout)

        self.hidden_activation = hidden_activation
        self.hidden_size = hidden_size
        self.drop = l.Dropout(dropout)
        self.state_size = self.hidden_size
        self.temporal_smoothness = temporal_smoothness
        if self.temporal_smoothness:
            self.tmp_smooth = TemporalSmoothness(0.5, self.temporal_smoothness)

    def get_initial_state(self, inputs):
        x, a = inputs
        self.state_size = (*x.shape[:-1], self.hidden_size)
        return tf.zeros(shape=(*self.state_size,))

    def build(self, input_shape):
        self.b_u = self.add_weight(name="b_u", shape=(self.hidden_size,), initializer="glorot_normal")
        self.b_r = self.add_weight(name="b_r", shape=(self.hidden_size,), initializer="glorot_normal")
        self.b_c = self.add_weight(name="b_c", shape=(self.hidden_size,), initializer="glorot_normal")
        self.W_u = self.add_weight(name="W_u", shape=(self.hidden_size * 2, self.hidden_size),
                                   initializer="glorot_normal")
        self.W_r = self.add_weight(name="W_r", shape=(self.hidden_size * 2, self.hidden_size),
                                   initializer="glorot_normal")
        self.W_c = self.add_weight(name="W_c", shape=(self.hidden_size * 2, self.hidden_size),
                                   initializer="glorot_normal")

    def call(self, inputs, states, training, *args, **kwargs):
        x, a = inputs
        if states is None:
            h = self.get_initial_state(inputs)
        else:
            h = states

        # Encoding
        a = gcn_filter(a.numpy(), symmetric=False)
        inputs = [a, inputs[1]]
        conv_u = self.gnn_u(inputs, training=training)
        conv_r = self.gnn_r(inputs, training=training)
        conv_c = self.gnn_c(inputs, training=training)

        # Recurrence
        u = tf.nn.sigmoid(self.b_u + tf.concat([conv_u, h], -1) @ self.W_u)
        r = tf.nn.sigmoid(self.b_r + tf.concat([conv_r, h], -1) @ self.W_r)
        c = tf.nn.tanh(self.b_c + tf.concat([conv_c, r * h], -1) @ self.W_c)
        h_prime = u * h + (1 - u) * c
        h_prime = self.drop(h_prime, training=training)
        if self.temporal_smoothness:
            states_transition = tf.concat([states, h_prime], axis=-1)
            self.add_loss(self.tmp_smooth(states_transition))
        return h_prime


class GRUGCNDirected(l.Layer):
    def __init__(self, hidden_size=10, dropout=0.2, hidden_activation="relu"):
        super(GRUGCNDirected, self).__init__()
        self.gnn_u = GCNDirected(hidden_size=hidden_size, activation=hidden_activation, dropout_rate=dropout)
        self.gnn_r = GCNDirected(hidden_size=hidden_size, activation=hidden_activation, dropout_rate=dropout)
        self.gnn_c = GCNDirected(hidden_size=hidden_size, activation=hidden_activation, dropout_rate=dropout)

        self.hidden_activation = hidden_activation
        self.hidden_size = hidden_size
        self.drop = l.Dropout(dropout)
        self.state_size = self.hidden_size

    def get_initial_state(self, inputs):
        x, a = inputs
        self.state_size = (*x.shape[:-1], self.hidden_size)
        return tf.zeros(shape=(*self.state_size,))

    def build(self, input_shape):
        self.b_u = self.add_weight(name="b_u", shape=(self.hidden_size,), initializer="glorot_normal")
        self.b_r = self.add_weight(name="b_r", shape=(self.hidden_size,), initializer="glorot_normal")
        self.b_c = self.add_weight(name="b_c", shape=(self.hidden_size,), initializer="glorot_normal")
        self.W_u = self.add_weight(name="W_u", shape=(self.hidden_size * 2, self.hidden_size),
                                   initializer="glorot_normal")
        self.W_r = self.add_weight(name="W_r", shape=(self.hidden_size * 2, self.hidden_size),
                                   initializer="glorot_normal")
        self.W_c = self.add_weight(name="W_c", shape=(self.hidden_size * 2, self.hidden_size),
                                   initializer="glorot_normal")

    def call(self, inputs, states, training, *args, **kwargs):
        if states is None:
            h = self.get_initial_state(inputs)
        else:
            h = states

        # Encoding
        conv_u = self.gnn_u(inputs, training=training)
        conv_r = self.gnn_r(inputs, training=training)
        conv_c = self.gnn_c(inputs, training=training)

        # Recurrence
        u = tf.nn.sigmoid(self.b_u + tf.concat([conv_u, h], -1) @ self.W_u)
        r = tf.nn.sigmoid(self.b_r + tf.concat([conv_r, h], -1) @ self.W_r)
        c = tf.nn.tanh(self.b_c + tf.concat([conv_c, r * h], -1) @ self.W_c)
        h_prime = u * h + (1 - u) * c
        h_prime = self.drop(h_prime, training=training)
        return h_prime


class GRU(l.Layer):
    def __init__(self, hidden_size=10, dropout=0.2, hidden_activation="relu"):
        super(GRU, self).__init__()
        self.enc_u = l.Dense(hidden_size, activation=hidden_activation)
        self.enc_r = l.Dense(hidden_size, activation=hidden_activation)
        self.enc_c = l.Dense(hidden_size, activation=hidden_activation)
        self.hidden_activation = hidden_activation
        self.drop = l.Dropout(dropout)
        self.hidden_size = hidden_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        x, a = inputs
        return tf.zeros(shape=(*x.shape[:-1], self.hidden_size))

    def build(self, input_shape):
        self.b_u = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal", name="b_u")
        self.b_r = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal", name="b_r")
        self.b_c = self.add_weight(shape=(self.hidden_size,), initializer="glorot_normal", name="b_c")
        self.W_u = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal",
                                   name="W_u")
        self.W_r = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal",
                                   name="W_r")
        self.W_c = self.add_weight(shape=(self.hidden_size * 2, self.hidden_size), initializer="glorot_normal",
                                   name="W_c")

    def call(self, inputs, state, training, *args, **kwargs):
        x, a = inputs
        # Encoding
        if state is None:
            h = self.get_initial_state(inputs)
        else:
            h = state

        conv_u = self.enc_u(x, training=training)  # B x N x d
        conv_r = self.enc_r(x, training=training)  # B x N x d
        conv_c = self.enc_c(x, training=training)  # B x N x d

        # Recurrence
        u = tf.nn.sigmoid(self.b_u + tf.concat([conv_u, h], -1) @ self.W_u)
        r = tf.nn.sigmoid(self.b_r + tf.concat([conv_r, h], -1) @ self.W_r)
        c = tf.nn.tanh(self.b_c + tf.concat([conv_c, r * h], -1) @ self.W_c)
        h_prime = u * h + (1 - u) * c
        h_prime = self.drop(h_prime, training=training)
        return h_prime


class GRUGATLognormal(m.Model):
    def __init__(self, hidden_size=4, attn_heads=4, dropout=0.2, hidden_activation="relu", temporal_smoothness=""):
        super(GRUGATLognormal, self).__init__()
        # Encoders
        self.GatRnn_p = GRUGAT(hidden_size=hidden_size, attn_heads=attn_heads, dropout=dropout,
                               hidden_activation=hidden_activation, temporal_smoothness=temporal_smoothness)
        self.GatRnn_mu = GRUGAT(hidden_size=hidden_size, attn_heads=attn_heads, dropout=dropout,
                                hidden_activation=hidden_activation, temporal_smoothness=temporal_smoothness)
        self.GatRnn_sigma = GRUGAT(hidden_size=hidden_size, attn_heads=attn_heads, dropout=dropout,
                                   hidden_activation=hidden_activation, temporal_smoothness=temporal_smoothness)

        # Decoders
        self.decoder_mu = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_sigma = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_p = BatchBilinearDecoderDense(activation=None, qr=False)

    def call(self, inputs, states, training=None, mask=None):
        # Encoding
        h_prime_p = self.GatRnn_p(inputs, states[0])
        h_prime_mu = self.GatRnn_mu(inputs, states[1])
        h_prime_sigma = self.GatRnn_sigma(inputs, states[2])

        # Decoding
        p = self.decoder_p(h_prime_p)
        p = tf.expand_dims(p, -1)
        mu = self.decoder_mu(h_prime_mu)
        mu = tf.expand_dims(mu, -1)
        sigma = self.decoder_sigma(h_prime_sigma)
        sigma = tf.expand_dims(sigma, -1)
        logits = tf.concat([p, mu, sigma], -1)
        return logits, h_prime_p, h_prime_mu, h_prime_sigma


class RNNGATLognormal(m.Model):
    def __init__(self, hidden_size=4, attn_heads=4, dropout=0.2, hidden_activation="relu"):
        super(RNNGATLognormal, self).__init__()
        # Encoders
        self.GatRnn_p = RNNGAT(hidden_size=hidden_size, attn_heads=attn_heads, dropout=dropout,
                               hidden_activation=hidden_activation)
        self.GatRnn_mu = RNNGAT(hidden_size=hidden_size, attn_heads=attn_heads, dropout=dropout,
                                hidden_activation=hidden_activation)
        self.GatRnn_sigma = RNNGAT(hidden_size=hidden_size, attn_heads=attn_heads, dropout=dropout,
                                   hidden_activation=hidden_activation)

        # Decoders
        self.decoder_mu = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_sigma = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_p = BatchBilinearDecoderDense(activation=None, qr=False)

    def call(self, inputs, states, training=None, mask=None):
        # Encoding
        h_prime_p = self.GatRnn_p(inputs, states[0])
        h_prime_mu = self.GatRnn_mu(inputs, states[1])
        h_prime_sigma = self.GatRnn_sigma(inputs, states[2])

        # Decoding
        p = self.decoder_p(h_prime_p)
        p = tf.expand_dims(p, -1)
        mu = self.decoder_mu(h_prime_mu)
        mu = tf.expand_dims(mu, -1)
        sigma = self.decoder_sigma(h_prime_sigma)
        sigma = tf.expand_dims(sigma, -1)
        logits = tf.concat([p, mu, sigma], -1)
        return logits, h_prime_p, h_prime_mu, h_prime_sigma


class GRULognormal(m.Model):
    def __init__(self, hidden_size=15, dropout=0.2, hidden_activation="tanh"):
        super(GRULognormal, self).__init__()
        # Encoders
        self.rnn_p = GRU(hidden_size=hidden_size, dropout=dropout,
                         hidden_activation=hidden_activation)
        self.rnn_mu = GRU(hidden_size=hidden_size, dropout=dropout,
                          hidden_activation=hidden_activation)
        self.rnn_sigma = GRU(hidden_size=hidden_size, dropout=dropout,
                             hidden_activation=hidden_activation)

        # Decoders
        self.decoder_mu = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_sigma = BatchBilinearDecoderDense(activation=None, qr=False)
        self.decoder_p = BatchBilinearDecoderDense(activation=None, qr=False)

    def call(self, inputs, states, training=None, mask=None):
        # Encoding
        h_prime_p = self.rnn_p(inputs, states[0])
        h_prime_mu = self.rnn_mu(inputs, states[1])
        h_prime_sigma = self.rnn_sigma(inputs, states[2])

        # Decoding
        p = self.decoder_p(h_prime_p)
        p = tf.expand_dims(p, -1)
        mu = self.decoder_mu(h_prime_mu)
        mu = tf.expand_dims(mu, -1)
        sigma = self.decoder_sigma(h_prime_sigma)
        sigma = tf.expand_dims(sigma, -1)
        logits = tf.concat([p, mu, sigma], -1)
        return logits, h_prime_p, h_prime_mu, h_prime_sigma


class NTN(m.Model):
    def __init__(self, n_layers=1, activation="tanh", output_activation="relu"):
        super(NTN, self).__init__()
        self.n_layers = n_layers
        self.ntns = []
        for i in range(self.n_layers):
            self.ntns.append(
                layers.NTN(activation=activation, output_activation=output_activation)
            )

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
    def __init__(self, hidden, activation="relu", use_mask=False, use_variance=False, qr=True, **kwargs):
        super(Bilinear, self).__init__(**kwargs)
        self.hidden = hidden
        self.use_mask = use_mask
        self.activation = activation
        self.qr = qr
        self.use_variance = use_variance

    def build(self, input_shape):
        self.bilinear = layers.Bilinear(self.hidden, self.activation, qr=self.qr)
        if self.use_mask and not self.use_variance:
            self.bilinear_mask = layers.Bilinear(
                self.hidden, activation="sigmoid", qr=self.qr
            )
        if self.use_variance and self.use_mask:
            self.bilinear_var = layers.Bilinear(
                self.hidden, activation=self.activation, qr=self.qr
            )
            self.bilinear_mask = layers.Bilinear(
                self.hidden, activation=self.activation, qr=self.qr
            )
        # self.regularizer = losses.SparsityRegularizerLayer(0.5)

    def call(self, inputs, **kwargs):
        X, A = self.bilinear(inputs)
        if self.use_mask:
            X_mask, A_mask = self.bilinear_mask(inputs)
            # A = tf.math.multiply(A, A_mask)
        if self.use_variance:
            X_var, A_var = self.bilinear_var(inputs)
        # Eliminate diagonal
        A = A * (tf.ones_like(A) - tf.eye(A.shape[0]))
        # A_flat = tf.reshape(A, [-1])
        # self.add_loss(self.regularizer(A_flat))

        if self.use_variance and self.use_mask:
            return X, X_mask, X_var, A, A_mask, A_var
        elif self.use_mask and not self.use_variance:
            return X, X_mask, A, A_mask
        else:
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
                    losses.EmbeddingSmoothnessRegularizerSparse(
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
                    losses.EmbeddingSmoothnessRegularizerSparse(
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
                    losses.EmbeddingSmoothnessRegularizerSparse(
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
                    losses.EmbeddingSmoothnessRegularizerSparse(
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
    def __init__(
            self,
            channels=10,
            attn_heads=10,
            concat_heads=False,
            add_self_loop=False,
            dropout_rate=0.5,
            output_activation="relu",
            use_mask=False,
            sparsity=0.0,
            **kwargs,
    ):
        super(GAT_BIL_spektral, self).__init__()

        self.channles = channels
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.add_self_loop = add_self_loop
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.use_mask = use_mask
        self.sparsity = sparsity

        self.gat = GATConv(
            channels=self.channles,
            attn_heads=self.attn_heads,
            concat_heads=self.concat_heads,
            dropout_rate=self.dropout_rate,
            add_self_loop=self.add_self_loop,
            **kwargs,
        )
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
    def __init__(
            self,
            channels=10,
            attn_heads=10,
            n_layers=2,
            concat_heads=True,
            dropout_rate=0.60,
            add_self_loop=False,
            return_attn_coef=False,
            output_activation="relu",
            use_mask=False,
            sparsity=0.0,
            **kwargs,
    ):
        super(GAT_BIL_spektral_dense, self).__init__()
        self.channles = channels
        self.attn_heads = attn_heads
        self.n_layers = n_layers
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.add_self_loop = add_self_loop
        self.return_attn_coef = return_attn_coef
        self.output_activation = output_activation
        self.use_mask = use_mask
        self.sparsity = sparsity

        self.gat = GATConv(
            channels=self.channles,
            attn_heads=self.attn_heads,
            concat_heads=self.concat_heads,
            dropout_rate=self.dropout_rate,
            add_self_loop=self.add_self_loop,
            return_attn_coef=self.return_attn_coef,
            **kwargs,
        )
        self.gat2 = GATConv(
            channels=self.channles // 2,
            attn_heads=self.attn_heads // 2,
            concat_heads=self.concat_heads,
            dropout_rate=self.dropout_rate,
            add_self_loop=self.add_self_loop,
            return_attn_coef=self.return_attn_coef,
            **kwargs,
        )
        self.bil_w = layers.BilinearDecoderDense(activation=self.output_activation)
        if self.use_mask:
            self.gat_bin = GATConv(
                channels=self.channles,
                attn_heads=self.attn_heads,
                concat_heads=self.concat_heads,
                dropout_rate=self.dropout_rate,
                add_self_loop=self.add_self_loop,
                return_attn_coef=self.return_attn_coef,
                **kwargs,
            )
            self.bil_mask = layers.BilinearDecoderDense("sigmoid")

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        zero_diag = tf.linalg.diag(tf.ones(a.shape[0]))
        zero_diag = tf.ones(a.shape) - zero_diag
        if self.return_attn_coef:
            xw, attn = self.gat(inputs)
            if self.use_mask:
                xb, attnb = self.gat_bin(inputs)
        else:
            xw = self.gat(inputs)
            xw = self.gat2([xw, inputs[-1]])
            if self.use_mask:
                xb = self.gat_bin(inputs)
        _, a_w = self.bil_w([xw, a])
        if not add_self_loop:
            a_w *= zero_diag
        if self.sparsity:
            self.add_loss(losses.SparsityRegularizerLayer(self.sparsity)(a_w))
        if self.use_mask:
            _, a_mask = self.bil_mask([xb, a])
            if not add_self_loop:
                a_mask *= zero_diag
            mask = tf.where(a_mask > 0.5, 1.0, 0.0)
            if self.sparsity:
                self.add_loss(losses.SparsityRegularizerLayer(self.sparsity)(a_mask))
            # a_final = tf.math.multiply(a_w, mask)
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
            tfp.distributions.RelaxedOneHotCategorical(0.1, 0.5 * tf.ones_like(inputs))
        )
        distr_layer = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda x: tfp.distributions.RelaxedOneHotCategorical(
                probs=x, temperature=self.temperature
            ),
            convert_to_tensor_fn=lambda x: x.sample(),
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1),
        )
        # bernulli_sampler = tfp.distributions.Bernoulli(probs=probs_dense)
        """a_m_sample = bernulli_sampler.sample()
        a_m_sample = tf.cast(a_m_sample, tf.float32)"""
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
            tfp.distributions.RelaxedOneHotCategorical(0.1, 0.5 * tf.ones_like(inputs))
        )
        distr_layer = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda x: tfp.distributions.RelaxedOneHotCategorical(
                probs=x, temperature=self.temperature
            ),
            convert_to_tensor_fn=lambda x: x.sample(),
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1),
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
        model = GAT_BIL_spektral(
            channels=10, attn_heads=10, concat_heads=False, dropout_rate=0.5
        )
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
