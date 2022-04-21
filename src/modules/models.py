import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, _config_for_enable_caching_device, \
    _caching_device
from spektral.layers import GATConv
from src.modules.layers import BatchBilinearDecoderDense
from src.modules.losses import TemporalSmoothness
from tensorflow import keras


class NestedGRUCell(DropoutRNNCellMixin, l.Layer):
    def __init__(self, nodes, dropout, recurent_dropout, hidden_size_in, hidden_size_out, regularizer=None, gnn_h=False,
                 layer_norm=False, **kwargs):
        super(NestedGRUCell, self).__init__(**kwargs)
        self.tot_nodes = nodes
        self.hidden_size_in = hidden_size_in
        self.hidden_size_out = hidden_size_out
        self.recurrent_dropout = recurent_dropout
        self.dropout = dropout
        self.regularizer = regularizer
        self.gnn_h = gnn_h
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln = l.LayerNormalization()
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)

        if self.gnn_h:
            self.gat = GATConv(channels=hidden_size_out, attn_heads=1, concat_heads=True, dropout_rate=0)

    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        self.b_u = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_u",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.b_r = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_r",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.b_c = self.add_weight(shape=(self.tot_nodes, 1), initializer="glorot_normal", name="b_c",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.W_u = self.add_weight(shape=(self.hidden_size_out + self.hidden_size_in, self.hidden_size_out),
                                   initializer="glorot_normal", name="W_u_p",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.W_r = self.add_weight(shape=(self.hidden_size_out + self.hidden_size_in, self.hidden_size_out),
                                   initializer="glorot_normal", name="W_r_p",
                                   regularizer=self.regularizer, caching_device=default_caching_device)
        self.W_c = self.add_weight(shape=(self.hidden_size_out + self.hidden_size_in, self.hidden_size_out),
                                   initializer="glorot_normal", name="W_c_p",
                                   regularizer=self.regularizer, caching_device=default_caching_device)

    def call(self, inputs, states, training, *args, **kwargs):
        if self.gnn_h:
            x, a = inputs
        else:
            x = inputs
        h = states
        if 0 < self.recurrent_dropout < 1:
            h_mask = self.get_recurrent_dropout_mask_for_cell(inputs=states, training=training, count=1)
            h = h * h_mask
        if 0 < self.dropout < 1:
            x_mask = self.get_dropout_mask_for_cell(inputs=x, training=training, count=1)
            x = x * x_mask
        if self.gnn_h:
            h = self.gat([x, a])
        u = tf.nn.sigmoid(self.b_u + tf.concat([x, h], -1) @ self.W_u)
        r = tf.nn.sigmoid(self.b_r + tf.concat([x, h], -1) @ self.W_r)
        c = tf.nn.tanh(self.b_c + tf.concat([x, r * h], -1) @ self.W_c)
        h_prime = u * h + (1 - u) * c
        if self.layer_norm:
            h_prime = self.ln(h_prime)
        return h_prime

    def get_config(self):
        config = {"nodes": self.tot_nodes,
                  "hidden_size_in": self.hidden_size_in,
                  "hidden_size_out": self.hidden_size_out,
                  "dropout": self.dropout,
                  "recurrent_dropout": self.recurrent_dropout,
                  "regularizer": self.regularizer
                  }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(NestedGRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RecurrentEncoderDecoder(DropoutRNNCellMixin, keras.layers.Layer):
    def __init__(self, nodes, features, channels, attention_heads, hidden_size, dropout_adj, dropout, recurrent_dropout,
                 regularizer=None, qr=True, symmetric=True, add_self_loop=False, gnn=True, gnn_h=False, **kwargs):
        super(RecurrentEncoderDecoder, self).__init__(**kwargs)
        self.tot_nodes = nodes
        self.nodes_features = features
        if symmetric:
            self.hidden_size_in = 2 * (channels * attention_heads)
        else:
            self.hidden_size_in = channels * attention_heads
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.channels = channels
        self.regularizer = regularizer
        self.symmetric = symmetric
        self.gnn = gnn
        self.gnn_h = gnn_h

        self.state_size = [tf.TensorShape((self.tot_nodes, self.hidden_size)),
                           tf.TensorShape((self.tot_nodes, self.hidden_size)),
                           tf.TensorShape((self.tot_nodes, self.hidden_size))]
        self.output_size = [tf.TensorShape((self.tot_nodes, self.tot_nodes, 3))]
        '''self.dnn_emb = l.Dense(self.nodes_features, "relu")
        self.dnn_emb1 = l.Dense(self.nodes_features, "relu")'''
        if self.gnn:
            if self.symmetric:
                self.gnn_in = GATConv(channels=self.channels, attn_heads=self.attention_heads, concat_heads=True,
                                      dropout_rate=0, kernel_regularizer=self.regularizer,
                                      bias_regularizer=self.regularizer, attn_kernel_regularizer=self.regularizer,
                                      activation="relu", add_self_loops=add_self_loop)
                self.gnn_in1 = GATConv(channels=self.channels, attn_heads=self.attention_heads, concat_heads=True,
                                       dropout_rate=0, kernel_regularizer=self.regularizer,
                                       bias_regularizer=self.regularizer, attn_kernel_regularizer=self.regularizer,
                                       activation="relu", add_self_loops=add_self_loop)
                self.gnn_out = GATConv(channels=self.channels, attn_heads=self.attention_heads, concat_heads=True,
                                       dropout_rate=0, kernel_regularizer=self.regularizer,
                                       bias_regularizer=self.regularizer, attn_kernel_regularizer=self.regularizer,
                                       activation="relu", add_self_loops=add_self_loop)
                self.gnn_out1 = GATConv(channels=self.channels, attn_heads=self.attention_heads, concat_heads=True,
                                        dropout_rate=0, kernel_regularizer=self.regularizer,
                                        bias_regularizer=self.regularizer, attn_kernel_regularizer=self.regularizer,
                                        activation="relu", add_self_loops=add_self_loop)
            else:
                self.gnn_gat = GATConv(channels=self.channels, attn_heads=self.attention_heads, concat_heads=True,
                                       dropout_rate=0, kernel_regularizer=self.regularizer,
                                       bias_regularizer=self.regularizer, attn_kernel_regularizer=self.regularizer,
                                       activation="relu", add_self_loops=add_self_loop)
        self.recurrent_p = NestedGRUCell(nodes, dropout, recurrent_dropout, self.hidden_size_in, self.hidden_size,
                                         regularizer=self.regularizer, name="rec_p", gnn_h=self.gnn_h)
        self.recurrent_mu = NestedGRUCell(nodes, dropout, recurrent_dropout, self.hidden_size_in, self.hidden_size,
                                          regularizer=self.regularizer, name="rec_mu", gnn_h=self.gnn_h)
        self.recurrent_sigma = NestedGRUCell(nodes, dropout, recurrent_dropout, self.hidden_size_in, self.hidden_size,
                                             regularizer=self.regularizer, name="rec_sigma", gnn_h=self.gnn_h)
        self.decoder_mu = BatchBilinearDecoderDense(activation="relu", regularizer=None, qr=qr)
        self.decoder_sigma = BatchBilinearDecoderDense(activation=None, regularizer=None, qr=qr)
        self.decoder_p = BatchBilinearDecoderDense(activation=None, regularizer=None, qr=qr, zero_diag=False)
        self.dropout = dropout_adj
        self._enable_caching_device = kwargs.pop('enable_caching_device', True)

    def build(self, input_shapes):
        x, a = input_shapes
        if not self.gnn:
            self.kernel = self.add_weight(shape=(x[-1], self.hidden_size_in), name="x_kernel")

    def call(self, inputs, states, training=True):
        x, a = tf.nest.flatten(inputs)
        h_p, h_mu, h_sigma = tf.nest.flatten(states)
        if 0 < self.dropout < 1:
            inputs_mask = self.get_dropout_mask_for_cell(inputs=a, training=training, count=1)
            a = a * inputs_mask
        # x = self.dnn_emb1(self.dnn_emb(x))
        if self.gnn:
            if self.symmetric:
                '''a_lower = tf.linalg.LinearOperatorLowerTriangular(a).to_dense()
                a_lower_symm = a_lower + tf.transpose(a_lower, perm=(0, 2, 1))
                a_upper = tf.linalg.LinearOperatorLowerTriangular(tf.transpose(a, perm=(0, 2, 1))).to_dense()
                a_upper_symm = a_upper + tf.transpose(a_upper, perm=(0, 2, 1))'''
                a_upper_symm = tf.transpose(a, perm=(0, 2, 1))
                '''x_prime_in = self.gnn_in1([self.gnn_in([x, a_lower_symm]),  a_lower_symm])  # B x N x d
                x_prime_out = self.gnn_out1([self.gnn_out([x, a_upper_symm]), a_upper_symm])  # B x N x d
                '''
                x_prime_in = self.gnn_in([x, a])  # B x N x d
                x_prime_out = self.gnn_out([x, a_upper_symm])  # B x N x d
                x_prime = tf.concat([x_prime_in, x_prime_out], -1)
            else:
                x_prime = self.gnn_gat([x, a])
        else:
            x_prime = x @ self.kernel

        # Recurrence
        if self.gnn_h:
            x_prime = [x_prime, a]
        h_prime_p = self.recurrent_p(x_prime, h_p, training=training)
        h_prime_mu = self.recurrent_mu(x_prime, h_mu, training=training)
        h_prime_sigma = self.recurrent_sigma(x_prime, h_sigma, training=training)

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
                  "regularizer": self.regularizer,
                  "symmetric": self.symmetric,
                  "gnn": self.gnn,
                  "output_size": self.output_size,
                  "state_size": self.state_size
                  }
        config.update(_config_for_enable_caching_device(self))
        base_config = super(RecurrentEncoderDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RecurrentEncoderDecoder2(DropoutRNNCellMixin, keras.layers.Layer):
    def __init__(self, nodes, features, channels, attention_heads, hidden_size, dropout, recurrent_dropout, **kwargs):
        super(RecurrentEncoderDecoder2, self).__init__(**kwargs)
        self.tot_nodes = nodes
        self.nodes_features = features
        self.hidden_size_in = channels * attention_heads
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.channels = channels
        self.state_size = [tf.TensorShape((self.tot_nodes, self.hidden_size))]
        self.output_size = [tf.TensorShape((self.tot_nodes, self.tot_nodes))]
        self.gnn = GATConv(channels=self.channels, attn_heads=self.attention_heads, concat_heads=True, dropout_rate=0)
        self.recurrent = NestedGRUCell(nodes, dropout, recurrent_dropout, self.hidden_size_in, self.hidden_size,
                                       name="rec")
        self.decoder = BatchBilinearDecoderDense(activation="relu")
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self._enable_caching_device = kwargs.pop('enable_caching_device', True)

    def build(self, input_shapes):
        pass

    def call(self, inputs, states, training, *args, **kwargs):
        x, a = tf.nest.flatten(inputs)
        h = states
        if 0 < self.dropout < 1:
            inputs_mask = self.get_dropout_mask_for_cell(inputs=a, training=training, count=1)
            a = a * inputs_mask
        conv = self.gnn([x, a], training=training)  # B x N x d
        # Recurrence
        h_prime = self.recurrent(conv, h, training=training)
        A = self.decoder(h_prime)
        return A, h_prime

    def get_config(self):
        config = {"nodes": self.tot_nodes,
                  "nodes_features": self.nodes_features,
                  "hidden_size": self.hidden_size,
                  "attention_heads": self.attention_heads,
                  "channels": self.channels,
                  "dropout": self.dropout,
                  "recurrent_dropout": self.recurrent_dropout}
        config.update(_config_for_enable_caching_device(self))
        base_config = super(RecurrentEncoderDecoder2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

