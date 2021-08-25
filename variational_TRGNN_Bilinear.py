import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from tensorflow import keras as k
from tensorflow.keras.layers import Conv1D

# Data Dict: {"Nodes Features", "Relations"}
# Relations Tensor: Time x  Relations x Nodes x Nodes
# Features Tensor: T x N x R x d

class Attention(k.layers.Layer):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        nodes_features = input_shape[0]
        relations_adj = input_shape[1]
        print(f"Node features shape: {nodes_features} \nRelations Adj shape: {relations_adj}")
        assert nodes_features[0] == relations_adj[0] == relations_adj[1]

        n_nodes = nodes_features[0]
        n_features = nodes_features[1]
        n_relations = relations_adj[2]

        key_convs = []
        value_convs = []
        query_convs = []
        
        for i in range(n_relations):
            key_convs.append(Conv1D(n_features, 1))
            value_convs.append(Conv1D(n_features, 1))
            query_convs.append(Conv1D(n_features, 1))


    def call(self, inputs, **kwargs):

        neighbours = tf.where()
        pass



class GCNN(k.layers.Layer):
    def __init__(self, **kwargs):
        super(GCNN, self).__init__(**kwargs)
        self.W = k.backend.variable()


class GAT(k.models.Model):
    pass
