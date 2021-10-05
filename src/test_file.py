import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras import initializers
from tensorflow.keras import layers as l
from tensorflow.keras import activations as a
from tensorflow.keras import models as m
import numpy as np


class BaseGraph(l.Layer):
    def __init__(self, **kwargs):
        super(BaseGraph, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)

    def call(self, inputs, *args, **kwargs):
        edgelist = inputs[0]
        s_embedding = inputs[1]
        t_embedding = inputs[2]
        r_embedding = inputs[3]
        values = inputs[4]

        message_s = self.aggregate(s_embedding, edgelist[:, 1])
        message_t = self.aggregate(r_embedding, edgelist[:2])
        t_propagate = self.propagate(message_s, edgelist[:, 0])
        return t_propagate

    def aggregate(self, inputs):
        raise NotImplemented

    def propagate(self, embeddings, indices):
        raise NotImplemented




