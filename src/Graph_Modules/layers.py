import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l
import tensorflow.keras.backend as k
from tensorflow.keras.losses import mean_squared_error, sparse_categorical_crossentropy

physical_devices = tf.config.list_physical_devices('CPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


nodes = 100
features = 10
features_dim = 5
E = np.empty((nodes, nodes, features))  # edge feature matrix
F = np.random.normal(0, 1, size=(features, features_dim))  # features of edges

for i in range(nodes):
    for j in range(nodes):
        z = np.zeros(features)
        choice = np.random.choice(np.arange(features))
        z[choice] = 1
        E[i, j, :] = z

E1 = np.einsum('ijk,kl-> ijl', E, F)
print(E1.shape)



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
        return tf.math.dot()

































