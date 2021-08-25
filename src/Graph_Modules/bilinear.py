import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as k
import numpy as np
from tensorflow.keras.losses import mean_squared_error, sparse_categorical_crossentropy

physical_devices = tf.config.list_physical_devices('CPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


class BilinearLayer(keras.layers.Layer):
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


class idx_to_Features(keras.layers.Layer):
    def __init__(self, n_relations, n_output, *args, **kwargs):
        super(idx_to_Features, self).__init__(*args, **kwargs)
        self.n_output = n_output
        self.n_relations = n_relations

    def build(self, input_shape):
        self.layers = []
        for i in range(self.n_relations):
            self.layers.append(keras.layers.Dense(units=self.n_output, activation="relu"))

    def call(self, inputs, **kwargs):
        output = []
        for i in range(self.n_relations):
            output.append(self.layers[i](inputs))
        return tf.transpose(output, perm=[1, 0, 2, 3])  # batches x relations x nodes x features


if __name__ == "__main__":
    batches = 10
    nodes = 20
    relations = 5
    features = 4
    nodes_idx_to_onehot = np.zeros(shape=(batches, nodes, nodes))  # batches x nodes x features
    idxs = np.arange(nodes)
    nodes_idx_to_onehot[:, idxs, idxs] = 1.

    np_input = np.ones(shape=(batches, relations, nodes, features))  # (batch, relations, nodes, features)
    adj = np.random.choice([0, 1], batches * nodes * nodes * relations, p=[0.5, 0.5]).reshape(batches, relations, nodes,
                                                                                              nodes)

    # input_layer = keras.layers.InputLayer(input_shape=np_input.shape[1:])
    input_layer = keras.layers.Input(shape=nodes_idx_to_onehot.shape[1:])
    idx_to_features = idx_to_Features(relations, features)(input_layer)
    bilinear = BilinearLayer(sum_relations=False)(idx_to_features)
    model = keras.models.Model(input_layer, bilinear)
    model.compile(optimizer="rmsprop", loss=mean_squared_error)
    model.fit(nodes_idx_to_onehot, adj, epochs=20)
    preds = model.predict(nodes_idx_to_onehot)
    print(preds[0,0,:,:])
