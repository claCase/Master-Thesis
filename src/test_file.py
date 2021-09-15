import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras import layers as l
from tensorflow.keras import models as m
import numpy as np


class GraphLayer(l.Layer):
    def __init__(self, **kwargs):
        super(GraphLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)

    def call(self, inputs, *args, **kwargs):
        print(inputs)
        print(args)
        print(kwargs)
        return inputs


def make_data(years, relations, nodes):
    tmp_adj_list = []
    for t in range(years):
        for r in range(relations):
            for importer in range(nodes):
                for exporter in range(nodes):
                    if importer!=exporter:
                        link = np.random.choice((0,1))
                        if link:
                            value = np.random.lognormal(0,1)
                            tmp_adj_list.append((t, r, importer, exporter, value))
    tmp_adj_list = np.asarray(tmp_adj_list)
    sparse = tf.sparse.SparseTensor(tmp_adj_list[:,:-1], tmp_adj_list[:,-1], (years, relations, nodes, nodes))
    return sparse


if __name__ == '__main__':
    layer = GraphLayer()
    nodes = 150
    ft = 10
    t = 20
    rel = 5
    t = 60
    '''input_shape1 = (t, nodes, nodes, rel)
    input_shape2 = (t, nodes, ft, rel)
    i1 = l.Input(input_shape1, name="features")
    i2 = l.Input(input_shape2, name="realtions")
    o1 = layer([i1, i2])
    model = m.Model({i1, i2}, o1)
    features = np.random.normal(0,1,input_shape1)
    relations = np.random.normal(0,1,input_shape2)
    input = {features, relations}
    preds = model.predict(input)
    '''
    #data = make_data(t, rel, nodes)
