import tensorflow as tf
from tensorflow import scan
from tensorflow.keras.layers import LSTM
import tensorflow.keras as k
from tensorflow.keras.layers import TimeDistributed
from spektral.layers.convolutional import GATConv
from src.modules.layers import BilinearDecoderDense
from src.modules.utils import generate_list_lower_triang
from src.modules.models import GAT_BIL_spektral_dense
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import os

CUDA_VISIBLE_DEVICES = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

time = 50
nodes = 10
features = nodes
input_shape_nodes = (time, nodes, features)
input_shape_adj = (time, nodes, nodes)
Xt = np.random.normal(size=input_shape_nodes)
Xt = np.asarray(Xt, dtype=np.float32)
At = np.random.choice((0,1), size=input_shape_adj)
At = np.asarray(At, dtype=np.float32)
encoder = GATConv(channels=5, attn_heads=15, concat_heads=True)
channels = 4
attn_heads = 10
temporal_encoder = GATConv(channels=channels, attn_heads=attn_heads, concat_heads=True)
lower_triang_adj = generate_list_lower_triang(nodes, time, nodes)
decoder = BilinearDecoderDense()
z_dim = channels*attn_heads
latent_foward = k.layers.Dense(channels*attn_heads, activation="relu")
gatbil = GAT_BIL_spektral_dense()

Zts = [np.zeros((nodes, z_dim))]

A_preds = []

class GATRNN(k.models.Model):
    def __init__(self):
        super(GATRNN, self).__init__()
        self.encoder = GATConv(channels=10, attn_heads=10, concat_heads=True)
        self.decoder = BilinearDecoderDense()
        self.temporal_module = GATConv(channels=10, attn_heads=10, concat_heads=True)
        self.merger = k.layers.Dense(20)

    def call(self, inputs, training=None, mask=None):
        return scan(self._step, inputs)

    def _step(self, prev, current):


'''for i in range(1, time):
    A = At[i]
    X = Xt[i]
    Zt = encoder([X, A])
    prev_Zt = Zts[i-1]
    combine = tf.concat([Zt, prev_Zt], -1)
    Zt_prime = latent_foward(combine)
    Zts.append(Zt_prime)
    A_prime_pred = decoder([Zt_prime, A])
    A_preds.append(A_prime_pred)
'''
'''@tf.function
def gat_function(X, A):
    x, a = gatbil([X, A])
    return x, a

writer = tf.summary.create_file_writer('./Figures/log1/')
tf.summary.trace_on(graph=True, profiler=False)

rnn_gat(Xt,At)

with writer.as_default():
    tf.summary.trace_export(name="rnn_graph", step=0, profiler_outdir='./Figures/log1/')'''