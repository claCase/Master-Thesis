import tensorflow as tf
from tensorflow.keras.layers import LSTM
import tensorflow.keras as k
from src.modules.models import GAT_BIL_spektral_dense
from spektral.data.loaders import BatchLoader
from spektral.data.dataset import Dataset
from tensorflow_addons.layers import MultiHeadAttention
from spektral.data.graph import Graph
from spektral.layers.convolutional import GATConv
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def _get_positional_encoding_matrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


n_nodes = 50
f_dim = 2
t = 40
pos = _get_positional_encoding_matrix(t, 2)
# drift = np.random.uniform(0, 1, size=(n_nodes, 2))
d = 0.
drift = np.zeros(shape=(n_nodes, f_dim)) + d
X0 = np.zeros(shape=(n_nodes, f_dim))
trajectories = np.zeros(shape=(t, n_nodes, f_dim * 2))
trajectories[0, :, :2] = X0
lower_adj = np.tril(np.ones(shape=(t, t)))
n_prev = 30
prev = np.zeros(shape=(n_prev, t))
sub_lower = np.vstack([prev, lower_adj])[:-n_prev]
lower_adj = lower_adj - sub_lower
nodes_lower_adj = np.asarray([lower_adj] * n_nodes)

for i in range(1, t - 1):
    update = np.random.normal(loc=drift, scale=0.5)
    # update = np.random.lognormal(mean=drift, sigma=0.1)
    X_prime = trajectories[i - 1, :, :2] + update
    trajectories[i, :, :2] = X_prime
    print(trajectories[i, :10, :])

nodes_trajectories = np.swapaxes(trajectories, 0, 1)

fig, axes = plt.subplots(3, 1)
fig_l, axes_l = plt.subplots(2, 1)

for i in range(n_nodes):
    axes[0].plot(nodes_trajectories[i, :, 0], nodes_trajectories[i, :, 1])
axes[0].set_title("True Trajectories")

g_list = []
for i in range(n_nodes):
    g_list.append(Graph(x=nodes_trajectories[i, :, :], adj=nodes_lower_adj[i]))

for n in range(n_nodes):
    nodes_trajectories[n, :, 2:] = pos

gat = GATConv(channels=2, attn_heads=1, concat_heads=False, add_self_loops=False)
# o = k.layers.Dense(2, "tanh")
optimizer = k.optimizers.RMSprop(0.001)
l = k.losses.MeanSquaredError()
loss_hist = []


for i in range(1000):
    with tf.GradientTape() as tape:
        X = gat([nodes_trajectories[:, :-1, :], nodes_lower_adj[:, :-1, :-1]])
        loss = l(nodes_trajectories[:, 1:, :2], X)
        loss_hist.append(loss)
        print(f"loss {loss}")
    gradients = tape.gradient(loss, gat.trainable_weights)
    optimizer.apply_gradients(zip(gradients, gat.trainable_weights))

axes_l[0].set_title("GAT Loss History")
axes_l[0].plot(loss_hist)

axes[1].set_title("Gat Prediction")
for i in range(n_nodes):
    axes[1].plot(X[i, :, 0], X[i, :, 1])

i = k.Input(shape=(None, f_dim * 2), batch_size=n_nodes)
lstm = LSTM(10, return_sequences=True)(i)
o = k.layers.Dense(2)(lstm)
lstm_model = k.models.Model(i, o)

loss_hist = []
for i in range(1000):
    with tf.GradientTape() as tape:
        X = lstm_model(nodes_trajectories[:, :-1, :])
        loss = l(nodes_trajectories[:, 1:, :2], X)
        #X = lstm_model(nodes_trajectories)
        #loss = l(nodes_trajectories[:, :, :2], X)
        loss_hist.append(loss)
        print(f"loss {loss}")
    gradients = tape.gradient(loss, lstm_model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, lstm_model.trainable_weights))

axes_l[1].set_title("LSTM Loss History")
axes_l[1].plot(loss_hist)
axes[2].set_title("LSTM Predictions")
for i in range(n_nodes):
    axes[2].plot(X[i, :, 0], X[i, :, 1])
plt.tight_layout()

plt.show()
