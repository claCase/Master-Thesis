import tensorflow as tf
from tensorflow.keras.layers import LSTM, TimeDistributed
import tensorflow.keras as k
from spektral.layers.convolutional import GATConv
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from src.modules.layers import SelfAttention


def _get_positional_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def generate_list_lower_triang(batch, t, lag):
    lower_adj = np.tril(np.ones(shape=(t, t)))
    prev = np.zeros(shape=(lag, t))
    sub_lower = np.vstack([prev, lower_adj])[:-lag]
    lower_adj = lower_adj - sub_lower
    return np.asarray([lower_adj] * batch)


n_nodes = 100
f_dim = 2
t = 20
pos = _get_positional_encoding_matrix(t, f_dim)
# drift = np.random.uniform(0, 1, size=(n_nodes, 2))
d = 0.0
drift = np.zeros(shape=(n_nodes, f_dim)) + d
trajectories = np.zeros(shape=(t, n_nodes, f_dim * 2))
lags = 15
nodes_lower_adj = generate_list_lower_triang(n_nodes, t, lags)

for i in range(1, t-1):
    update = np.random.normal(loc=drift, scale=0.5)
    X_prime = trajectories[i - 1, :, :2] + update
    trajectories[i, :, :2] = X_prime

nodes_trajectories = np.swapaxes(trajectories, 0, 1)  # NxTxd

fig, axes = plt.subplots(3, 1)
fig_l, axes_l = plt.subplots()

for i in range(n_nodes):
    axes[0].plot(nodes_trajectories[i, 1:-1, 0], nodes_trajectories[i, 1:-1, 1])
axes[0].set_title("True Trajectories")

for n in range(n_nodes):
    nodes_trajectories[n, :, 2:] = pos

spektral_gat = True
activations = 5
heads = 5
drop_rate = 0.1
concat = True
if spektral_gat:
    gat = GATConv(
        channels=activations,
        attn_heads=heads,
        dropout_rate=drop_rate,
        concat_heads=concat,
        add_self_loops=False,
        return_attn_coef=True
    )
else:
    gat = SelfAttention(
        channels=activations,
        attn_heads=heads,
        dropout_rate=drop_rate,
        concat_heads=concat,
        return_attn=True
    )

o = k.layers.Dense(2, "linear")
optimizer = k.optimizers.RMSprop(0.001)
l = k.losses.MeanSquaredError()
loss_hist = []

epochs = 1000
proc = tqdm.tqdm(total=epochs, position=0, leave=False, desc="Training GAT")
for i in range(epochs):
    with tf.GradientTape() as tape:
        X, attn = gat([nodes_trajectories[:, :-1, :], nodes_lower_adj[:, :-1, :-1]])
        X = o(X)
        loss = l(nodes_trajectories[:, 1:, :2], X)
        loss_hist.append(loss)
    gradients = tape.gradient(loss, gat.trainable_weights)
    optimizer.apply_gradients(zip(gradients, gat.trainable_weights))
    proc.update(1)
proc.close()

axes_l.set_title("Loss History")
(h1,) = axes_l.plot(loss_hist)

axes[1].set_title("Gat Prediction")
for i in range(n_nodes):
    axes[1].plot(X[i, :, 0], X[i, :, 1])

plt.figure()
if spektral_gat:
    plt.imshow(attn[0, :, 0, :])
else:
    plt.imshow(attn[0, 0, :, :])

i = k.Input(shape=(None, f_dim * 2), batch_size=n_nodes)
lstm = LSTM(activations, return_sequences=True, dropout=drop_rate)(i)
o = TimeDistributed(k.layers.Dense(2, "linear"))(lstm)
lstm_model = k.models.Model(i, o)

optimizer = k.optimizers.RMSprop(0.001)
proc = tqdm.tqdm(total=epochs, position=0, leave=False, desc="Training LSTM")
loss_hist = []
for i in range(epochs):
    with tf.GradientTape() as tape:
        X = lstm_model(nodes_trajectories[:, :-1, :])
        loss = l(nodes_trajectories[:, 1:, :2], X)
        # X = lstm_model(nodes_trajectories)
        # loss = l(nodes_trajectories[:, :, :2], X)
        loss_hist.append(loss)
    gradients = tape.gradient(loss, lstm_model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, lstm_model.trainable_weights))
    proc.update(1)
proc.close()
(h2,) = axes_l.plot(loss_hist)
axes_l.legend([h1, h2], ["Gat loss", "LSTM Loss"], loc="upper right", shadow=True)
axes[2].set_title("LSTM Predictions")
for i in range(n_nodes):
    axes[2].plot(X[i, :, 0], X[i, :, 1])
plt.tight_layout()

plt.show()
