import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.losses import MeanSquaredError as mse
from src.modules.models import GAT_BIL_spektral_dense, GAT_BIL_spektral
from src.modules.losses import square_loss
from src.modules.losses import embedding_smoothness
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import tqdm
import pickle as pkl
import argparse
import os


def generate_data_block_matrix(data, sparse=True):
    t, r, n = data.shape[0], data.shape[1], data.shape[2]
    times = []
    for i in range(t):
        times.append(block_diag(*data[i]))
    block = block_diag(*times)
    diag_inputs = np.ones(t * r * n) * 5
    diag = np.empty(block.shape)
    index = 0
    tot_values = diag.shape[0]
    for i in range(t):
        for j in range(r):
            index += 1
            diag_upper = np.diagflat(diag_inputs, index * n)[:tot_values, :tot_values]
            diag_lower = np.diagflat(diag_inputs, -index * n)[:tot_values, :tot_values]
            diag += diag_upper + diag_lower
    if sparse:
        return tf.sparse.from_dense(block), tf.sparse.from_dense(diag)
    return block, diag


def generate_synthetic_block_matrix(time, relations, nodes, sparse=True):
    rels = []
    for i in range(relations):
        block = np.random.lognormal(
            size=(nodes, nodes)
        )  # * np.random.choice((0,1),size=(nodes, nodes))
        block = np.asarray(block, dtype=np.float32)
        rels.append(block)
    block_rels = block_diag(*rels)

    yrs = []
    for i in range(time):
        yrs.append(block_rels)
    block_time = block_diag(*yrs)
    # add self loop for r and t
    tot_values = nodes * time * relations
    diag = np.empty((tot_values, tot_values))
    diag_inputs = np.ones(tot_values) * 5
    index = 0
    for i in range(time):
        for j in range(relations):
            index += 1
            diag_upper = np.diagflat(diag_inputs, index * nodes)[
                :tot_values, :tot_values
            ]
            diag_lower = np.diagflat(diag_inputs, -index * nodes)[
                :tot_values, :tot_values
            ]
            diag += diag_upper + diag_lower
    if sparse:
        return tf.sparse.from_dense(block_time)
    return block_time, diag


def generate_features(data=None, t=10, r=10, n=10):
    if data is not None:
        t, r, n = data.shape[0], data.shape[1], data.shape[2]
    nodes_idx = np.arange(n)
    relations_idx = np.arange(r)
    time_idx = np.arange(t)
    comb = np.array(np.meshgrid(nodes_idx, relations_idx, time_idx)).T.reshape(-1, 3)
    i = np.arange(t * r * n)
    idxs_nodes = np.concatenate([i[:, None], comb[:, 0][:, None]], -1)
    idxs_relations = np.concatenate([i[:, None], comb[:, 1][:, None]], -1)
    idxs_times = np.concatenate([i[:, None], comb[:, 2][:, None]], -1)
    nodes = np.tile(np.eye(n), r * t).T
    relations = np.tile(np.eye(r), n * t).T
    times = np.tile(np.eye(t), n * r).T
    nodes[idxs_nodes] = 1.0
    relations[idxs_relations] = 1.0
    times[idxs_times] = 1.0
    F = np.concatenate([nodes, relations, times], -1)
    return F


parser = argparse.ArgumentParser()
parser.add_argument("--synthetic", action="store_true")
parser.add_argument("--t", type=int, default=1)
parser.add_argument("--r", type=int, default=5)
parser.add_argument("--n", type=int, default=174)
parser.add_argument("--sparse", action="store_true")
args = parser.parse_args()
synthetic = args.synthetic
t = args.t
r = args.r
n = args.n
sparse = args.sparse

if synthetic:
    block_time, diag = generate_synthetic_block_matrix(t, r, n, sparse=sparse)
    A = block_time + diag
else:
    with open(
            os.path.join(os.getcwd(), "Data", "complete_data_final_transformed_no_duplicate.pkl"),
            "rb",
    ) as file:
        data_np = pkl.load(file)
    data_sp = tf.sparse.SparseTensor(
        data_np[:, :4], data_np[:, 4], np.max(data_np, 0)[:4] + 1
    )
    data_sp = tf.sparse.reorder(data_sp)

    data_slice = tf.sparse.slice(data_sp, (20, 0, 0, 0), (t, r, n, n))
    data_dense = tf.sparse.to_dense(data_slice)
    data_dense = tf.math.log(data_dense)
    data_dense = tf.clip_by_value(data_dense, 0.0, 1e12)
    block_time, diag = generate_data_block_matrix(data_dense, sparse=sparse)
    if sparse:
        A = tf.sparse.add(block_time, diag)
    else:
        A = block_time  # + diag

X = np.eye(block_time.shape[0], dtype=np.float32)
if sparse:
    model = GAT_BIL_spektral(dropout_rate=0.5)
else:
    model = GAT_BIL_spektral_dense(
        add_self_loop=False, dropout_rate=0.5, use_mask=False
    )

mse = square_loss
epochs = 600
tq = tqdm.tqdm(total=epochs)
p = 0.2
loss_hist = []
optimizer = k.optimizers.RMSprop(0.001)

plt.imshow(A)
# plt.show()
for i in range(epochs):
    # model, loss_hist, pred = models.grad_step(inputs, model, losses.square_loss, optimizer, loss_hist)
    with tf.GradientTape() as tape:
        # A_mask = np.random.choice((0, 1), p=(p, 1 - p), size=block_time.shape)
        # A_corrupted = block_time * A_mask
        # inputs = [inputs[0], A_corrupted]
        x, a = model([X, A])
        loss = mse(block_time, a)
        # losse = embedding_smoothness(X, block_time)
        # loss_tot = loss + losse
        loss_hist.append(loss)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    tq.update(1)

if sparse:
    block_time = tf.sparse.to_dense(block_time).numpy()
    a = tf.sparse.to_dense(a)
fig, axs = plt.subplots(1, 2)
imp = axs[0].imshow(a)
axs[0].set_title("Pred")
imt = axs[1].imshow(block_time)
axs[1].set_title("True")
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.8, 0.1, 0.075, 0.8])
plt.colorbar(imt, cax=cax)
plt.figure()
plt.plot(loss_hist)
# plt.savefig("./Figures/sparse_block_loss.png")


a = a.numpy().flatten()
a_edges = a[a > 0.5]
block_edges = block_time.flatten()
block_edges = block_edges[block_edges > 0.5]
plt.figure()
plt.hist(a_edges, 100, color="red", alpha=0.5, density=True)
plt.hist(block_edges, 100, color="blue", alpha=0.5, density=True)
plt.show()
