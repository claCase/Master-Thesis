import numpy as np
import tensorflow as tf
import pickle as pkl
from src.modules import models, losses, layers, utils
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, SpectralClustering
import matplotlib.pyplot as plt
from matplotlib import cm
from src.graph_data_loader import plot_names
import tqdm

'''with open(
        "A:\\Users\\Claudio\\Documents\\PROJECTS\\Master-Thesis\\Data\\complete_data_final_transformed_no_duplicate.pkl",
        "rb",
) as file:
    data_np = pkl.load(file)
data_sp = tf.sparse.SparseTensor(
    data_np[:, :4], data_np[:, 4], np.max(data_np, 0)[:4]
)
data_sp = tf.sparse.reorder(data_sp)
'''
with open("/data_test.pkl", "rb") as file:
    A_sp = pkl.load(file)
# A_sp = tf.sparse.reduce_sum(tf.sparse.slice(data_sp, (50, 10, 0, 0), (1, 1, 174, 174)), (0, 1), output_is_sparse=True)

A = tf.sparse.to_dense(A_sp)
'''diag_out = tf.math.reduce_sum(A, -1)
diag_in = tf.math.reduce_sum(A, 0)
diag = diag_in + diag_out
A = tf.linalg.set_diag(A, diag)'''
A_log = tf.clip_by_value(tf.math.log(A), 0., 1e100)
A_bin = tf.where(A_log > 0.0, 1., 0.)
A_log_sp = tf.sparse.from_dense(A_log)
X = np.eye(174)  # np.random.normal(size=(174,174))
model = models.GAT_BIL_spektral_dense(channels=5, attn_heads=20, use_mask=True, sparsity=0, return_attn_coef=True)
# model = models.GAT_BIL_spektral(25)
loss_hist = []
optimizer = tf.keras.optimizers.Adam(0.01)
# x, a = model([X, A_log_sp])
x0, a0, a_m0, attn0 = model([X, A_log])
# a = tf.sparse.to_dense(a)
# inputs = [X, A_log_sp]
inputs = [X, A_log]
bincross = tf.keras.losses.BinaryCrossentropy()
total = 600
tq = tqdm.tqdm(total=total)
p = 0.2
for i in range(total):
    # model, loss_hist, pred = models.grad_step(inputs, model, losses.square_loss, optimizer, loss_hist)
    with tf.GradientTape() as tape:
        A_mask = np.random.choice((0, 1), p=(p, 1 - p), size=(174, 174))
        A_corrupted = A_log * A_mask
        # inputs = [inputs[0], A_corrupted]
        x, a, a_m, attn = model([X, A_corrupted])
        loss_a = losses.square_loss(A_log, a)
        # loss_m = losses.square_loss(A_bin, a_m)
        loss_m = bincross(A_bin, a_m)
        loss = [loss_a, loss_m]
        loss_hist.append(loss)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    tq.update(1)
# pred = [pred[0], tf.sparse.to_dense(pred[-1])]
inputs = [A, A_log]
pred = [x, a]
loss_hist = np.asarray(loss_hist)
plt.plot(loss_hist[:, 0])
plt.title("Loss History W")
plt.figure()
plt.plot(loss_hist[:, 1])
plt.title("Loss History B")
x0_tnse = TSNE(n_components=2).fit_transform(x0)
plot_names(x0_tnse)
plt.figure()
plt.imshow(A_log)
plt.colorbar()
plt.title("True Adj")
plt.figure()
plt.imshow(pred[-1])
plt.colorbar()
plt.title("Pred Weighted Adj")
plt.figure()
plt.imshow((a_m.numpy() > 0.5) * a)
plt.colorbar()
plt.title("Pred Weighted*Bin Adj")
plt.figure()
plt.imshow(np.abs(pred[-1].numpy() - A_log))
plt.colorbar()
plt.title("Absolute Difference Weighted Adj")
plt.figure()
plt.imshow(np.abs((a_m.numpy() > 0.5) * a - A_log))
plt.colorbar()
plt.title("Absolute Difference Weighted Adj*bin")
plt.figure()
att = attn[:, 0, :]  # np.mean(attn.numpy(), axis=1), 0.0001))
plt.imshow(att)
plt.title("Attention Coefficient")
plt.colorbar()
plt.figure()
att0 = attn0[:, 0, :]  # np.mean(attn0.numpy(), axis=1), 0.0001))
plt.imshow(att0)
plt.title("Initial Attention Coefficient")
plt.colorbar()
x_tnse = TSNE(n_components=2).fit_transform(pred[0])
# clustering = DBSCAN().fit(pred[0])
clustering = SpectralClustering(n_clusters=10, affinity='nearest_neighbors').fit(x_tnse)
labels = clustering.labels_
colormap = cm.get_cmap("Set1")
colors = colormap(labels)
fig, ax = plt.subplots()
ax.scatter(x_tnse[:, 0], x_tnse[:, 1], color=colors)
plot_names(x_tnse, ax)
plt.show()
