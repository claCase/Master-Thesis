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
A_log = tf.clip_by_value(tf.math.log(A), 0., 1e100)
A_bin = tf.where(A_log > 0.0, 1., 0.)
A_log_sp = tf.sparse.from_dense(A_log)
X = np.eye(174)  # np.random.normal(size=(174,174))
#model = models.GAT_BIL_spektral_dense(channels=15, attn_heads=20, use_mask=True, sparsity=0, return_attn_coef=True)
model = models.GAT_BIL_spektral(25)
loss_hist = []
optimizer = tf.keras.optimizers.Adam(0.01)
x0, a0 = model([X, A_log_sp])
a0 = tf.sparse.to_dense(a0)
#x0, a0, a_m0, attn0 = model([X, A_log])
# a = tf.sparse.to_dense(a)
inputs = [X, A_log_sp]
#inputs = [X, A_log]
bincross = tf.keras.losses.BinaryCrossentropy()
total = 600
tq = tqdm.tqdm(total=total)

for i in range(total):
    # model, loss_hist, pred = models.grad_step(inputs, model, losses.square_loss, optimizer, loss_hist)
    with tf.GradientTape() as tape:
        x, a = model(inputs)
        loss = losses.square_loss(inputs[-1], a)
    loss_hist.append(loss)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    tq.update(1)
pred = [x, tf.sparse.to_dense(a)]
abs_diff = tf.abs(tf.sparse.add(inputs[-1], a.__mul__(-1)))
abs_diff = tf.sparse.to_dense(abs_diff)
#pred = [x, a]
loss_hist = np.asarray(loss_hist)
plt.plot(loss_hist)
plt.title("Loss History")


x0_tnse = TSNE(n_components=2).fit_transform(x0)
plot_names(x0_tnse)

plt.figure()
plt.imshow(A_log)
plt.colorbar()
plt.title("True Adj")

plt.figure()
plt.imshow(pred[-1])
plt.colorbar()
plt.title("Pred")

plt.figure()
plt.imshow(abs_diff)
plt.colorbar()
plt.title("Absolute Difference")

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
