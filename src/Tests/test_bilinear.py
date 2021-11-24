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
import os

os.chdir("../../")
print(os.getcwd())
with open("./data_test.pkl", "rb") as file:
    A_sp = pkl.load(file)

A = tf.sparse.to_dense(A_sp)
'''diag_out = tf.math.reduce_sum(A, -1)
diag_in = tf.math.reduce_sum(A, 0)
diag = diag_in + diag_out
A = tf.linalg.set_diag(A, diag)'''
A_log = tf.clip_by_value(tf.math.log(A), 0., 1e100)
A_bin = tf.where(A_log > 0.0, 1., 0.)
A_log_sp = tf.sparse.from_dense(A_log)


model = models.Bilinear(15, qr=True)
x0, a0 = model(A_log)
x0 = x0.numpy()
a0 = a0.numpy()
bincross = tf.keras.losses.BinaryCrossentropy()
total = 200
tq = tqdm.tqdm(total=total)
p = 0.2
loss_hist = []
optimizer = tf.keras.optimizers.Adam(0.01)
for i in range(total):
    with tf.GradientTape() as tape:
        x, a = model(A_log)
        loss = losses.square_loss(A_log, a)
    loss_hist.append(loss)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    tq.update(1)
# pred = [pred[0], tf.sparse.to_dense(pred[-1])]
pred = [x.numpy(), a.numpy()]
loss_hist = np.asarray(loss_hist)
plt.plot(loss_hist)
plt.title("Loss History")
x0_tnse = TSNE(n_components=2, perplexity=80).fit_transform(x0)
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
plt.imshow(pred[-1] - A_log)
plt.colorbar()
plt.title("Difference Weighted - True Adj")

x_tnse = TSNE(n_components=2, perplexity=80).fit_transform(pred[0])
clustering = SpectralClustering(n_clusters=10, affinity='nearest_neighbors').fit(x_tnse)
labels = clustering.labels_
colormap = cm.get_cmap("Set1")
colors = colormap(labels)
fig, ax = plt.subplots()
ax.scatter(x_tnse[:, 0], x_tnse[:, 1], color=colors)
plot_names(x_tnse, ax)
plt.show()
