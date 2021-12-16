import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import pickle as pkl
from src.modules import models, losses, layers, utils
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, SpectralClustering
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as stats
from src.graph_data_loader import plot_names
import tqdm
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--mask", action="store_true")
    args = parser.parse_args()
    normalize = args.normalize
    use_mask = args.mask

    os.chdir("../../")
    print(os.getcwd())
    with open("./data_test.pkl", "rb") as file:
        A_sp = pkl.load(file)

    """A = tf.sparse.to_dense(A_sp)
    A_log = tf.clip_by_value(tf.math.log(A), 0., 1e100)
    A_bin = tf.where(A_log > 0.0, 1., 0.)
    A_log_sp = tf.sparse.from_dense(A_log)
    """
    with open(
        "A:\\Users\\Claudio\\Documents\\PROJECTS\\Master-Thesis\\Data\\complete_data_final_transformed_no_duplicate.pkl",
        "rb",
    ) as file:
        data_np = pkl.load(file)
    data_sp = tf.sparse.SparseTensor(
        data_np[:, :4], data_np[:, 4], np.max(data_np, 0)[:4]
    )
    data_sp = tf.sparse.reorder(data_sp)
    data_slice = tf.sparse.slice(data_sp, (50, 30, 0, 0), (1, 1, 174, 174))
    data_dense = tf.sparse.reduce_sum(data_slice, (0, 1))
    data_dense += 1
    data_dense = tf.math.log(data_dense)
    if normalize:
        data_dense = (data_dense - tf.reduce_mean(data_dense, (0, 1))) / (
            tf.math.reduce_std(data_dense, (0, 1))
        )
    A_log = data_dense
    A_bin = tf.where(data_dense > 0.0, 1.0, 0.0)

    model = models.Bilinear(15, qr=True, use_mask=use_mask, activation="relu")
    if use_mask:
        x0, x_m0, a0, a_m0 = model(A_log)
        x0 = x0.numpy()
        a0 = (a0 * a_m0).numpy()
    else:
        x0, a0 = model(A_log)
        x0 = x0.numpy()
        a0 = a0.numpy()

    bincross = tf.keras.losses.BinaryCrossentropy()
    total = 300
    tq = tqdm.tqdm(total=total)
    l = tf.keras.losses.MeanSquaredError()
    p = 0.2
    loss_hist = []
    optimizer = tf.keras.optimizers.Adam(0.01)
    for i in range(total):
        with tf.GradientTape() as tape:
            if use_mask:
                x, x_m, a, a_m = model(A_log)
                loss_a = losses.square_loss(A_log, a)
                loss_a_m = tf.reduce_mean(k.losses.binary_crossentropy(A_bin, a_m))
                loss = loss_a + loss_a_m
            else:
                x, a = model(A_log)
                loss = losses.square_loss(A_log, a)
        loss_hist.append(loss)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        tq.update(1)

    A_log = tf.clip_by_value(data_dense, 0.0, 1e12)
    loss_hist = np.asarray(loss_hist)
    plt.plot(loss_hist)
    plt.title("Loss History")
    x0_tnse = TSNE(n_components=2, perplexity=80).fit_transform(x0)
    plot_names(x0_tnse)
    plt.figure()
    plt.imshow(A_log.numpy())
    plt.colorbar()
    plt.title("True Adj")
    plt.figure()
    if use_mask:
        plt.imshow((a * a_m).numpy())
    else:
        plt.imshow(a.numpy())
    plt.colorbar()
    plt.title("Pred Weighted Adj")
    plt.figure()
    diff = a.numpy() - A_log.numpy()
    plt.imshow(diff)
    plt.colorbar()
    plt.title("Difference Weighted - True Adj")

    x_tnse = TSNE(n_components=2, perplexity=80).fit_transform(x.numpy())
    clustering = SpectralClustering(n_clusters=10, affinity="nearest_neighbors").fit(
        x_tnse
    )
    labels = clustering.labels_
    colormap = cm.get_cmap("Set1")
    colors = colormap(labels)
    fig, ax = plt.subplots()
    ax.scatter(x_tnse[:, 0], x_tnse[:, 1], color=colors)
    plot_names(x_tnse, ax)

    if use_mask:
        a_pred = (a * a_m).numpy().flatten()
        a_pred = a_pred[a_pred > 0.2]
    else:
        a_pred = a.numpy().flatten()
        a_pred = a_pred[a_pred > 0.2]
    a_true = A_log.numpy().flatten()
    a_true = a_true[a_true > 0.2]
    plt.figure()
    plt.hist(a_pred, 100, color="red", alpha=0.5, density=True, label="Pred")
    plt.hist(a_true, 100, color="blue", alpha=0.5, density=True, label="True")
    plt.legend()
    plt.figure()
    diff = diff[(diff > 0.01) | (diff < -0.01)]
    plt.hist(diff.flatten(), bins=100)

    plt.figure()
    stats.probplot(diff.flatten(), dist="norm", plot=plt)
    plt.title("QQ-plot True-Pred")
    plt.show()
