import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import pickle as pkl
from src.modules import models, losses, layers, utils
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, SpectralClustering
import matplotlib.pyplot as plt
import matplotlib as mplt
from matplotlib import cm
import scipy.stats as stats
from src.graph_data_loader import plot_names
import tqdm
import os
import argparse
from src.modules.losses import zero_inflated_lognormal_loss
from src.modules.utils import zero_inflated_lognormal
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--lognormal", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    normalize = args.normalize
    use_mask = args.mask
    lognormal = args.lognormal
    save = args.save

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
            os.path.join(os.getcwd(), "Data", "complete_data_final_transformed_no_duplicate.pkl"),
            "rb",
    ) as file:
        data_np = pkl.load(file)
    data_sp = tf.sparse.SparseTensor(
        data_np[:, :4], data_np[:, 4], np.max(data_np, 0)[:4]
    )
    data_sp = tf.sparse.reorder(data_sp)
    data_slice = tf.sparse.slice(data_sp, (50, 30, 0, 0), (1, 1, 174, 174))
    data_dense = tf.sparse.reduce_sum(data_slice, (0, 1))
    if not lognormal:
        data_dense = tf.clip_by_value(tf.math.log(data_dense), 0.0, 1e100)
        if normalize:
            data_dense = (data_dense - tf.reduce_mean(data_dense, (0, 1))) / (
                tf.math.reduce_std(data_dense, (0, 1))
            )
        A_bin = tf.where(data_dense > 0.0, 1.0, 0.0)
    A_log = data_dense

    if lognormal:
        model = models.Bilinear(15, qr=True, use_mask=True, use_variance=True, activation=None)
    else:
        model = models.Bilinear(5, qr=True, use_mask=use_mask, activation="relu")
    if lognormal:
        x0, x_m0, x0_var, a0, a_m0, a0_var = model(A_log)
    else:
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
            if lognormal:
                x, x_m, x_var, a, a_m, a_var = model(A_log)
                logits = tf.transpose([a_m, a, a_var], perm=(1, 2, 0))
                loss = zero_inflated_lognormal_loss(tf.expand_dims(A_log, -1), logits, reduce_axis=(0, 1, 2))
            elif use_mask:
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

    if lognormal:
        A_log = tf.clip_by_value(tf.math.log(A_log), 0.0, 1e12)
    else:
        A_log = tf.clip_by_value(data_dense, 0.0, 1e12)
    loss_hist = np.asarray(loss_hist)
    mplt.rcParams['figure.figsize'] = (15, 10)
    plt.plot(loss_hist)
    plt.title("Loss History")
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "loss.png"))
        plt.close()

    x0_tnse = TSNE(n_components=2, perplexity=80).fit_transform(x0.numpy())
    plot_names(x0_tnse)
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "embeddings_initial.png"))
        plt.close()

    plt.figure()
    plt.imshow(A_log.numpy())
    plt.colorbar()
    plt.title("True Adj")
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "adj_true.png"))
        plt.close()

    plt.figure()
    if lognormal:
        logn = zero_inflated_lognormal(logits).sample(1)
        a = tf.squeeze(logn)
        a = tf.clip_by_value(tf.math.log(a), 0, 1e100)
        plt.imshow(a.numpy())
    elif use_mask:
        plt.imshow((a * a_m).numpy())
    else:
        plt.imshow(a.numpy())
    plt.colorbar()
    plt.title("Pred Weighted Adj")
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "adj_pred.png"))
        plt.close()

    plt.figure()
    diff = a.numpy() - A_log.numpy()
    plt.imshow(diff)
    plt.colorbar()
    plt.title("Difference Weighted - True Adj")
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "abs_error.png"))
        plt.close()

    scale = tf.math.maximum(
        k.backend.softplus(logits[..., 2:]),
        tf.math.sqrt(k.backend.epsilon()))

    plt.figure()
    plt.imshow(scale.numpy().squeeze())
    plt.colorbar()
    plt.title("Variance")
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "adj_variance.png"))
        plt.close()

    x_tnse = TSNE(n_components=2, perplexity=80).fit_transform(x.numpy())
    clustering = SpectralClustering(n_clusters=10, affinity="nearest_neighbors").fit(
        x_tnse
    )
    labels = clustering.labels_
    colormap = cm.get_cmap("Set1")
    colors = colormap(labels)
    fig, ax = plt.subplots()
    ax.scatter(x_tnse[:, 0], x_tnse[:, 1], color=colors)
    ax.set_title("Mean Embedding")
    plot_names(x_tnse, ax)
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "embeddings_mean.png"))
        plt.close()

    x_tnse_var = TSNE(n_components=2, perplexity=80).fit_transform(x_var.numpy())
    clustering = SpectralClustering(n_clusters=10, affinity="nearest_neighbors").fit(
        x_tnse_var
    )
    labels = clustering.labels_
    colormap = cm.get_cmap("Set1")
    colors = colormap(labels)
    fig1, ax1 = plt.subplots()
    ax1.scatter(x_tnse_var[:, 0], x_tnse_var[:, 1], color=colors)
    ax1.set_title("Variance Embedding")
    plot_names(x_tnse_var, ax1)
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "embeddings_var.png"))
        plt.close()

    if lognormal:
        a_pred = a.numpy().flatten()
        a_pred = a_pred[a_pred > 0.2]
    elif use_mask:
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
    if save:
        plt.savefig(
            os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "edge_distr.png"))
        plt.close()

    plt.figure()
    diff = diff[(diff > 0.01) | (diff < -0.01)]
    plt.hist(diff.flatten(), bins=100)
    if save:
        plt.savefig(
            os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "true_pred_edge_distr.png"))
        plt.close()

    plt.figure()
    stats.probplot(diff.flatten(), dist="norm", plot=plt)
    plt.title("QQ-plot True-Pred")
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "error_distr.png"))
        plt.close()

    R = model.layers[0].trainable_weights[0]
    X = model.layers[0].trainable_weights[1]
    plt.figure()
    plt.imshow(R.numpy())
    plt.colorbar()
    plt.title("R Coefficient")
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "coeff_R.png"))
        plt.close()

    plt.figure()
    img = plt.imshow(X.numpy().T)
    plt.colorbar(img, fraction=0.0046, pad=0.04)
    plt.title("X embeddings")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "x_embs.png"))
        plt.close()

    plt.figure()
    bin_true = tf.reshape(tf.where(A_log == 0, 0.0, 1.0), [-1]).numpy()
    p_pred = tf.reshape(tf.nn.sigmoid(logits[..., :1]), [-1]).numpy()
    fpr, tpr, thr = roc_curve(bin_true, p_pred, drop_intermediate=False)
    cmap = cm.get_cmap("viridis")
    plt.scatter(fpr, tpr, color=cmap(thr), s=2)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.colorbar()
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "roc.png"))
        plt.close()

    plt.figure()
    prec, rec, thr = precision_recall_curve(bin_true, p_pred)
    plt.scatter(prec[:-1], rec[:-1], color=cmap(thr), s=2)
    plt.title("Precision Recall Curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.colorbar()
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "prec_rec.png"))
        plt.close()

    plt.figure()
    true_pos = p_pred[bin_true == 1]
    false_negative = p_pred[bin_true == 0]
    plt.hist(true_pos, bins=100, density=True, alpha=0.4, color="green", label="positive preds")
    plt.hist(false_negative, bins=100, density=True, alpha=0.4, color="red", label="negative preds")
    plt.plot((0.5,0.5), (0,70), lw=1)
    plt.legend()
    if save:
        plt.savefig(os.path.join(os.getcwd(), "src", "Tests", "Figures", "Bilinear Lognormal", "distr_preds.png"))
        plt.close()

    plt.show()
