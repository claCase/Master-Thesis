import tensorflow as tf
import numpy as np
from tensorflow.keras import layers as l
from tensorflow.keras import models as m
from src.Tests.rnn_test import GRUGATLognormal
import pickle as pkl
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
# from tensorflow.keras.backend import function
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("-folder", type=str)
    args = parser.parse_args()
    fake = args.fake
    save = args.save
    folder = args.folder
    t = 55
    if fake:
        range_x = np.linspace(0, t, 100)
        trajectories = []
        for i in range(20):
            fake_data = np.stack([np.cos(range_x)[..., None],
                                  np.cos(range_x * np.pi * 5 / (1 + i))[..., None],
                                  np.cos(range_x * np.pi * 20 / (i + 1))[..., None]], axis=-1)
            trajectories.append(fake_data)
        trajectories = np.stack(trajectories, axis=1).swapaxes(0, 1).squeeze()

    else:
        os.chdir("../../")
        weights_path = os.path.join(os.getcwd(), "src", "Tests", "Rnn Bilinear Lognormal", folder,
                                    "Weights", "gat_rnn")
        with open(
                os.path.join(os.getcwd(), "Data", "complete_data_final_transformed_no_duplicate.pkl"),
                "rb",
        ) as file:
            data_np = pkl.load(file)
        data_sp = tf.sparse.SparseTensor(
            data_np[:, :4], data_np[:, 4], np.max(data_np, 0)[:4] + 1
        )
        data_sp = tf.sparse.reorder(data_sp)
        data_slice = tf.sparse.slice(data_sp, (0, 31, 0, 0), (t, 1, 174, 174))
        data_dense = tf.sparse.reduce_sum(data_slice, 1)
        a_train = tf.expand_dims(data_dense, 1)
        Xt = [np.eye(174)] * t
        x = tf.expand_dims(Xt, 1)
        states = [None, None, None]

    attn_heads = 10
    channels = 10
    hidden_size = attn_heads * channels // 2

    if fake:
        hidden_size = 3
    else:
        hidden_size = 50

    input = l.Input(shape=(None, hidden_size), name="input")
    cnn = l.Conv1D(filters=10, kernel_size=15, strides=1, padding="same", activation="tanh", name="hidden")(input)
    cnn = l.Dropout(0.2)(cnn)
    cnn = l.LayerNormalization()(cnn)
    cnn = l.Conv1D(filters=5, kernel_size=10, strides=1, padding="same", activation="tanh", name="hidden2")(cnn)
    cnn = l.Dropout(0.2)(cnn)
    cnn = l.LayerNormalization()(cnn)
    o = l.Conv1D(filters=hidden_size, kernel_size=4, strides=1, padding="same", activation=None, name="output")(cnn)
    model = m.Model(input, o)
    model.compile(optimizer="Adam", loss="mean_squared_error")
    graph_model = GRUGATLognormal(attn_heads=attn_heads, hidden_size=channels)

    if not fake:
        h_primes_mu = []
        h_primes_p = []
        h_primes_sigma = []
        for i in range(t):
            x_train_t = x[i]
            a_train_t = a_train[i]
            logits, h_prime_p, h_prime_mu, h_prime_sigma = graph_model([x_train_t, a_train_t], states=states,
                                                                       training=False)
            states = [h_prime_p, h_prime_mu, h_prime_sigma]
            h_primes_mu.append(h_prime_mu.numpy())
            h_primes_sigma.append(h_prime_sigma.numpy())
            h_primes_p.append(h_prime_p.numpy())

        h_primes_p = np.asarray(h_primes_p).swapaxes(0, 2).squeeze()
        h_primes_sigma = np.asarray(h_primes_sigma).swapaxes(0, 2).squeeze()
        h_primes_mu = np.asarray(h_primes_mu).swapaxes(0, 2).squeeze()
        model.fit(h_primes_mu, h_primes_mu, epochs=150)
        _ = graph_model([x[0], a_train[0]], states=states)
        graph_model.load_weights(weights_path)

    else:
        model.fit(trajectories, trajectories, epochs=150)

    intermediate = m.Model(model.input, model.get_layer("hidden2").output)
    if fake:
        preds = intermediate.predict(trajectories)
    else:
        preds = intermediate.predict(h_primes_mu)
    x_proj = PCA(n_components=2).fit_transform(preds.reshape(-1, 5))
    # x_proj = x_proj.reshape(174, -1, 2)
    if fake:
        x_proj = x_proj.reshape((*trajectories.shape[:-1], 2))  # N x T x d
    else:
        x_proj = x_proj.reshape((*h_primes_mu.shape[:-1], 2))  # N x T x d

    save_dir = os.path.join(os.getcwd(), "src", "Tests", "Rnn Bilinear Lognormal", folder, "Figures")
    counter = np.arange(x_proj.shape[1])
    counter = np.minimum(counter, 2)
    map = cm.get_cmap("tab20")
    fig, ax = plt.subplots(1, 1)
    min_xy = np.min(x_proj, axis=(0, 1))
    max_xy = np.max(x_proj, axis=(0, 1))


    def update(t):
        ax.cla()
        for i in range(x_proj.shape[0]):
            ax.plot(x_proj[i, t - counter[t]:t, 0], x_proj[i, t - counter[t]:t, 1], color=map(i % 20), alpha=t / 55)
        ax.set_title(f"Time {t + 1964}")
        ax.set_xlim(min_xy[0], max_xy[0])
        ax.set_ylim(min_xy[1], max_xy[1])


    anim1 = animation.FuncAnimation(fig, update, frames=t, repeat=True)
    if save:
        anim1.save(os.path.join(save_dir, "time_embedding.gif"), writer="pillow")
        plt.close()

    fig2, ax2 = plt.subplots(1, 1)
    window = 4
    for i in range(0, x_proj.shape[1] - window, window):
        ax2.scatter(x_proj[:, i:i + window, 0], x_proj[:, i:i + window, 1], color=map((i + 1) % 20), s=2)
    ax2.set_xlim(min_xy[0], max_xy[0])
    ax2.set_ylim(min_xy[1], max_xy[1])
    if save:
        plt.savefig(os.path.join(save_dir, "static_time_embedding.png"))

    plt.figure()
    x_proj_tot = x_proj.reshape(-1, 2)
    plt.scatter(x_proj_tot[:, 0], x_proj_tot[:, 1], s=2)
    plt.xlim(min_xy[0], max_xy[0])
    plt.ylim(min_xy[1], max_xy[1])
    if save:
        plt.savefig(os.path.join(save_dir, "all_time_embedding.png"))
    plt.show()
