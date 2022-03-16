import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import os
import pickle as pkl
from networkx import in_degree_centrality, out_degree_centrality, density
from networkx.convert_matrix import from_numpy_matrix
import networkx as nx
import pycountry_convert as pcc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.cluster import SpectralClustering
from matplotlib import cm
from scipy import stats
from src.modules.utils import zero_inflated_lognormal, compute_sigma, optimal_point, load_dataset
import src.modules.models
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
import tensorflow_probability as tfp


def in_degree_seq(A_true, A_pred=None, binary=False, save_path=None, time_from=1964):
    in_degrees = []
    out_degrees = []
    if A_pred is not None:
        for i, A_t, A_p in enumerate(zip(A_true, A_pred)):
            if len(A_true.shape == 3):
                A_t = A_t[0]
                A_p = A_p[0]
            if binary:
                A_t = np.asarray(A_t > 0, np.float32)
                A_p = np.asarray(A_p > 0, np.float32)
            in_deg_true = np.sum(A_t, -1)
            out_deg_true = np.sum(A_t, -2)
            in_deg_pred = np.sum(A_p, -1)
            out_deg_pred = np.sum(A_p, -2)
            in_degrees.append((in_deg_true, in_deg_pred))
            out_degrees.append((out_deg_true, out_deg_pred))

            if save_path is not None:
                fig, ax = plt.subplots(2, 1)
                ax[0].hist(in_deg_true, bins=10)
                ax[1].hist(out_deg_true, bins=10)
                ax[0].set_title(f"In Degree at time f{time_from + i}")
                ax[1].set_title(f"Out Degree at time f{time_from + i}")
                fig.suptitle(f"In Degrees of Dataset for time {time_from + i}")
                plt.savefig(os.path.join(save_path, f"{i}_true_deg"))
                plt.close()
                fig1, ax1 = plt.subplots(2, 1)
                ax1[0].hist(in_deg_pred, bins=10)
                ax1[1].hist(out_deg_pred, bins=10)
                ax1[0].set_title(f"In Degree at time f{time_from + i}")
                ax1[1].set_title(f"Out Degree at time f{time_from + i}")
                fig1.suptitle(f"In Degrees of Dataset for time {time_from + i}")
                plt.savefig(os.path.join(save_path, f"{i}_pred_deg"))
                plt.close()
    return in_degrees, out_degrees


def degrees_animation(in_degrees, out_degrees, time_from=1964, save_path=None):
    fig, ax = plt.subplots(2, 1)

    def degrees_animation_update(i):
        ax.cla()
        in_true = in_degrees[i][0]
        out_true = out_degrees[i][0]
        ax[0].hist(in_true, bins=10)
        ax[1].hist(out_true, bins=10)
        ax[0].set_title(f"In Degree at time f{time_from + i}")
        ax[1].set_title(f"Out Degree at time f{time_from + i}")

    anim = animation.FuncAnimation(degrees_animation_update, len(in_degrees), writer="pillow")
    if save_path is not None:
        anim.save(os.path.join(save_path, "true_degree_animation.gif"))
        plt.close()

    fig, ax = plt.subplots(2, 1)

    def degrees_animation_update2(i):
        ax.cla()
        in_true = in_degrees[i][1]
        out_true = out_degrees[i][1]
        ax[0].hist(in_true, bins=10)
        ax[1].hist(out_true, bins=10)
        ax[0].set_title(f"In Degree at time f{time_from + i}")
        ax[1].set_title(f"Out Degree at time f{time_from + i}")

    anim = animation.FuncAnimation(degrees_animation_update2, len(in_degrees), writer="pillow")
    if save_path is not None:
        anim.save(os.path.join(save_path, "pred_degree_animation.gif"))


def plot_cum_trade(A, baci=True, upto=60):
    if baci:
        df_iso = pd.read_csv(os.path.join(os.getcwd(), "Data", "countries_idx_name_iso3_baci.csv"), delimiter=";")
        with open(os.path.join(os.getcwd(), "Data", "baci_idx_to_countries.pkl"), "rb") as file:
            i2c = pkl.load(file)
    else:
        df_iso = pd.read_excel(
            os.path.join(os.getcwd(), "Comtrade", "Reference Table", "Comtrade Country Code and ISO list.xls")
        )
        df_iso = df_iso.rename(
            columns={"Country Code": "country_code", "Country Name, Abbreviation": "country_name_abbreviation",
                     "ISO2-digit Alpha": "iso_2digit_alpha"})
        with open(os.path.join(os.getcwd(), "Data", "idx_to_countries.pkl"), "rb") as file:
            i2c = pkl.load(file)

    sum_trade_in = np.sum(A, 1)
    sum_trade_out = np.sum(A, 1)
    sum_trade = sum_trade_in + sum_trade_out
    sc = np.argsort(sum_trade)[::-1][:upto]
    lc = [df_iso[df_iso["country_code"] == i2c[k]]["country_name_abbreviation"].values[0] for k in sc]
    plt.figure()
    plt.bar(np.arange(upto), np.sort(sum_trade)[::-1][:upto])
    plt.xticks(np.arange(upto), lc, rotation="vertical", fontsize=7)
    return sum_trade, lc


def plot_names(X, ax=None, baci=False, sizes=None):
    if baci:
        with open("A:/Users/Claudio/Documents/PROJECTS/Master-Thesis/Data/baci_idx_to_countries.pkl", "rb") as file:
            idx2country = pkl.load(file)
    else:
        with open("A:/Users/Claudio/Documents/PROJECTS/Master-Thesis/Data/idx_to_countries.pkl", "rb") as file:
            idx2country = pkl.load(file)
    with open(
            "A:/Users/Claudio/Documents/PROJECTS/Master-Thesis/Data/iso3_to_name.pkl",
            "rb",
    ) as file:
        iso3_to_name = pkl.load(file)
    iso3 = [idx2country[i] for i in range(len(X))]
    names = [iso3_to_name[int(i)] for i in iso3]
    if ax is None:
        fig, ax = plt.subplots()
        if sizes is not None:
            ax.scatter(X[:, 0], X[:, 1], s=sizes)
        else:
            ax.scatter(X[:, 0], X[:, 1])
    for c, (x, y) in zip(names, X[:, :2]):
        ax.text(x, y, f"{c}", size=7)


def plot_products(X, ax=None):
    with open(
            "A:\\Users\\Claudio\\Documents\\PROJECTS\\Master-Thesis\\Data\\idx_to_prod_SITC1.pkl",
            "rb",
    ) as file:
        idx2prod = pkl.load(file)

    if ax is None:
        plt.scatter(X[:, 0], X[:, 1])
        for p, (x, y) in zip(idx2prod.values(), X):
            plt.text(x, y, f"{p}", size=7)
    else:
        ax.scatter(X[:, 0], X[:, 1])
        for p, (x, y) in zip(idx2prod.values(), X):
            ax.text(x, y, f"{p}", size=7)


def load_sparse_tensor(sparse_tensor: tf.sparse.SparseTensor, baci=True):
    """
    sparse tensor of shape NxNxr or RxNxN
    """
    if baci:
        df_iso = pd.read_csv(os.path.join(os.getcwd(), "Data", "countries_idx_name_iso3_baci.csv"), delimiter=";")
        with open(os.path.join(os.getcwd(), "Data", "baci_countries_to_idx.pkl"), "rb") as file:
            c2i = pkl.load(file)
        i2c = {}
        for k, v in c2i.items():
            i2c[v] = k
    else:
        df_iso = pd.read_excel(
            os.path.join(os.getcwd(), "Comtrade", "Reference Table", "Comtrade Country Code and ISO list.xls")
        )
        df_iso = df_iso.rename(columns={"Country Code": "country_code", "ISO2-digit Alpha": "iso_2digit_alpha"})
        with open(os.path.join(os.getcwd(), "Data", "idx_to_countries.pkl"), "rb") as file:
            i2c = pkl.load(file)
    indices = sparse_tensor.indices.numpy()
    values = sparse_tensor.values.numpy()

    indices_i = []
    indices_j = []
    indices_k = []
    for k, i, j in indices:
        indices_i.append(df_iso[df_iso["country_code"] == i2c[i]]["iso_2digit_alpha"].values.tolist()[0])
        indices_j.append(df_iso[df_iso["country_code"] == i2c[j]]["iso_2digit_alpha"].values.tolist()[0])
        indices_k.append(k)

    df = pd.DataFrame({"i": indices_i, "j": indices_j, "r": indices_k, "v": values})
    G = nx.from_pandas_edgelist(
        df=df,
        source="i",
        target="j",
        edge_attr=["v"],
        edge_key="r",
        create_using=nx.MultiDiGraph(),
    )
    return G


def get_product_subgraph(G: nx.MultiGraph, prod_key: str) -> nx.Graph:
    edges = [(i, j, k) for i, j, k in G.edges if k == prod_key]
    G_sub = G.edge_subgraph(edges).copy()
    return G_sub


def draw_subgraph(
        G: nx.MultiDiGraph,
        prod_keys: [str],
        nodes: [str] = None,
        log_scale=True,
        normalize=True,
        quantile=0.10,
):
    with open(os.path.join(os.getcwd(), "Data", "iso2_long_lat.pkl"), "rb") as file:
        lat_long = pkl.load(file)
    radius = np.linspace(-0.1, -0.5, len(prod_keys))
    col_idx = np.arange(0, len(prod_keys))
    colors = plt.get_cmap("Set1")(col_idx)
    ax = plt.axes(projection=ccrs.PlateCarree())
    for i, key in enumerate(prod_keys):
        G_sub = get_product_subgraph(G, key)
        remove_nodes = []
        for name in G_sub.nodes:
            if nodes is not None:
                print(name)
                if name not in nodes:
                    remove_nodes.append(name)
                    continue
            try:
                i_pos = lat_long[name]
                G_sub.nodes[name]["pos"] = i_pos
            except Exception as e:
                print(e)
                remove_nodes.append(name)

        for name in remove_nodes:
            G_sub.remove_node(name)

        width = [G_sub.get_edge_data(i, j, k)["v"] for i, j, k in G_sub.edges]
        width = np.asarray(width)

        if log_scale:
            width = np.log(width)
        if normalize:
            width = (width - width.min()) / (width.max() - width.min()) + 0.5
        if quantile:
            q = np.quantile(a=width, q=quantile)

        for j, edge in enumerate(G_sub.edges):
            if quantile and (width[j] < q):
                continue
            if G_sub.nodes[0]["attr"]:
                weights = []
                for node in G_sub.nodes:
                    weights.append(G_sub[node]["attr"])
            else:
                weights = 10
            xy1 = G_sub.nodes[edge[0]]["pos"]
            xy2 = G_sub.nodes[edge[1]]["pos"]
            plt.scatter([xy1[0], xy2[0]], [xy1[1], xy2[1]], color="b", s=weights)
            ax.annotate(
                "",
                xy=xy2,
                xycoords="data",
                xytext=xy1,
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="->",
                    color=colors[i],
                    shrinkA=5,
                    shrinkB=5,
                    patchA=None,
                    patchB=None,
                    connectionstyle=f"arc3,rad={radius[i]}",
                    lw=width[j],
                ),
                transform=ccrs.Geodetic(),
            )

    ax.stock_img()
    ax.add_feature(cfeature.BORDERS, alpha=0.5, linestyle=":")
    ax.add_feature(cfeature.COASTLINE, alpha=0.5)


def inner_product_similarity(A: np.matrix):
    if len(A.shape) == 3:
        for a in A:
            g = np.asarray(a * 0.5 + 0.5 * a.T, dtype=np.bool)


def spectral_clustering(A):
    clusters = SpectralClustering(7, affinity="precomputed", n_init=20).fit_predict(A)
    return clusters


def link_density(A):
    g = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    return nx.classes.density(g)


def temporal_density(At, At_pred, baci=True, save_dir=None):
    ds = []
    ds_pred = []
    print(At.shape, At_pred.shape)
    for a, a_pred in zip(At, At_pred):
        d = link_density(a)
        ds.append(d)
        d_pred = link_density(a_pred)
        ds_pred.append(d_pred)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    p = ax.plot(ds)
    p_pred = ax.plot(ds_pred)
    ax.legend(*[p, p_pred], labels=["True Data", "Predicted Data"])
    if baci:
        plt.xticks(ticks=np.arange(At.shape[0]), labels=np.arange(1995, At.shape[0] + 1995), rotation=50)
    else:
        plt.xticks(ticks=np.arange(At.shape[0]), labels=np.arange(1964, At.shape[0] + 1964), rotation=50)
    ax.set_ylabel("Link Density")
    ax.set_xlabel("Years")
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "link_density.png"))
        plt.close()
    return ds, ds_pred


def relational_temporal_density(Atr, baci=True, save_dir=None, plot=False):
    dtr = []
    for ar in Atr:
        drs = temporal_density(ar)
        dtr.append(drs)
    dtr = np.asarray(dtr)  # T x R
    colors = cm.get_cmap("tab20")
    if plot:
        plt.figure()
        plots = []
        for r in range(Atr.shape[1]):
            pl, = plt.plot(dtr[:, r], color=colors(r))
            plots.append(pl)
        if baci:
            plt.xticks(ticks=np.arange(Atr.shape[0]), labels=np.arange(1995, Atr.shape[0] + 1995), rotation=45)
        else:
            plt.xticks(ticks=np.arange(Atr.shape[0]), labels=np.arange(1964, Atr.shape[0] + 1964), rotation=45)
        plt.ylabel("Link Density")
        plt.xlabel("Years")
        plt.legend(handles=iter(plots), labels=[f"Prod {i}" for i in range(Atr.shape[1])], bbox_to_anchor=(1.05, .5),
                   fancybox=True,
                   shadow=True, loc='center left')
        plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "link_density_relational.png"))
        plt.close()
    return dtr


def node_degree_evolution(stats_folder, by=5, save=True):
    df = pd.read_csv(os.path.join(stats_folder, "stats.csv"))
    df_details = pd.read_csv(os.path.join(stats_folder, "stats_details.csv"))
    years = df["time"].unique()
    per_year = len(years) // by
    cmap = cm.get_cmap("Blues")
    cmap_pred = cm.get_cmap("Reds")
    colors = cmap(np.linspace(0.2, 1, by))
    colors_pred = cmap_pred(np.linspace(0.2, 1, by))
    pos_deg = np.linspace(0, 90, 100)
    pos_clos = np.linspace(0, 1, 100)
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    plot_legends = []
    plot_legends1 = []
    for i in range(by):
        in_deg_true = df[(df["time"] == i * per_year + 1) & (df["true"] == True)]["in"].values
        kde_in_true = stats.gaussian_kde(in_deg_true).pdf(pos_deg)
        in_deg_pred = df[(df["time"] == i * per_year) & (df["true"] == False)]["in"].values
        kde_in_deg_pred = stats.gaussian_kde(in_deg_pred).pdf(pos_deg)
        out_deg_true = df[(df["time"] == i * per_year + 1) & (df["true"] == True)]["out"].values
        kde_out_deg_true = stats.gaussian_kde(out_deg_true).pdf(pos_deg)
        out_deg_pred = df[(df["time"] == i * per_year) & (df["true"] == False)]["out"].values
        kde_out_deg_pred = stats.gaussian_kde(out_deg_pred).pdf(pos_deg)
        clos_deg_true = df[(df["time"] == i * per_year + 1) & (df["true"] == True)]["clos"].values
        kde_clos_deg_true = stats.gaussian_kde(clos_deg_true).pdf(pos_clos)
        clos_deg_pred = df[(df["time"] == i * per_year) & (df["true"] == False)]["clos"].values
        kde_clos_deg_pred = stats.gaussian_kde(clos_deg_pred).pdf(pos_clos)
        axs[0, 0].plot(pos_deg, kde_in_true, color=colors[i], linestyle="--")
        axs[0, 0].set_title("Data Density")
        axs[0, 1].set_title("Predicted Density")
        axs[0, 0].xaxis.set_label_position('top')
        axs[0, 0].set_xlabel("In-Degree")
        axs[0, 1].plot(pos_deg, kde_in_deg_pred, color=colors_pred[i], linestyle="--")
        axs[0, 1].xaxis.set_label_position('top')
        axs[0, 1].set_xlabel("In-Degree")
        legend_ax0 = axs[1, 0].plot(pos_deg, kde_out_deg_true, color=colors[i], linestyle="--")
        plot_legends1.append(legend_ax0)
        axs[1, 0].xaxis.set_label_position('top')
        axs[1, 0].set_xlabel("Out-Degree")
        legend_ax, = axs[1, 1].plot(pos_deg, kde_out_deg_pred, color=colors_pred[i], linestyle="--")
        plot_legends.append(legend_ax)
        axs[1, 1].xaxis.set_label_position('top')
        axs[1, 1].set_xlabel("Out-Degree")
        axs[2, 0].plot(pos_clos, kde_clos_deg_true, color=colors[i], linestyle="--")
        axs[2, 0].xaxis.set_label_position('top')
        axs[2, 0].set_xlabel("Closeness Centrality")
        axs[2, 1].plot(pos_clos, kde_clos_deg_pred, color=colors_pred[i], linestyle="--")
        axs[2, 1].xaxis.set_label_position('top')
        axs[2, 1].set_xlabel("Closeness Centrality")
    if df_details["baci"].values[0]:
        from_time = 1996
    else:
        from_time = 1965
    axs[1, 1].legend(*plot_legends, labels=from_time + np.arange(0, by) * per_year)
    axs[0, 1].legend(*plot_legends1, labels=from_time + np.arange(0, by) * per_year)
    if save:
        plt.savefig(os.path.join(stats_folder, "in_out_clos.png"))
        plt.close()


def samples_from_model(model=None, data=None, weights_path=None, n_samples=10, n_samples_dropout=10, save=True):
    if model is None and weights_path is not None:
        specs = pd.read_csv(os.path.join(weights_path, "model_spec.csv"))
        model = src.modules.models.RNNGATLognormal(attn_heads=specs["attention"].values[0],
                                                   hidden_size=specs["channels"].values[0])

        t = specs["t"].values[0]
        n = specs["n"].values[0]
        r = specs["r"].values[0]
        if specs["baci"].values[0]:
            a_t, Xt, n, t = load_dataset(True, r=r, model_pred=True)
        else:
            a_t, Xt, n, t = load_dataset(False, r=r, model_pred=True)

        states = [None, None, None]
        # initialize model
        _ = model([Xt[0], a_t[0]], states=states)
        model.load_weights(os.path.join(weights_path, "gat_rnn"))
    elif data is None:
        raise Exception("Data cannot be None when weights_path is None")
    else:
        a_t = data[1]
        Xt = data[0]

    states = [None, None, None]
    states_history = []
    logits_history = []
    predictions = []
    for i in range(a_t.shape[0]):
        print(i)
        a = a_t[i]
        x = Xt[i]
        samples_drop = []
        logits_drop = []
        states_drop = []
        for i in range(n_samples_dropout):
            if n_samples_dropout == 1:
                training = False
            else:
                training = True
            logits, h_prime_p, h_prime_mu, h_prime_sigma = model([x, a], states=states, training=training)
            states = [h_prime_p, h_prime_mu, h_prime_sigma]
            states_np = [states[0].numpy(), states[1].numpy(), states[2].numpy()]
            sample = zero_inflated_lognormal(logits).sample(n_samples).numpy()
            sample = np.squeeze(sample)
            states_np = np.squeeze(states_np)
            states_drop.append(states_np)
            logits_drop.append(logits.numpy())
            samples_drop.append(sample)
        states_history.append(states_drop)
        logits_history.append(logits_drop)
        predictions.append(samples_drop)
    states_history = np.asarray(states_history)
    logits_history = np.asarray(logits_history)
    predictions = np.asarray(predictions)
    if save:
        with open(os.path.join(weights_path, "states_history.pkl"), "wb") as file:
            pkl.dump(states_history, file)
        with open(os.path.join(weights_path, "logits_history.pkl"), "wb") as file:
            pkl.dump(logits_history, file)
        with open(os.path.join(weights_path, "predictions.pkl"), "wb") as file:
            pkl.dump(predictions, file)
    return states_history, logits_history, predictions


def plot_uncertainty(samples_path=None, samples=None, logits=None, baci=True, r=6, save=False, save_path=None,
                     from_nodes=[80, 82], to_nodes=[2, 4], montecarlo=False):
    if samples_path is not None:
        save_path = samples_path
        with open(os.path.join(samples_path, "predictions.pkl"), "rb") as file:
            samples = pkl.load(file)
        with open(os.path.join(samples_path, "logits_history.pkl"), "rb") as file:
            logits = pkl.load(file)
        specs = pd.read_csv(os.path.join(samples_path, "model_spec.csv"))
        r = specs["r"].values[0]
        baci = specs["baci"].values[0]
    elif samples is None or logits is None:
        raise ValueError("Samples path cannot be None when samples or logits are None")

    if tf.is_tensor(samples) or tf.is_tensor(logits):
        samples = samples.numpy()
        logits = logits.numpy()
    a_t, t, n = load_dataset(baci, model_pred=False, r=r)
    a_t = a_t[:, from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]]
    a_t = tf.clip_by_value(tf.math.log(a_t), 0, 1e5).numpy()

    # Logits : T x n_drop x 1 x N x N x 3
    lognormal = zero_inflated_lognormal(logits)
    logits = tf.squeeze(logits).numpy()
    Mu = logits[..., 1:2].squeeze()
    P = tf.nn.sigmoid(logits[..., :1]).numpy().squeeze()
    V = compute_sigma(logits[..., -1:]).numpy().squeeze()
    mean_Mu = np.mean(Mu, axis=1)
    mean_V = np.mean(V, axis=1)
    mean_P = np.mean(P, axis=1)

    # Calculate tot mean and variance of samples by formula
    # T x n_drop x N x N -> T x N x N
    drop_mean_samples = tf.reduce_mean(lognormal.mean(), axis=1).numpy()
    drop_mean_samples = tf.clip_by_value(tf.math.log(drop_mean_samples), 0, 100).numpy()
    drop_variance_samples = tf.reduce_mean(lognormal.variance(), axis=1).numpy()
    drop_variance_samples = tf.clip_by_value(tf.math.log(tf.math.sqrt(drop_variance_samples)), 0, 100).numpy()

    # Samples :  T x n_drop x n_samples x N x N
    # Calculate tot mean and variance with Monte Carlo
    tot_sample_mean_samples = np.mean(samples, axis=(1, 2))
    tot_sample_mean_samples = tf.clip_by_value(tf.math.log(tot_sample_mean_samples), 0, 100).numpy()
    tot_sample_variance_samples = np.sqrt(np.var(samples, axis=(1, 2)))
    tot_sample_variance_samples = tf.clip_by_value(tf.math.log(tot_sample_variance_samples), 0, 100).numpy()

    # Calculate sample epistemic uncertainty over parameters
    if samples.shape[1] != 1:
        variance_Mu = np.var(Mu, axis=(1)).squeeze()
        variance_V = np.var(V, axis=(1)).squeeze()
        variance_P = np.var(P, axis=(1)).squeeze()

    edges = (from_nodes[1] - from_nodes[0]) * (to_nodes[1] - to_nodes[0])
    years = np.arange(logits.shape[0])
    if samples.shape[1] == 1:
        fig, axs = plt.subplots(3, 1)
        fig.suptitle("Deterministic Parameters")
        axs[0].plot(mean_Mu[:, from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(-1, edges))
        axs[0].set_title("Mean Parameter")
        axs[1].plot(mean_V[:, from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(-1, edges))
        axs[1].set_title("Standard Deviation Parameter")
        axs[2].plot(mean_P[:, from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(-1, edges))
        axs[2].set_title("Probability Parameter")
        axs[2].set_ylim(0, 1)
        if save:
            plt.savefig(os.path.join(save_path, "deterministic_parameters.png"))
            plt.close()
    else:
        fig, axs = plt.subplots(3, 1, figsize=(15, 10))
        Mu_ = Mu[..., from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(logits.shape[0],  # T
                                                                                    logits.shape[1],  # S_d
                                                                                    edges)  # E
        V_ = V[..., from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(logits.shape[0],  # T
                                                                                  logits.shape[1],  # S_d
                                                                                  edges)  # E
        P_ = P[..., from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(logits.shape[0],  # T
                                                                                  logits.shape[1],  # S_d
                                                                                  edges)  # E
        mean_Mu = mean_Mu[..., from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(logits.shape[0],  # T
                                                                                             edges)  # E
        mean_V = mean_V[..., from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(logits.shape[0],  # T
                                                                                           edges)  # E
        mean_P = mean_P[..., from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(logits.shape[0],  # T
                                                                                           edges)  # E
        variance_Mu = variance_Mu[..., from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(logits.shape[0],
                                                                                                     # T
                                                                                                     edges)  # E
        variance_Mu = np.sqrt(variance_Mu)
        variance_V = variance_V[..., from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(logits.shape[0],  # T
                                                                                                   edges)  # E
        variance_V = np.sqrt(variance_V)
        variance_P = variance_P[..., from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(logits.shape[0],  # T
                                                                                                   edges)  # E
        variance_P = np.sqrt(variance_P)
        lower_mu, upper_mu = mean_Mu - variance_Mu, mean_Mu + variance_Mu
        lower_p, upper_p = mean_P - variance_P, mean_P + variance_P
        lower_v, upper_v = mean_V - variance_V, mean_V + variance_V

        cmap = cm.get_cmap("Set1")
        for i in range(Mu_.shape[-1]):
            for j in range(Mu_.shape[1]):
                axs[0].scatter(years, Mu_[:, j, i], color=cmap(i), s=1)
                axs[1].scatter(years, V_[:, j, i], color=cmap(i), s=1)
                axs[2].scatter(years, P_[:, j, i], color=cmap(i), s=1)
            axs[0].fill_between(years, lower_mu[:, i], upper_mu[:, i], alpha=0.5, color=cmap(i))
            axs[0].plot(mean_Mu[:, i], lw=2, color=cmap(i))
            axs[1].fill_between(years, lower_v[:, i], upper_v[:, i], alpha=0.5, color=cmap(i))
            axs[1].plot(mean_V[:, i], lw=2, color=cmap(i))
            axs[2].fill_between(years, lower_p[:, i], upper_p[:, i], alpha=0.5, color=cmap(i))
            axs[2].plot(mean_P[:, i], lw=2, color=cmap(i))
        axs[0].set_title("Epistemic Uncertainty over the Mean")
        axs[1].set_title("Epistemic Uncertainty over the Variance")
        axs[2].set_title("Epistemic Uncertainty over the Probability")
        if save:
            plt.savefig(os.path.join(save_path, "epistemic_parameters.png"))
            plt.close()

    fig3, axs3 = plt.subplots(1, 1, figsize=(15, 10))
    fig3.suptitle("Sample Aleatoric + Epistemic Uncertainty")
    tot_mean_reshaped = tot_sample_mean_samples[:, from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(-1,
                                                                                                                 edges)
    a_t_reshaped = a_t.reshape(-1, edges)
    for i in range(edges):
        axs3.plot(tot_mean_reshaped[:, i], label="Predicted Edge Values", color=cmap(i))
    axs3.set_ylabel("Log of Mean and Standard Deviation")
    axs3.set_xlabel("Years")
    upper_mean = tot_sample_mean_samples + np.sqrt(tot_sample_variance_samples)
    lower_mean = tot_sample_mean_samples - np.sqrt(tot_sample_variance_samples)
    lower_mean = lower_mean[:, from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(-1, edges)
    upper_mean = upper_mean[:, from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(-1, edges)
    for i in range(edges):
        axs3.fill_between(years, lower_mean[:, i], upper_mean[:, i], color=cmap(i), alpha=0.5)
    for i in range(edges):
        axs3.plot(a_t_reshaped[:, i], "--", label="True Edge Values", color=cmap(i))
    if save:
        plt.savefig(os.path.join(save_path, "sample_aleatoric_uncertainty.png"))
        plt.close()
    if montecarlo:
        fig2, axs2 = plt.subplots(1, 1, figsize=(15, 10))
        if samples.shape[1] == 1:
            fig2.suptitle("Montecarlo Aleatoric Uncertainty")
        else:
            fig2.suptitle("Montecarlo Epistemic + Aleatoric Uncertainty")
        drop_mean_reshaped = drop_mean_samples[:, from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(-1, edges)
        for i in range(edges):
            axs2.plot(drop_mean_reshaped[:, i], color=cmap(i))
        axs2.set_ylabel("Log of Mean and Standard Deviation")
        axs2.set_xlabel("Years")
        upper_mean = drop_mean_samples + drop_variance_samples
        lower_mean = drop_mean_samples - drop_variance_samples
        lower_mean = lower_mean[:, from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(-1, edges)
        upper_mean = upper_mean[:, from_nodes[0]:from_nodes[1], to_nodes[0]:to_nodes[1]].reshape(-1, edges)
        for i in range(edges):
            axs2.fill_between(years, lower_mean[:, i], upper_mean[:, i], color=cmap(i), alpha=0.5)
        for i in range(edges):
            axs2.plot(a_t_reshaped[:, i], "--", label="True Edge Values", color=cmap(i))
        if save:
            plt.savefig(os.path.join(save_path, "aleatoric_uncertainty.png"))
            plt.close()


def plot_centrality(statistics_path, model_samples_path, centralities=["eig_in", "eig_out"],
                    save=False, by=20):
    df = pd.read_csv(os.path.join(statistics_path, "stats.csv"))
    stats_details = pd.read_csv(os.path.join(statistics_path, "stats_details.csv"))
    baci = stats_details["baci"].values[0]
    t = stats_details["t"].values[0]
    r = stats_details["r"].values[0]
    n = stats_details["n"].values[0]
    if baci:
        with open(os.path.join(os.getcwd(), "Data", "baci_sparse_price.pkl"), "rb") as file:
            data = pkl.load(file)
            data = tf.sparse.reorder(data)
            data_slice = tf.sparse.slice(data, (0, 0, 0, r), (t, n, n, 1))
            data_slice = tf.sparse.transpose(data_slice, perm=(0, 3, 1, 2))
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
        data_slice = tf.sparse.slice(data_sp, (1, r, 0, 0), (t, 1, n, n))

    A_true_t = tf.sparse.reduce_sum(data_slice, 1).numpy()
    with open(os.path.join(model_samples_path, "predictions.pkl"), "rb") as file:
        model_samples = pkl.load(file)

    A_pred_t = np.mean(model_samples, (1, 2))
    centrality_mapping = {"eig_in": "In-Eigenvecor", "eig_out": "Out-Eigenvecor", "in": "In-Degree",
                          "out": "Out-Degree", "clos": "Closeness"}
    for centrality in centralities:
        for i in range(0, t - 1, by):
            A_true = A_true_t[i]
            A_true = np.log(np.where(A_true == 0, 1, A_true))
            A_pred = A_pred_t[i]
            A_pred = np.log(np.where(A_pred == 0, 1, A_pred))
            eig_in_true = df[(df["time"] == t) & (df["true"] == True)][centrality].values
            eig_out_true = df[(df["time"] == t) & (df["true"] == True)][centrality].values
            eig_in_pred = df[(df["time"] == t) & (df["true"] == False)][centrality].values
            eig_out_pred = df[(df["time"] == t) & (df["true"] == False)][centrality].values
            g_true = nx.from_numpy_matrix(A_true, create_using=nx.DiGraph)
            g_pred = nx.from_numpy_matrix(A_pred, create_using=nx.DiGraph)
            layout_true = np.asarray(list(nx.drawing.spring_layout(g_true, k=2).values()))
            layout_pred = np.asarray(list(nx.drawing.spring_layout(g_pred, k=2).values()))
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
            fig.suptitle(f"{centrality} for Year {t}")
            axs[0, 0].scatter(layout_true[:, 0], layout_true[:, 1], s=eig_in_true)
            axs[0, 0].set_title(f"{centrality_mapping[centrality]} Centrality")
            axs[0, 0].set_ylabel("True Graph")
            axs[0, 1].scatter(layout_true[:, 0], layout_true[:, 1], s=eig_out_true)
            axs[0, 1].set_title(f"{centrality_mapping[centrality]} Centrality")
            axs[1, 0].scatter(layout_pred[:, 0], layout_pred[:, 1], s=eig_in_pred)
            axs[1, 0].set_ylabel("Predicted Graph")
            axs[1, 1].scatter(layout_pred[:, 0], layout_pred[:, 1], s=eig_out_pred)
            plot_names(layout_true, axs[0, 0], baci=baci)
            plot_names(layout_true, axs[0, 1], baci=baci)
            plot_names(layout_pred, axs[1, 0], baci=baci)
            plot_names(layout_pred, axs[1, 1], baci=baci)
            if save:
                plt.savefig(os.path.join(statistics_path, f"centrality_{centrality}_{i}.png"))
                plt.close()


def plot_eigen_centrality_timeseries(statistics_path):
    df = pd.read_csv(statistics_path)


def plot_evolution_of_edges(samples_path, by=4, save=True):
    with open(os.path.join(samples_path, "predictions.pkl"), "rb") as file:
        predictions = pkl.load(file)
    specs = pd.read_csv(os.path.join(samples_path, "model_spec.csv"))
    predictions = np.mean(predictions, axis=(1, 2))
    t = specs["t"].values[0]
    n = specs["n"].values[0]
    r = specs["r"].values[0]
    baci = specs["baci"].values[0]
    A_true_t, n, t = load_dataset(baci, r=r, model_pred=False)
    A_true_t = A_true_t[1:]
    A_true_t = tf.clip_by_value(tf.math.log(A_true_t), 0, 1e10).numpy()
    predictions = tf.clip_by_value(tf.math.log(predictions), 0, 1e10).numpy()
    per_year = predictions.shape[0] // by
    lim = np.linspace(0, 30, 100)
    cmap = cm.get_cmap("Blues")
    cmap_pred = cm.get_cmap("Reds")
    colors = cmap(np.linspace(0.2, 1, by))
    colors_pred = cmap_pred(np.linspace(0.2, 1, by))
    fig, axs = plt.subplots(1, 2)
    for i in range(by):
        at_true = A_true_t[i * per_year].flatten()
        at_true = at_true[at_true > 0.2]
        at = predictions[i * per_year].flatten()
        at = at[at > 1]
        kde_true = stats.gaussian_kde(at_true).pdf(lim)
        kde_pred = stats.gaussian_kde(at).pdf(lim)
        legend_ax0 = axs[0].plot(lim, kde_true, color=colors[i], linestyle="--")
        legend_ax = axs[1].plot(lim, kde_pred, color=colors_pred[i], linestyle="--")
    axs[0].set_title("Data Edge Density")
    axs[1].set_title("Predicted Edge Density")
    axs[0].xaxis.set_label_position('top')
    axs[1].xaxis.set_label_position('top')
    axs[0].set_ylabel("Kernel Density Estimate")
    axs[1].set_ylabel("Log of edge values")
    if baci:
        from_time = 1996
    else:
        from_time = 1965
    axs[1].legend(legend_ax, labels=from_time + np.arange(0, by) * per_year)
    axs[0].legend(legend_ax, labels=from_time + np.arange(0, by) * per_year)
    if save:
        plt.savefig(os.path.join(samples_path, "edge_distr_evolution.png"))
        plt.close()


def plot_roc_precision_confusion(a, logits, figures_dir, save=True):
    plt.figure()
    bin_true = tf.reshape(tf.where(a == 0, 0.0, 1.0), [-1]).numpy()
    p_pred = tf.reshape(tf.nn.sigmoid(logits[..., :1]), [-1]).numpy()
    fpr, tpr, thr = roc_curve(bin_true, p_pred, drop_intermediate=False)
    cmap = plt.cm.get_cmap("viridis")
    plt.scatter(fpr, tpr, color=cmap(thr), s=2)
    best = optimal_point(np.concatenate([fpr[:, None], tpr[:, None]], -1), type="roc")
    best_x = fpr[best]
    best_y = tpr[best]
    opt_thr = thr[best]
    plt.scatter(best_x, best_y, color="red", s=20)
    plt.title("ROC Curve \n Optimal Threshold:{:.2f}".format(round(opt_thr, 2)))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot((0, 1), (0, 1), color="black")
    plt.colorbar()
    if save:
        plt.savefig(os.path.join(figures_dir, "roc.png"))
        plt.close()

    plt.figure()
    prec, rec, thr = precision_recall_curve(bin_true, p_pred)
    plt.scatter(prec[:-1], rec[:-1], color=cmap(thr), s=2)
    opt = optimal_point(np.concatenate([prec[:, None], rec[:, None]], -1), "prec")
    opt_x = prec[opt]
    opt_y = rec[opt]
    thr_opt = thr[opt]
    plt.scatter(opt_x, opt_y, color="red", s=20)
    plt.title("Precision Recall Curve \n Optimal Threshold {:.2f}".format(round(thr_opt, 2)))
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.plot((0, 1), (1, 0), color="black")
    plt.colorbar()
    if save:
        plt.savefig(os.path.join(figures_dir, "prec_rec.png"))
        plt.close()

    fig, ax = plt.subplots(1, 1)
    conf = confusion_matrix(bin_true, p_pred > 0.5)
    conf = conf.reshape(2, 2)
    img = ax.imshow(conf, cmap="Blues")
    ax.set_xticklabels(["No Edge", "Edge"])
    ax.set_yticklabels(["No Edge", "Edge"])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.text(0, 0, str(conf[0, 0]), fontsize=13)
    ax.text(0, 1, str(conf[1, 0]), fontsize=13)
    ax.text(1, 0, str(conf[0, 1]), fontsize=13)
    ax.text(1, 1, str(conf[1, 1]), fontsize=13)

    plt.colorbar(img, ax=ax)
    if save:
        plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"))
        plt.close()

    plt.figure()
    true_pos = p_pred[bin_true == 1]
    false_negative = p_pred[bin_true == 0]
    plt.hist(true_pos, bins=100, density=True, alpha=0.4, color="green", label="positive preds")
    plt.hist(false_negative, bins=100, density=True, alpha=0.4, color="red", label="negative preds")
    plt.plot((0.5, 0.5), (0, 70), lw=1)
    plt.legend()
    if save:
        plt.savefig(os.path.join(figures_dir, "distr_preds.png"))
        plt.close()

    plt.figure()
    ece = tfp.stats.expected_calibration_error(num_bins=10, logits=tf.reshape(logits[..., :1][:500], (-1, 1)),
                                               labels_true=tf.reshape(bin_true.astype(int)[:500], (-1, 1)))
    cal_true, cal_pred = calibration_curve(bin_true, p_pred)
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.plot(cal_true, cal_pred, color="blue")
    plt.plot((0, 1), (0, 1), "-", color="black")
    plt.title(f"Calibration Curve \n ECE: {ece}")
    if save:
        plt.savefig(os.path.join(figures_dir, "calibration_curve.png"))
    plt.close()


def visualize_jaccard_matrix(stats_path, by=5):
    with open(os.path.join(stats_path, "jaccard_pred_true.pkl"), "rb") as file:
        jaccards = pkl.load(file)
    stats = pd.read_csv(os.path.join(stats_path, "stats_details.csv"))
    baci = stats["baci"].values[0]
    if baci:
        from_time = 1995
    else:
        from_time = 1964
    for i, jacc in enumerate(jaccards[1::15]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].set_title(f"Time {from_time + i * 15 + 1}", loc="right")
        ax[0].imshow(jacc[1], cmap="Blues")
        ax[0].set_xlabel("True adjacency Jaccard Coefficients")
        ax[1].set_xlabel("Predicted adjacency Jaccard Coefficients")
        a = ax[1].imshow(jacc[0], cmap="Reds")
        plt.savefig(os.path.join(stats_path, f"jaccard{i}.png"))
        plt.close()


def plot_predictions_graph(predictions_path, save=True, from_pred=[140, 144], to_pred=[153, 154]):
    with open(os.path.join(predictions_path, "predictions.pkl"), "rb") as file:
        predictions = pkl.load(file)
    specs = pd.read_csv(os.path.join(predictions_path, "model_spec.csv"))
    baci = specs["baci"].values[0]
    r = specs["r"].values[0]
    data, t, n = load_dataset(baci, r=r, model_pred=False)
    edges = (from_pred[1] - from_pred[0]) * (to_pred[1] - to_pred[0])
    data = data[:, from_pred[0]:from_pred[1], to_pred[0]:to_pred[1]].reshape(-1, edges)
    data = tf.clip_by_value(tf.math.log(data), 0, 1e10).numpy()
    predictions = np.mean(predictions, (1, 2))[:, from_pred[0]:from_pred[1], to_pred[0]:to_pred[1]].reshape(-1, edges)
    predictions = tf.clip_by_value(tf.math.log(predictions), 0, 1e10).numpy()
    cmap = cm.get_cmap("tab20")
    plt.figure(figsize=(15, 10))
    for i in range(edges):
        plt.plot(predictions[:, i], color=cmap(i))
        plt.plot(data[:, i], "--", color=cmap(i))
    if save:
        plt.savefig(os.path.join(predictions_path, "predictions.png"))
        plt.close()


def rank_by_centrality(stats_path):
    df = pd.read_csv(os.path.join(stats_path, "stats.csv"))
    df[""]["eig_in"]


if __name__ == "__main__":
    os.chdir("../../")
    path_stats = "A:\\Users\\Claudio\\Documents\\PROJECTS\\Master-Thesis\\src\\Tests\\Rnn Bilinear Lognormal\\2022-02-20T04_59_53.347279\\Figures\\Stats"
    path_weights = "A:\\Users\\Claudio\\Documents\\PROJECTS\\Master-Thesis\\src\\Tests\\Rnn Bilinear Lognormal\\2022-02-20T04_59_53.347279\\Weights"
    path_figures = "A:\\Users\\Claudio\\Documents\\PROJECTS\\Master-Thesis\\src\\Tests\\Rnn Bilinear Lognormal\\2022-02-20T04_59_53.347279\\Figures\\"
    baci = True
    r = 10
    # At = load_dataset(baci, r=r)

    samples_from_model(weights_path=path_weights)
    plot_uncertainty(path_weights)
    '''plot_evolution_of_edges(path_weights, by=10, save=True)
    node_degree_evolution(path_stats, by=10)
    plot_evolution_of_edges(path_weights, by=10, save=True)
    plot_centrality(path_stats, path_weights, save=True)
    ds = temporal_density(At, baci=baci, save_dir=path_figures)
    v, c = plot_cum_trade(At, baci=baci)'''
    plt.show()
