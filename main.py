import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import layers as l
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, _config_for_enable_caching_device, \
    _caching_device
from tensorflow.keras import models as m
from tensorflow.keras import activations as a
from src.modules import utils, losses, models
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from src.modules.utils import compute_sigma, optimal_point, sample_from_logits, mean_accuracy_score, load_dataset
from tensorflow.keras.metrics import mean_squared_error, mean_absolute_error
from src.modules.plotter import plot_roc_precision_confusion, samples_from_model, plot_uncertainty, \
    node_degree_evolution, plot_evolution_of_edges, plot_centrality, temporal_density, plot_cum_trade, \
    visualize_jaccard_matrix, plot_predictions_graph, in_out_degree, edge_distribution_animation
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, explained_variance_score
import pandas as pd
import datetime
from matplotlib import cm
import pickle as pkl
import networkx as nx
from networkx.algorithms.link_prediction import jaccard_coefficient
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", action="store_true")
    parser.add_argument("--gat", action="store_true")
    parser.add_argument("--stateful", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--logits", action="store_true")
    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--baci", action="store_true")
    parser.add_argument("--epistemic", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--unroll", action="store_true")
    parser.add_argument("--r", type=int, default=0)
    parser.add_argument("--r2", type=int, default=8)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--n", type=int, default=175)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    keras = args.k
    gat = args.gat
    stateful = args.stateful
    dropout = args.dropout
    logits = args.logits
    seq_len = args.seq_len
    features = args.features
    baci = args.baci
    epistemic = args.epistemic
    epochs = args.epochs
    unroll = args.unroll
    test_split = args.test_split
    symmetric = args.symmetric
    r = args.r
    r2 = args.r2
    n = args.n
    save = args.save
    from_nodes = [92, 93]
    to_nodes = [150, 152]
    if baci:
        from_year = 1995
    else:
        from_year = 1964

    if save:
        today = datetime.datetime.now().isoformat().replace(":", "_")
        weights_dir = os.path.join(os.getcwd(), "src", "Tests", "Nested Rnn", today, "Weights")
        figures_dir = os.path.join(os.getcwd(), "src", "Tests", "Nested Rnn", today, "Figures")
        os.makedirs(weights_dir)
        os.makedirs(figures_dir)
        print(f"Making {figures_dir}, {weights_dir} ")
    else:
        weights_dir = None
        figures_dir = None
    if gat:
        a, x, N, t = load_dataset(baci=baci, features=features, model_pred=True, r=r, r2=r2, n=n)
        f = x.shape[-1]
        a = tf.transpose(a, perm=(1, 0, 2, 3))  # BxTxNxN
        x = tf.transpose(x, perm=(1, 0, 2, 3))  # BxTxNxf
        zero_diag_in = 1 - np.asarray([np.eye(a.shape[-1])] * a.shape[0])
        zero_diag_in = np.expand_dims(zero_diag_in, 0)
        train_mask_in = np.random.choice((0, 1), p=(test_split, 1 - test_split), size=(*a.shape,)).astype(np.float32)  # BxTxNxN
        train_mask_in = train_mask_in * zero_diag_in
        train_mask_in = train_mask_in.astype(np.float32)
        test_mask_in = 1 - train_mask_in
        train_mask_out = tf.transpose([train_mask_in] * 3, perm=(1, 2, 3, 4, 0))
        test_mask_out = 1 - train_mask_out

        if unroll:
            input_1 = tf.keras.Input((a.shape[1] - 1, N, f))
            input_2 = tf.keras.Input((a.shape[1] - 1, N, N))
        else:
            input_1 = tf.keras.Input((None, N, f))
            input_2 = tf.keras.Input((None, N, N))
        if logits:
            if stateful:
                batches_a = np.empty(shape=(1, seq_len, N, N))
                batches_a_mask_train_in = np.empty(shape=(1, seq_len, N, N))
                batches_a_mask_train_out = np.empty(shape=(1, seq_len, N, N, 3))
                batches_a_mask_test_in = np.empty(shape=(1, seq_len, N, N))
                batches_a_mask_test_out = np.empty(shape=(1, seq_len, N, N, 3))
                batches_x = np.empty(shape=(1, seq_len, N, f))
                for i in range(0, t - seq_len, seq_len):
                    batches_a = np.concatenate([batches_a, a[:, i:i + seq_len, :]], axis=0)
                    batches_x = np.concatenate([batches_x, x[:, i:i + seq_len, :]], axis=0)
                    batches_a_mask_train_in = np.concatenate(
                        [batches_a_mask_train_in, train_mask_in[:, i:i + seq_len, :]], axis=0)
                    batches_a_mask_train_out = np.concatenate(
                        [batches_a_mask_train_out, train_mask_out[:, i:i + seq_len, :]], axis=0)
                    batches_a_mask_test_in = np.concatenate(
                        [batches_a_mask_test_in, test_mask_in[:, i:i + seq_len, :]], axis=0)
                    batches_a_mask_test_out = np.concatenate(
                        [batches_a_mask_test_out, test_mask_out[:, i:i + seq_len, :]], axis=0)
                a = np.concatenate([batches_a, a[:1, t - seq_len:, :]], axis=0)[1:]
                x = np.concatenate([batches_x, x[:1, t - seq_len:, :]], axis=0)[1:]
                train_mask_in = np.concatenate(
                    [batches_a_mask_train_in, train_mask_in[:, t - seq_len:, :]], axis=0)[1:]
                train_mask_out = np.concatenate(
                    [batches_a_mask_train_out, train_mask_out[:, t - seq_len:, :]], axis=0)[1:]
                test_mask_in = np.concatenate(
                    [batches_a_mask_test_in, test_mask_in[:, t - seq_len:, :]], axis=0)[1:]
                test_mask_out = np.concatenate(
                    [batches_a_mask_test_out, test_mask_out[:, t - seq_len:, :]], axis=0)[1:]
                train_mask_in = tf.constant(train_mask_in, dtype=tf.float32)
                train_mask_out = tf.constant(train_mask_out, dtype=tf.float32)
                test_mask_in = tf.constant(test_mask_in, dtype=tf.float32)
                test_mask_out = tf.constant(test_mask_out, dtype=tf.float32)
                if unroll:
                    input_1 = tf.keras.Input((a.shape[1] - 1, N, f), batch_size=a.shape[0])
                    input_2 = tf.keras.Input((a.shape[1] - 1, N, N), batch_size=a.shape[0])
                else:
                    input_1 = tf.keras.Input((None, N, f), batch_size=a.shape[0])
                    input_2 = tf.keras.Input((None, N, N), batch_size=a.shape[0])

            cell = models.RecurrentEncoderDecoder(nodes=N,
                                                  features=f,
                                                  channels=5,
                                                  attention_heads=10,
                                                  hidden_size=25,
                                                  dropout_adj=0,
                                                  dropout=dropout,
                                                  recurrent_dropout=dropout,
                                                  symmetric=symmetric,
                                                  regularizer=None,
                                                  qr=False,
                                                  gnn=True,
                                                  gnn_h=False)
            rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=False, stateful=stateful, unroll=unroll)
            outputs = rnn((input_1, input_2), training=True)
            model = tf.keras.models.Model([input_1, input_2], outputs)


            def loss_function(mask_logits, mask_true):
                def lognormal_loss(true, logits):
                    return tf.reduce_sum(
                        losses.zero_inflated_lognormal_loss(tf.expand_dims(true, -1) * tf.expand_dims(mask_true, -1),
                                                            logits * mask_logits,
                                                            reduce_axis=(-1, -2, -3)
                                                            ),
                        axis=1)

                return lognormal_loss


            model.compile(optimizer="rmsprop",
                          loss=loss_function(train_mask_out[:, 1:, :], train_mask_in[:, :-1, :]),
                          metrics=loss_function(test_mask_out[:, 1:, :], test_mask_in[:, :-1, :]))
            history = model.fit(x=[x[:, :-1, :], a[:, :-1, :]],
                                y=a[:, 1:, :],
                                epochs=epochs, verbose=1)
            if save:
                model.save(os.path.join(weights_dir, "model"))
            history = history.history["loss"]
            plt.plot(history)
            if save:
                plt.savefig(os.path.join(figures_dir, "loss.png"))
                plt.close()
            a_plot = tf.reshape(a, (-1, N, N)).numpy()
            if unroll:
                a = a[:, :-1, :]
                x = x[:, :-1, :]
            if epistemic:
                logits = []
                samples = []
                for i in range(15):
                    logit = model.predict([x, a])
                    logit = tf.squeeze(logit)
                    if stateful:
                        logit = tf.reshape(logit, shape=(-1, *logit.shape[-3:]))
                    ln = utils.zero_inflated_lognormal(logit)
                    sample = ln.sample(10).numpy().squeeze()
                    logits.append(logit)
                    samples.append(sample)
                epistemic_samples = np.asarray(samples)
                epistemic_samples = np.transpose(epistemic_samples, axes=(2, 0, 1, 3, 4))
                logits = np.swapaxes(logits, 0, 1)
                plot_uncertainty(samples=epistemic_samples,
                                 logits=logits,
                                 baci=baci,
                                 from_nodes=from_nodes,
                                 to_nodes=to_nodes,
                                 a_true=a_plot, save_path=figures_dir, save=save)
                plot_uncertainty(samples=epistemic_samples,
                                 logits=logits,
                                 baci=baci,
                                 from_nodes=from_nodes,
                                 to_nodes=to_nodes,
                                 inverse=True,
                                 a_true=a_plot, save_path=figures_dir, save=save)
                logits_plot = tf.reduce_mean(logits, axis=1)
            else:
                logits = model([x, a])
                logits = tf.reshape(logits, (-1, N, N, 3))
                ln = utils.zero_inflated_lognormal(logits)
                samples = ln.sample(20)
                epistemic_samples = tf.expand_dims(tf.transpose(samples, perm=(1, 0, 2, 3)), 1)
                samples = tf.reduce_mean(samples, axis=0).numpy()
                logits_plot = tf.expand_dims(logits, 1)

                plot_uncertainty(samples=epistemic_samples,
                                 logits=logits_plot,
                                 baci=baci,
                                 from_nodes=to_nodes,
                                 to_nodes=from_nodes,
                                 a_true=a_plot, save_path=figures_dir, save=save)
                plot_uncertainty(samples=epistemic_samples,
                                 logits=logits_plot,
                                 baci=baci,
                                 from_nodes=to_nodes,
                                 to_nodes=from_nodes,
                                 inverse=True,
                                 a_true=a_plot, save_path=figures_dir, save=save)
            if save:
                with open(os.path.join(figures_dir, "epistemic_samples.pkl"), "wb") as file:
                    pkl.dump(epistemic_samples, file)
                with open(os.path.join(figures_dir, "epistemic_logits.pkl"), "wb") as file:
                    pkl.dump(logits, file)

            samples = tf.clip_by_value(tf.math.log(epistemic_samples), 0, 1e10)
            samples = np.mean(samples, (1, 2))
            with open(os.path.join(weights_dir, "predictions.pkl"), "wb") as file:
                pkl.dump(samples, file)
            a = tf.reshape(a, (-1, N, N))
            a = tf.clip_by_value(tf.math.log(a), 0, 1e10).numpy()
            stochastic_sample = utils.zero_inflated_lognormal(logits_plot).sample(1).numpy().squeeze()
            stochastic_sample = tf.clip_by_value(tf.math.log(stochastic_sample), 0, 1000).numpy()
            fig_logits, ax_logits = plt.subplots(1, 3, figsize=(10,10))


            def update_p(i):
                fig_logits.suptitle(f"Time {i}")
                ax_logits[0].cla()
                ax_logits[1].cla()
                ax_logits[2].cla()
                ax_logits[0].imshow(tf.nn.sigmoid(logits_plot[i, :, :, 0]).numpy())
                ax_logits[1].imshow(logits_plot[i, :, :, 1].numpy())
                ax_logits[2].imshow(compute_sigma(logits_plot[i, :, :, 2]).numpy())
                ax_logits[0].set_title("P")
                ax_logits[1].set_title("Mu")
                ax_logits[2].set_title("Sigma")

            anim_p = animation.FuncAnimation(fig_logits, update_p, logits.shape[0] - 1, repeat=True)
            if save:
                anim_p.save(os.path.join(figures_dir, "parameters.gif"), writer="pillow")
                plt.close()

            fig, axs = plt.subplots(1, 2, figsize=(10,10))
            axs[0].set_title("Pred")
            axs[1].set_title("True")

            def update(i):
                fig.suptitle(f"Time {i}")
                axs[0].cla()
                axs[1].cla()
                axs[0].imshow(stochastic_sample[i])
                axs[1].imshow(a[i])


            anim = animation.FuncAnimation(fig, update, samples.shape[0] - 2, repeat=True)
            if save:
                anim.save(os.path.join(figures_dir, "sample_avg.gif"), writer="pillow")
                plt.close()

            mean_acc,mean_rmse,mean_abs,exp_var = [],[],[],[]
            for t in range(a.shape[0] - 1):
                ma = np.clip(mean_accuracy_score(a[t+1], samples[t]), -10, 1)
                mean_acc.append(ma)
                msq = np.clip(mean_squared_error(a[t+1].reshape(-1), samples[t].reshape(-1)), 0, 100)
                mean_rmse.append(msq)
                mabs = np.clip(mean_absolute_error(a[t+1].reshape(-1), samples[t].reshape(-1)), 0, 100)
                mean_abs.append(mabs)
                evar = explained_variance_score(a[t+1].reshape(-1), samples[t].reshape(-1))
                exp_var.append(evar)


            metrics = {"Mean Accuracy": mean_acc, "RMSE": mean_rmse, "MAE": mean_abs, "Explained Variance": exp_var}
            for metric in metrics.keys():
                plt.figure()
                plt.plot(metrics[metric])
                plt.title(f"{metric}")
                plt.xlabel("Years")
                plt.xticks(ticks=np.arange(len(metrics[metric])),
                           labels=np.arange(from_year, from_year + len(metrics[metric])),
                           rotation=45)
                if save:
                    plt.savefig(os.path.join(figures_dir, f"{metric}_per_epoch.png"))
                    plt.close()
            if save:
                df = pd.DataFrame(metrics)
                df.to_csv(os.path.join(figures_dir, "training_metrics.pkl"))

            values = {"time": [], "in": [], "out": [], "clos": [], "eig_in": [], "eig_out": [], "true": [], "i": []}
            jaccards = []
            for i, (true_data, sample) in enumerate(zip(a, stochastic_sample)):
                sample = np.where(sample<1, 0, sample)
                g = nx.from_numpy_matrix(sample, create_using=nx.DiGraph)
                in_deg = dict(g.in_degree).values()
                out_deg = dict(g.out_degree).values()
                closeness = dict(nx.closeness_centrality(g)).values()
                eigen_centr_in = dict(nx.eigenvector_centrality(g)).values()
                eigen_centr_out = dict(nx.eigenvector_centrality(g.reverse())).values()
                jacc = jaccard_coefficient(g.to_undirected())
                #lapl = SpectralClustering(8, affinity="precomputed").fit_predict(nx.to_numpy_matrix(g.to_undirected()))
                ij, v = [], []
                for ii, jj, p in jacc:
                    ij.append((ii, jj))
                    v.append(p)
                jacc_sp = tf.sparse.SparseTensor(ij, v, dense_shape=sample.shape)
                jacc_sp = tf.sparse.reorder(jacc_sp)
                jacc_sp = tf.sparse.to_dense(jacc_sp).numpy()
                time = np.ones(len(in_deg)) * i
                idx = np.arange(len(in_deg))
                true = [False] * len(in_deg)
                values["in"].extend(in_deg)
                values["out"].extend(out_deg)
                values["clos"].extend(closeness)
                values["eig_in"].extend(eigen_centr_in)
                values["eig_out"].extend(eigen_centr_out)
                #values["lapl"].extend(lapl)
                values["true"].extend(true)
                values["i"].extend(idx)
                values["time"].extend(time)

                g2 = nx.from_numpy_matrix(true_data, create_using=nx.DiGraph)
                in_deg_true = dict(g2.in_degree).values()
                out_deg_true = dict(g2.out_degree).values()
                closeness_true = dict(nx.closeness_centrality(g2)).values()
                eigen_centr_in_true = dict(nx.eigenvector_centrality(g2)).values()
                eigen_centr_out_true = dict(nx.eigenvector_centrality(g2.reverse())).values()
                #lapl_true = SpectralClustering(8, affinity="precomputed").fit_predict(nx.to_numpy_matrix(g2.to_undirected()))
                jacc_true = jaccard_coefficient(g2.to_undirected())
                ij_true, v_true = [], []
                for ii, jj, p in jacc_true:
                    ij_true.append((ii, jj))
                    v_true.append(p)
                jacc_sp_true = tf.sparse.SparseTensor(ij_true, v_true, dense_shape=sample.shape)
                jacc_sp_true = tf.sparse.reorder(jacc_sp_true)
                jacc_sp_true = tf.sparse.to_dense(jacc_sp_true).numpy()
                jacc = np.concatenate([jacc_sp[None, :], jacc_sp_true[None, :]], 0)
                jaccards.append(jacc)
                true = [True] * len(in_deg)
                values["in"].extend(in_deg_true)
                values["out"].extend(out_deg_true)
                values["clos"].extend(closeness_true)
                values["eig_in"].extend(eigen_centr_in_true)
                values["eig_out"].extend(eigen_centr_out_true)
                #values["lapl"].extend(lapl_true)
                values["true"].extend(true)
                values["i"].extend(idx)
                values["time"].extend(time)

            if save:
                stats_dir = os.path.join(figures_dir, "Stats")
                os.makedirs(stats_dir)
                df_stats = pd.DataFrame(values)
                df_stats.to_csv(os.path.join(stats_dir, "stats.csv"))
                stats_details = pd.DataFrame({"r": [r], "n": [n], "t": [t], "baci": baci})
                stats_details.to_csv(os.path.join(stats_dir, "stats_details.csv"))
            with open(os.path.join(stats_dir, f"jaccard_pred_true.pkl"), "wb") as file:
                pkl.dump(jaccards, file)

            node_degree_evolution(stats_folder=stats_dir)
            edge_distribution_animation(samples=stochastic_sample, adj_true=a, save_path=figures_dir, baci=baci)
            plot_evolution_of_edges(predictions=stochastic_sample, a_true=a, by=10, save_path=figures_dir)
            in_out_degree(a, samples, save_path=figures_dir, baci=baci)
            visualize_jaccard_matrix(stats_path=stats_dir, save_path=figures_dir)
            plot_roc_precision_confusion(a=a, logits=logits_plot, save_path=figures_dir)
            plot_centrality(statistics_path=stats_dir, preds=stochastic_sample, a_true=a, save=save, by=5, baci=baci)
            d, d_pred = temporal_density(a[1:], stochastic_sample, baci=baci, save_dir=figures_dir)
            if save:
                with open(os.path.join(figures_dir, "settings.txt"), "w") as file:
                    settings = f"dropout_rate: {dropout} \n" \
                               f"baci:{baci} \n" \
                               f"features: {features} \n" \
                               f"tbp: {seq_len} \n" \
                               f"unroll: {unroll} \n" \
                               f"stateful: {stateful} \n" \
                               f"epochs: {epochs} \n" \
                               f"test_split: {test_split} \n" \

                    file.write(settings)
                print(f"Figures saved to {figures_dir}")
            plt.show()
        else:
            cell = models.RecurrentEncoderDecoder2(nodes=N,
                                                   features=f,
                                                   channels=5,
                                                   attention_heads=8,
                                                   hidden_size=25,
                                                   dropout=dropout,
                                                   recurrent_dropout=dropout)
            rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=False, stateful=stateful)
            outputs = rnn((input_1, input_2))
            model = tf.keras.models.Model([input_1, input_2], outputs)
            model.compile(optimizer="rmsprop",
                          loss=lambda true, pred:
                          tf.math.reduce_sum(
                              tf.math.sqrt(
                                  tf.reduce_mean(
                                      tf.math.square(true - pred)
                                  ), axis=(-1, -2)
                              )
                          )
                          )
            history = model.fit(x=[x[:, :-15, :], a[:, :-15, :]], y=a[:, 1:-14, :], epochs=150, verbose=1)
            history = history.history["loss"]
            samples = tf.squeeze(model([x, a])).numpy()
            fig, axs = plt.subplots(1, 2)


            def update(i):
                fig.suptitle(f"Time {i}")
                axs[0].cla()
                axs[1].cla()
                axs[0].imshow(samples[i])
                axs[1].imshow(a[i + 1])


            anim = animation.FuncAnimation(fig, update, samples.shape[0] - 1, repeat=True)
            plt.show()
