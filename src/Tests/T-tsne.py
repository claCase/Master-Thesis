import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.keras import models as m
from src.modules.utils import pair_wise_distance
import tensorflow.keras as K
from src.modules.utils import discrete_kl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE as TSNE_sk


def HBeta(probs, pair_wise_distance, variance):
    beta = 1.0/(2*variance)
    sumP = tf.reduce_sum(probs, -1)
    H = tf.math.log(sumP) + beta * tf.reduce_sum(probs * pair_wise_distance) / sumP


class TSNE(l.Layer):
    def __init__(self, output_dim, perplexity, gaussian_output=True, activation=None, hidden_layers=1):
        super(TSNE, self).__init__()
        self.encoder = [l.Dense(output_dim, activation) for i in range(hidden_layers)]
        self.perplexity = perplexity
        self.gaussian_output = gaussian_output

    def build(self, input_shape):
        self.variance = self.add_weight(shape=(input_shape[-2],), initializer="glorot_normal")
        self.const_var = tf.constant(1.0 / tf.math.sqrt(2.0))

    def call(self, inputs, *args, **kwargs):
        x_proj = inputs
        for l in self.encoder:
            x_proj = l(x_proj)
        pair_dist_input = pair_wise_distance(inputs)
        pair_dist_emb = pair_wise_distance(x_proj)

        var = tf.math.maximum(tf.nn.softplus(self.variance), tf.math.sqrt(K.backend.epsilon()))
        in_prob = tf.nn.softmax(pair_dist_input / var)
        if self.gaussian_output:
            out_prob = tf.nn.softmax(pair_dist_emb / self.const_var)
        else:
            numerator = 1 / (1 + pair_dist_emb)
            out_prob = numerator / tf.reduce_sum(numerator, axis=None)
        #self.add_loss(HBeta())
        return x_proj, in_prob, out_prob


if __name__ == "__main__":
    from tensorflow_probability import distributions as tfd

    p_mix = [1 / 3, 1 / 3, 1 / 3]
    mu = [(1, 1, 1), (5, 5, 5), (10, 10, 10)]
    mu = tf.constant(mu, dtype=tf.float32)
    sigma = tf.constant([[1,1,1], [1,1,1], [0.5, 0.5, 0.5]])
    distr = tfd.Mixture(
        cat=tfd.Categorical(probs=p_mix),
        components=[
            tfd.MultivariateNormalDiag(loc=mu[0], scale_diag=sigma[0]),
            tfd.MultivariateNormalDiag(loc=mu[1], scale_diag=sigma[1]),
            tfd.MultivariateNormalDiag(loc=mu[2], scale_diag=sigma[2]),
        ])

    samples = distr.sample(1000)
    tsne = TSNE(1, 2, activation=None, hidden_layers=3)

    optimizer = K.optimizers.Adam(0.001)
    loss_hist = []
    loss = K.losses.KLDivergence()
    Y = np.random.sample(size=(samples.shape[0], 1))
    Y = tf.Variable(initial_value=Y)
    initial_Y = Y.numpy().copy()
    parametric = False
    for i in range(50):
        with tf.GradientTape() as tape:
            if parametric:
                x_proj, p, q = tsne(samples)
            else:
                pair_dist_emb = pair_wise_distance(Y, symmetric=True)
                q = tf.nn.softmax(pair_dist_emb)
                pair_dist = pair_wise_distance(samples, symmetric=True)
                p = tf.nn.softmax(pair_dist)

            l = loss(tf.reshape(p, [-1]), tf.reshape(q, [-1]))
        loss_hist.append(l)
        if parametric:
            gradients = tape.gradient(l, tsne.trainable_weights)
            optimizer.apply_gradients(zip(gradients, tsne.trainable_weights))
        else:
            gradients = tape.gradient(l, Y)
            optimizer.apply_gradients([(gradients, Y)])
        print(l)

    samples = samples.numpy()
    if not parametric:
        x_proj = Y.numpy()
    plt.plot(loss_hist)
    plt.figure()
    plt.hist2d(samples[:,0], samples[:,1])
    fig, axs = plt.subplots(2,1)
    axs[0].scatter(x_proj, np.ones(shape=x_proj.shape[0]))
    axs[0].set_title("Final Points")

    axs[1].scatter(initial_Y, np.ones(shape=x_proj.shape[0]))
    axs[1].set_title("Initial Points")

    x0_tnse = TSNE_sk(n_components=1, perplexity=80).fit_transform(samples)
    print(x0_tnse.shape)
    plt.figure()
    plt.title("TSNE Embeddings")
    plt.scatter(x0_tnse, np.ones(x0_tnse.shape[0]))
    plt.show()

