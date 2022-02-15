import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import loggamma


def bijector(alpha, beta, l):
    return tfp.bijectors.Shift(l - 1)(tfp.bijectors.Exp()(tfp.distributions.Gamma(alpha, beta)))


class LogGamma(tfd.Distribution):
    def __init__(self, alpha, beta, l=0.0, dtype=tf.float32):
        super(LogGamma, self).__init__(
            dtype=dtype,
            reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=False,
            parameters={"alpha": alpha, "beta": beta, "l": l},
        )
        self.l = l
        self.gamma = tfd.Gamma(alpha, beta)

    def _sample_n(self, n, seed):
        x = self.gamma.sample(n, seed=seed)
        y = tf.math.exp(x) + self.l - 1
        return y

    def _prob(self, x):
        x = tf.math.log(x - self.l + 1)
        return self.gamma.prob(x)


if __name__ == "__main__":
    l = 0
    a = 20
    b = 40

    lg_sp = loggamma(a, b)
    samples_lg_sp = lg_sp.rvs(10000)
    probs_lg_sp = lg_sp.pdf(samples_lg_sp)

    lg_bj = bijector(a, b, l)
    samples_lg_bj = lg_bj.sample(10000)
    probs_lg_bj = lg_bj.prob(samples_lg_bj)
    samples_lg_bj = samples_lg_bj.numpy().flatten()
    probs_lg_bj = probs_lg_bj.numpy().flatten()

    plt.figure()
    plt.hist(samples_lg_sp, bins=100, density=True)
    plt.scatter(samples_lg_sp, probs_lg_sp, color="orange", s=1)
    plt.title("Scipy Gamma")

    plt.figure()
    plt.hist(samples_lg_bj, bins=100, density=True)
    plt.scatter(samples_lg_bj, probs_lg_bj, color="orange", s=1)
    plt.title("Bijector Gamma")

    plt.show()

