import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.keras import models as m
from tensorflow_probability import distributions as tfd
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    save_dir = os.path.join(os.getcwd(), "Figures", "Uncertainty Test")
    samples = 1000
    batches = 2
    sigma = np.cos(np.linspace(0, 2 * np.pi, samples)) * 5 + 6
    x = np.linspace(0, 2 * np.pi, samples)
    mu = x * 5
    target_data = np.empty(shape=(batches, samples))
    input_data = np.empty(shape=(batches, samples))
    for i in range(batches):
        normal_samples = np.random.normal(loc=mu, scale=sigma, size=samples)
        target_data[i] = normal_samples
        input_data[i] = x

    input_model_data = input_data.flatten().reshape(-1, 1)
    target_model_data = target_data.flatten().reshape(-1, 1)
    plt.scatter(input_model_data.flatten(), target_model_data.flatten(), color="lime", label="True Samples", s=3)
    i = l.Input(shape=(1,))
    f = l.Dense(100, "relu")(i)
    f = l.Dense(100, "relu")(f)
    f = l.Dense(100, "relu")(f)
    mean_var = l.Dense(2, None)(f)
    model = m.Model(i, mean_var)


    def distr(x):
        return tfd.Normal(loc=x[..., :1], scale=1e-3 + tf.math.softplus(0.05 * x[..., -1:]))


    model.compile(optimizer="adam", loss=lambda y, y_hat: -distr(y_hat).log_prob(y))

    model.fit(x=input_model_data, y=target_model_data, epochs=200, shuffle=True)
    preds = model.predict(input_model_data)
    model_samples = distr(preds).sample(1)

    plt.scatter(input_model_data.flatten(), model_samples.numpy().flatten(), color="fuchsia", label="Model Samples", s=2.5)
    plt.title("Samples")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "samples.png"))
    fig, ax = plt.subplots(2)
    preds2 = model.predict(x[:, None])
    mu_pred = preds2[:, 0]
    sigma_pred = 1e-3 + tf.math.softplus(0.05 * preds2[..., -1:])
    ax[0].plot(x, mu_pred, label="Pred Mean")
    ax[0].plot(x, mu, label="True Mean")
    ax[0].set_title("mu")
    ax[0].legend()
    ax[1].plot(x, sigma_pred, label="Pred Variance")
    ax[1].plot(x, sigma, label="True Variance")
    ax[1].set_title("Sigma")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(fname=os.path.join(save_dir, "params.png"))

    plt.show()
