import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.keras import models as m
from sklego.meta import ZeroInflatedRegressor
from sklearn.linear_model import GammaRegressor, LogisticRegression, TweedieRegressor
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    os.chdir("../../")
    with open(
            os.path.join(os.getcwd(), "Data", "complete_data_final_transformed_no_duplicate.pkl"),
            "rb",
    ) as file:
        data_np = pkl.load(file)
    data_sp = tf.sparse.SparseTensor(
        data_np[:, :4], data_np[:, 4], np.max(data_np, 0)[:4] + 1
    )
    data_sp = tf.sparse.reorder(data_sp)

    data_slice = tf.sparse.slice(data_sp, (0, 31, 0, 0), (1, 1, 174, 174))
    data_dense = tf.sparse.reduce_sum(data_slice, (0,1), output_is_sparse=False)
    data_dense += 1
    data = tf.sparse.from_dense(data_dense)
    x = np.eye(174)
    x = np.asarray(x, dtype=np.float32)
    x_i = tf.gather(x, data.indices[:,0])
    x_j = tf.gather(x, data.indices[:,1])
    y_ij = (data.values.numpy() - 1)
    #y_ij = np.exp(np.random.gamma(1,1,size=x_j.shape[0])) * np.random.choice((0,1), size=x_j.shape[0])
    x_ij = tf.concat([x_i, x_j], -1).numpy()
    print(x_ij.shape)
    print(y_ij.shape)
    #regressor = GammaRegressor()
    regressor = TweedieRegressor(power=1.8, link="log")
    classifier = LogisticRegression()
    model = ZeroInflatedRegressor(classifier, regressor)
    model.fit(x_ij, y_ij)
    preds = model.predict(x_ij)
    preds = preds.flatten()
    preds = preds[preds>0]
    preds = np.log(preds)
    y_ij = y_ij.flatten()
    y_ij = y_ij[y_ij>0]
    y_ij = np.log(y_ij)
    params = model.get_params()
    print(preds)
    plt.hist(preds, bins=100, color="red", alpha=0.4, label="Pred")
    plt.hist(y_ij, bins=100, color="blue", alpha=0.4, label="True")
    plt.legend()
    plt.show()