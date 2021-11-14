import tensorflow as tf
from tensorflow.keras.layers import LSTM
import tensorflow.keras as k
from src.modules.models import GAT_BIL_spektral_dense
from spektral.data.loaders import BatchLoader
from spektral.data.graph import Graph
from spektral.data.dataset import Dataset
from spektral.layers.convolutional import GATConv
import numpy as np
import matplotlib.pyplot as plt


gat_bil = GAT_BIL_spektral_dense(channels=10, attn_heads=10, concat_heads=True)
lstm = LSTM(units=10, unit_forget_bias=10, return_sequences=False)