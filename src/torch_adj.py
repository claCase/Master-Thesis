import os.path as osp
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import os
import torch

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected, to_networkx
from networkx.algorithms.components import is_weakly_connected

from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops,
    remove_self_loops,
)
from torch_scatter import scatter_add
import scipy
import pickle as pkl


def get_pr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):
    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )
    else:
        edge_weight = torch.FloatTensor(edge_weight).to(edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes
    )
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float("inf")] = 0
    p = deg_inv[row] * edge_weight
    p_dense = torch.sparse.FloatTensor(
        edge_index, p, torch.Size([num_nodes, num_nodes])
    ).to_dense()
    print(p_dense)
    # pagerank p
    p_pr = p_dense * (1.0 - alpha) + alpha / num_nodes * torch.ones(
        (num_nodes, num_nodes), dtype=dtype, device=p.device
    )

    eig_value, left_vector = scipy.linalg.eig(p_pr.numpy(), left=True, right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    # assert val[0] == 1.0

    pi = left_vector[:, ind[0]]  # choose the largest eig vector
    pi = pi / pi.sum()  # norm pi
    print(f"Normalized perron {pi}")
    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi < 0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float("inf")] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float("inf")] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_pr
    L = (
        torch.mm(torch.mm(pi_sqrt, p_pr), pi_inv_sqrt)
        + torch.mm(torch.mm(pi_inv_sqrt, p_pr.t()), pi_sqrt)
    ) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # # let little possbility connection to 0, make L sparse
    # L[ L < (1/num_nodes)] = 0
    # L[ L < 5e-4] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col], pi


if __name__ == "__main__":
    with open(
        "A:\\Users\\Claudio\\Documents\\PROJECTS\\Master-Thesis\\Data\\complete_data_final_transformed_no_duplicate.pkl",
        "rb",
    ) as file:
        data_np = pkl.load(file)
    print(data_np.shape)
    slice = np.argwhere((data_np[:, 0] == 1) & (data_np[:, 1] == 1))
    data_slice = data_np[slice]
    data_slice = data_slice[:, 0]
    data_slice = data_slice[:, 2:]
    print(data_slice.shape)
    # data_slice = data_slice[:,2:]
    # print(data_slice[:10, :-1].astype("int").tolist())
    # print(data_slice[:, -1].tolist())
    print(np.max(data_slice, 0)[:-1].astype("int").tolist())
    indices = data_slice[:, :-1].astype("int").tolist()
    values = data_slice[:, -1].tolist()
    shape = (np.max(data_slice, 0)[:-1] + 1).astype("int").tolist()
    data_sp = torch.sparse_coo_tensor(list(zip(*indices)), values, shape)
    print(data_sp.to_dense()[:10, :10])
    data_sp = data_sp.coalesce()
    e, d, pi = get_pr_directed_adj(
        0.2, data_sp.indices(), shape[-1], torch.float32, data_sp.values()
    )
    # data_sp = tf.sparse.reorder(data_sp)
