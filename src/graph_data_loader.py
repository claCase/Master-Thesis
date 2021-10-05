# import grand
import os
from typing import List, Dict, Set
import numpy as np
import pandas as pd
import zipfile as zp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import pycountry_convert as pcc
# from geopy.geocoders import Nominatim
# import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import urllib.request as req
from bs4 import BeautifulSoup as bs
import pickle as pkl
import tensorflow as tf

YEARS_FOLDER = "C:\\Users\\claud\\OneDrive\\Documents\\PROJECTS\\Master-Thesis"
COUNTRIES_CODES_PATH = os.path.join(os.getcwd(), "Comtrade", "Reference Table",
                                    "Comtrade Country Code and ISO list.xls")


def relational_graph_plotter(graph):
    if isinstance(graph, tf.sparse.SparseTensor):
        graph = tf.sparse.to_dense(graph)
    fig, ax = plt.subplots(graph.shape[0])
    for i in range(graph.shape[0], 1):
        ax[i, 0].imshow(graph[i, :, :], cmap="winter")
    plt.show()

def from_data_sparse_numpy_to_ntx(data_sp):

    df = pd.DataFrame("")
    G = nx.from_pandas_edgelist(df=df[["ReporterISO3", "PartnerISO3", "TradeValue", "ProductCode2"]],
                                source="ReporterISO3", target="PartnerISO3", edge_attr=["TradeValue"],
                                edge_key="ProductCode2", create_using=nx.MultiDiGraph())
def get_iso2_long_lat():
    if not os.path.exists("./Data/iso2_long_lat.pkl"):
        print("loading from website")
        URL = "https://developers.google.com/public-data/docs/canonical/countries_csv"
        doc = req.urlopen(URL).read()
        doc_bs = bs(doc, "html.parser")
        trs = doc_bs.findAll("tr")

        codes, latitude, longitude, name = [], [], [], []
        for tr in trs:
            tds = tr.findAll("td")
            if tds:
                if not str(tds[0].getText()) == "UM":
                    codes.append(str(tds[0].getText()))
                    latitude.append(float(tds[1].getText()))
                    longitude.append(float(tds[2].getText()))
                    name.append(str(tds[3].getText()))

        latitude = np.asarray(latitude)
        longitude = np.asarray(longitude)

        df = pd.DataFrame({"code": codes, "latitude": latitude, "longitude": longitude, "name": name})
        cc = {}
        for i, row in df[["code", "latitude", "longitude"]].iterrows():
            cc[row["code"]] = np.asarray([row["longitude"], row["latitude"]], dtype=np.float32)
        print(f"saving pickle")
        with open("./Data/iso2_long_lat.pkl", "wb") as file:
            pkl.dump(cc, file)
    else:
        print("loading from pickle")
        with open("./Data/iso2_long_lat.pkl", "rb") as file:
            cc = pkl.load(file)
    return cc


def from_edgelist_to_pd(edgelist, values):
    df_convert = pd.read_excel(COUNTRIES_CODES_PATH)
    edgelist = np.asarray(edgelist)
    for i in range(len(edgelist)):
        c1 = df_convert[df_convert["Country Code"] == edgelist[i, 0]]["ISO3-digit Alpha"]
        c2 = df_convert[df_convert["Country Code"] == edgelist[i, 1]]["ISO3-digit Alpha"]
        edgelist[i, 0] = c1
        edgelist[i, 1] = c2

    for i in range(len(edgelist)):
        edgelist[i, 0] = df_convert[edgelist[i, 0]]
        edgelist[i, 1] = df_convert[edgelist[i, 1]]

    df = pd.DataFrame({"code1": edgelist[:, 0], "code2": edgelist[:1], "prod": edgelist[:, 2], "tv": values})
    G = nx.from_pandas_edgelist(df=df,
                                source="code1", target="code2", edge_attr=["tv"],
                                edge_key="prod", create_using=nx.MultiDiGraph())

    return G


def select_edges(edge_list: [(int, int, str)], values: [int], years: [int], code1: [str], code2: [str],
                 products: [str]):
    subgraph_edgelist = []
    subgraph_values = []
    for row, tv in zip(edge_list, values):
        y = row[0]
        c1 = row[1]
        c2 = row[2]
        p = row[3]
        if y in years and c1 in code1 and c2 in code2 and p in products:
            subgraph_edgelist.append((y, c1, c2, p))
            subgraph_values.append(tv)
    return subgraph_edgelist, subgraph_values


def load_from_WITS(folder=YEARS_FOLDER, t1=1989, t2=1989):
    years = np.arange(t1, t2 + 1)
    for y in years:
        if y > 1999:
            month = "Jul08"
        else:
            month = "Jan17"
        file_name = f"COUNTRY_CAGR_{str(y)}_EXPORT_2020{month}.csv"
        df = pd.read_csv(os.path.join(folder, file_name), converters={"ProductCode": str})
        df = df[(df["ReporterISO3"] != "WLD") &
                (df["ReporterISO3"] != "PartnerISO3") &
                (df["PartnerISO3"] != "WLD")]
        df["ProductCode2"] = df["ProductCode"].astype(str).str[:4]
        codes = df.groupby(["ProductCode2", "ReporterISO3", "PartnerISO3"])["TradeValue"].sum()
        codes = pd.DataFrame(codes).reset_index()
        G = nx.from_pandas_edgelist(df=codes[["ReporterISO3", "PartnerISO3", "TradeValue", "ProductCode2"]],
                                    source="ReporterISO3", target="PartnerISO3", edge_attr=["TradeValue"],
                                    edge_key="ProductCode2", create_using=nx.MultiDiGraph())

        yield G


def get_product_subgraph(G: nx.MultiGraph, prod_key: str) -> nx.Graph:
    edges = [(i, j, k) for i, j, k in G.edges if k == prod_key]
    G_sub = G.edge_subgraph(edges).copy()
    return G_sub


def draw_subgraph(G: nx.MultiDiGraph, prod_keys: [str], nodes: [str] = None, log_scale=True, normalize=True,
                  quantile=.10):
    lat_long = get_iso2_long_lat()
    radius = np.linspace(-0.1, -0.5, len(prod_keys))
    col_idx = np.arange(0, len(prod_keys))
    colors = plt.get_cmap("Set1")(col_idx)
    ax = plt.axes(projection=ccrs.PlateCarree())
    for i, key in enumerate(prod_keys):
        G_sub = get_product_subgraph(G, key)
        iso3_2 = {}
        remove_nodes = []
        for name in G_sub.nodes:
            if nodes is not None:
                print(name)
                if name not in nodes:
                    remove_nodes.append(name)
                    continue
            try:
                i2 = pcc.country_alpha3_to_country_alpha2(str(name))
                iso3_2[name] = i2
                i_pos = lat_long[i2]
                G_sub.nodes[name]["pos"] = i_pos
            except Exception as e:
                print(e)
                remove_nodes.append(name)

        for name in remove_nodes:
            G_sub.remove_node(name)

        width = [G_sub.get_edge_data(i, j, k)["TradeValue"] for i, j, k in G_sub.edges]
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
            xy1 = G_sub.nodes[edge[0]]["pos"]
            xy2 = G_sub.nodes[edge[1]]["pos"]
            plt.scatter([xy1[0], xy2[0]], [xy1[1], xy2[1]], color="b", s=10)
            ax.annotate("",
                        xy=xy2, xycoords='data',
                        xytext=xy1, textcoords='data',
                        arrowprops=dict(arrowstyle="->", color=colors[i],
                                        shrinkA=5, shrinkB=5,
                                        patchA=None, patchB=None,
                                        connectionstyle=f"arc3,rad={radius[i]}",
                                        lw=width[j],
                                        ),
                        transform=ccrs.Geodetic()
                        )

    ax.stock_img()
    ax.add_feature(cfeature.BORDERS, alpha=.5, linestyle=":")
    ax.add_feature(cfeature.COASTLINE, alpha=.5)


def add_self_loops(data_sp: tf.sparse.SparseTensor):
    n_nodes = data_sp.shape[-1]
    years = data_sp.shape[0]
    r = data_sp.shape[1]
    self_edges = []
    self_values = []
    for t in range(years):
        data_t_dense = tf.sparse.to_dense(tf.sparse.slice(data_sp, (t, 0, 0, 0), (1, r, n_nodes, n_nodes)))[0]
        deg_out = tf.reduce_sum(data_t_dense, axis=2)
        deg_sparse = tf.sparse.from_dense(deg_out)
        self_edges_t = np.asarray(deg_sparse.indices)
        idx = self_edges_t[:, -1]
        self_edges_t = np.concatenate(
            [np.ones(len(idx), dtype=np.int64)[:, np.newaxis] * t, self_edges_t, idx[:, np.newaxis]], axis=1)
        self_values_t = deg_sparse.values
        self_edges.extend(self_edges_t.tolist())
        self_values.extend(self_values_t)
    self_data_sp = tf.sparse.SparseTensor(self_edges, self_values, data_sp.shape)
    data_sp = tf.sparse.concat(0, [data_sp, self_data_sp])
    data_sp = tf.sparse.reorder(data_sp)
    return data_sp
