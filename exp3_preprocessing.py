import joblib
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from md_network import *

def EXP3_process_reddit_data(
    show: bool=True, 
    bots_file: str="REDDIT_data/blacklist_anon.joblib", 
    data_file: str="REDDIT_data/edges_anon.csv"
    ) -> tuple[pd.DataFrame, nx.classes.graph.Graph]:
  """
  Does all the processing to the Reddit dataset and returns its initial opinions and graph.
  """
  if show:
    print("\n===========================================================\n")

  # ===== Load Data and Bots List =====

  bots = joblib.load(bots_file)
  df   = pd.read_csv(data_file)

  if show:
    print("First rows:\n", df.head())
    print("\nList of column names:\n", df.columns.tolist())
    print("\nShape of the data:\n", df.shape)
    print("\nNumber of nulls:\n", df.isnull().sum())
    print("\n===========================================================\n")

  # ===== Filter Bots, Columns and NaN =====

  df = df[(~df["child"].isin(bots)) & (~df["parent"].isin(bots))]
  df = df.drop(columns=["body", "toxicity"], errors="ignore")
  df = df.dropna(subset=["c_economic", "c_social", "p_economic", "p_social"])

  if show:
    print(f"Users after filtering: {len(set(df["child"]).union(df["parent"]))}")
    print(f"Edges after filtering: {len(df)}")
    print("\n===========================================================\n")


  # ===== Generate Numbered Opinions =====

  # Associate each value to a number
  soc_map = {
      "Auth": -1,
      "Centrist": 0,
      "Lib": 1
      }

  econ_map = {
      "Right": -1,
      "Centrist": 0,
      "Left": 1
      }

  df["c_social"]   = df["c_social"].map(soc_map)
  df["p_social"]   = df["p_social"].map(soc_map)
  df["c_economic"] = df["c_economic"].map(econ_map)
  df["p_economic"] = df["p_economic"].map(econ_map)

  # Create temporal tables to concatenate later
  c_ops = df[["child", "c_social", "c_economic"]].copy().rename(
          columns={"child": "user",
                  "c_social": "soc",
                  "c_economic": "econ"}
                  )

  p_ops = df[["parent", "p_social", "p_economic"]].copy().rename(
          columns={"parent": "user",
                  "p_social": "soc",
                  "p_economic": "econ"}
                  )

  ops = pd.concat([c_ops, p_ops]).drop_duplicates("user")

  if show:
    print(f"Real users: {len(ops)}\n")
    print(f"> Count of social opinions:\n{ops["soc"].value_counts()}")
    print("> Proportion of social opinions:")
    print(np.round(ops["soc"].value_counts(normalize=True), 3))
    print(f"\n> Count of economic opinions:\n{ops["econ"].value_counts()}")
    print("> Proportion of economic opinions:")
    print(np.round(ops["econ"].value_counts(normalize=True), 3))
    print("\n===========================================================\n")


  # ===== Process Edges =====

  # Count user pairs (total, directed)
  pairs_count = df.groupby(["child", "parent"]).size().reset_index(name="count")
  rev_pairs = pairs_count.rename(
    columns={"child": "parent",
            "parent": "child",
            "count": "rev_count"}
            )
  # Get only reciprocal pairs
  reciprocal = pairs_count.merge(rev_pairs, on=["child", "parent"])

  if show:
    print(f"Directed edge pairs: {len(pairs_count)}")
    print(f"Reciprocal edge pairs: {len(reciprocal)}")

  # Organize pairs so u < v, and sum reciprocal interactions
  reciprocal["u"] = reciprocal[["child", "parent"]].min(axis=1)
  reciprocal["v"] = reciprocal[["child", "parent"]].max(axis=1)
  total_rec = reciprocal.groupby(["u", "v"])["count"].sum().reset_index(name="count")

  if show:
    print(f"Unique reciprocal and undirected pairs: {len(total_rec)}")

  # Compute weights, with log-normalization
  total_rec["weight"] = (
    np.log1p(total_rec["count"] / 2) /
    np.log1p(total_rec["count"].max() / 2)
    ) * 0.99
  
  if show:
    print("\nWeights data:\n", total_rec.head())
    print("\n===========================================================\n")


  # ===== Generate Graph =====

  # Build graph and filter degrees (> 10)
  G = nx.from_pandas_edgelist(total_rec, "u", "v", edge_attr="weight")
  G = G.subgraph({n for n, d in G.degree() if d >= 10}).copy()
  G_final = G.subgraph(max(nx.connected_components(G), key=len)).copy()

  if show:
    print("Generated graph:")
    print("# Nodes     = ", G_final.number_of_nodes())
    print("# Edges     = ", G_final.number_of_edges())
    print("Avg. Degree = ", f"{2*G_final.number_of_edges()/G_final.number_of_nodes():.2f}")
    print("\n===========================================================\n")

  # Return graph
  return ops, G_final


# ===== Create Full Network =====

def EXP3_create_nwr(
    alphas: list,
    show: bool=True, 
    bots_file: str="REDDIT_data/blacklist_anon.joblib", 
    data_file: str="REDDIT_data/edges_anon.csv"
    ) -> list:
  """
  Builds the full network from the reddit dataset, for different levels of
  added cross-topic influence.
  """
  # Obtain graph
  ops, G_final = EXP3_process_reddit_data(show=show, bots_file=bots_file, data_file=data_file)

  # Get nodes
  nodes = sorted(G_final.nodes())
  node_dict = {node: i for i, node in enumerate(nodes)}
  n = len(nodes)

  # Get s
  s_pre = (ops[ops["user"].isin(nodes)].set_index("user").loc[nodes])
  s = np.concatenate(
    [s_pre["econ"].values.astype(float),
    s_pre["soc"].values.astype(float)]
    )

  # Get A
  rows, cols, ws = [], [], []
  for u, v, d in G_final.edges(data=True):
    i, j = node_dict[u], node_dict[v]
    rows += [i, j]
    cols += [j, i]
    ws += [d["weight"], d["weight"]]
  A_pre = sp.csr_matrix((ws, (rows, cols)), shape=(n, n))

  # Add variation to A
  nw_alphas = {}

  print("\nNetwork data by alpha:\n")
  for alpha in alphas:
    A_cross = alpha * A_pre
    A_full = sp.block_array(
      [[A_pre,     A_cross],
      [A_cross,   A_pre  ]], format="csr"
      )

    # Check weights limits
    print(f"alpha={alpha} —> Max. Weight = {A_full.max():.3f}")

    # Construct network
    nwr = MDNetwork()
    nwr.s = s
    nwr.sizes = [n]
    nwr.m = 2
    nwr.update_MDN(A_full)

    # Initial Metrics
    z_eq     = nwr.get_z_eq()
    metrics  = nwr.get_metrics(z=z_eq)
    Dz_terms = nwr.get_Dz_terms(z=z_eq)

    print(
      f"=> P={np.mean(metrics[0]):.3f},",
      f"D={np.mean(metrics[1]):.3f},",
      f"I={np.mean(metrics[2]):.3f},",
      f"D_st={np.mean(Dz_terms[0]):.3f},",
      f"D_ct={np.mean(Dz_terms[1]):.3f}\n"
      )

    nw_alphas[alpha] = nwr

  return nw_alphas