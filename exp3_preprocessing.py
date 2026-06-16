import joblib
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
from md_network import *

def EXP3_get_reddit_df(
    show: bool=True, 
    bots_file: str="REDDIT_data/blacklist_anon.joblib", 
    data_file: str="REDDIT_data/edges_anon.csv"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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

  return df, ops


def EXP3_build_topic_factors(df: pd.DataFrame) -> pd.DataFrame:
  """
  For each node, generates the proportion of the amount of people they share both
  topic opinions, and just one.
  """
  same_soc  = df["c_social"] == df["p_social"]
  same_econ = df["c_economic"] == df["p_economic"]

  df["share_soc"]  =  same_soc & ~same_econ
  df["share_econ"] = ~same_soc &  same_econ
  df["share_both"] =  same_soc &  same_econ

  conds   = ["share_soc", "share_econ", "share_both"]
  c_count = df.groupby("child")[conds].sum()
  p_count = df.groupby("parent")[conds].sum()
  count   = c_count.add(p_count, fill_value=0).astype(int)
  total   = count.sum(axis=1)

  factors = pd.DataFrame(index=count.index)
  factors["f_ss"] = (count["share_both"] + count["share_soc"]) / total
  factors["f_ee"] = (count["share_both"] + count["share_econ"]) / total
  factors["f_se"] = count["share_both"] / total

  return factors.clip(lower=1e-10)


def EXP3_build_weights(
    df: pd.DataFrame, min_deg: int=10, 
    show: bool=True
    ) -> tuple[sp.csr_matrix, dict]:
  """
  Creates weights for all combinations of nodes and topics.
  """
  # Create base weights, counting interactions once for each node
  pairs_count = df.groupby(["child", "parent"]).size().reset_index(name="count")
  rev_pairs   = pairs_count.rename(columns={"child": "parent", "parent": "child", "count": "rev_count"})
  reciprocal  = pairs_count.merge(rev_pairs, on=["child", "parent"])
  reciprocal["u"] = reciprocal[["child", "parent"]].min(axis=1)
  reciprocal["v"] = reciprocal[["child", "parent"]].max(axis=1)
  total_rec = reciprocal.groupby(["u", "v"])["count"].sum().reset_index(name="count")

  max_count = total_rec["count"].max()
  total_rec["w_ij"] = (
    np.log1p(total_rec["count"] / 2) /
    np.log1p(max_count / 2)
    ) * 0.99

  # Filter by degree
  G = nx.from_pandas_edgelist(total_rec, "u", "v", edge_attr="w_ij")
  G = G.subgraph({n for n, d in G.degree() if d >= min_deg}).copy()
  G_final = G.subgraph(max(nx.connected_components(G), key=len)).copy()
  kept = set(G_final.nodes())

  df_new = df[
    df["child"].isin(kept) & df["parent"].isin(kept)
    ].reset_index(drop=True)
  i_node = {node: i for i, node in enumerate(sorted(kept))}
  n = len(i_node)

  # Generate weight factors by topic
  factors = EXP3_build_topic_factors(df_new)

  # Define inner function to apply onto rows
  w_dict = total_rec.set_index(["u", "v"])["w_ij"].to_dict()
  def get_weight(row):
    """
    Returns the weight that correspond to the given row of users.
    """
    u, v = row["child"], row["parent"]
    return w_dict.get((min(u,v), max(u,v)), np.nan)

  df_new["w_ij"] = df_new.apply(get_weight, axis=1)
  df_new = df_new.dropna(subset=["w_ij"]).reset_index(drop=True)

  # Convert data into Numpy variables
  i_child  = df_new["child"].map(i_node).to_numpy()
  i_parent = df_new["parent"].map(i_node).to_numpy()
  w_ij = df_new["w_ij"].to_numpy()

  topic_map = {
     ("soc",   "soc"):  "f_ss",
     ("econ", "econ"):  "f_ee",
     ("soc",   "econ"): "f_se",
     ("econ", "soc"):   "f_se",
     }

  # Create submatrices
  all_A = {}
  for (a, b), f_col in topic_map.items():
    c_factor = df_new["child"].map(factors[f_col]).to_numpy()
    p_factor = df_new["parent"].map(factors[f_col]).to_numpy()
    w_ab = (c_factor + p_factor) / 2 * w_ij

    A_ab = coo_matrix((w_ab, (i_child, i_parent)), shape=(n, n)).tocsr()
    A_ab.sum_duplicates()
    all_A[(a, b)] = A_ab

  # Enforce symmetry
  for k in all_A:
    all_A[k] = (all_A[k] + all_A[k].T) / 2

  # Create final complete matrix A
  A_final = sp.block_array([
     [all_A[("soc", "soc")], all_A[("soc", "econ")]],
     [all_A[("econ", "soc")], all_A[("econ", "econ")]]
     ],
     format="csr")
  A_final.data = np.clip(A_final.data, 0, 1 - 1e-10)

  if show:
    print(f"Users after filtering: {n}")
    print(f"Total edges: {all_A[('soc','soc')].nnz // 2}")
    print(f"\nEdges per submatrix:")
    for (a, b), a_ab in all_A.items():
      print(f"A[{a},{b}]: {a_ab.nnz // 2}")
    print("\n===========================================================\n")

  return A_final, i_node

def EXP3_create_nwr(
    show: bool=True, 
    bots_file: str="REDDIT_data/blacklist_anon.joblib", 
    data_file: str="REDDIT_data/edges_anon.csv"
    ) -> MDNetwork:
  """
  Builds the full network from the reddit dataset, for different levels of
  added cross-topic influence.
  """
  # Obtain graph
  df, ops = EXP3_get_reddit_df(show=show, bots_file=bots_file, data_file=data_file)

  # Get A
  A_final, i_node = EXP3_build_weights(df)

  # Get s
  n = len(i_node)
  i_ops = ops.set_index("user")
  order = sorted(i_node, key=i_node.get)
  s = np.concatenate([
    i_ops.loc[order, "soc"].to_numpy(),
    i_ops.loc[order, "econ"].to_numpy(),
    ])

  # Construct network
  nwr = MDNetwork()
  nwr.s = s
  nwr.sizes = [n]
  nwr.m = 2
  nwr.update_MDN(A_final)

  # Initial Metrics
  z_eq     = nwr.get_z_eq()
  metrics  = nwr.get_metrics(z=z_eq)
  Dz_terms = nwr.get_Dz_terms(z=z_eq)

  print(
    f"P = {np.mean(metrics[0]):.2f}\n",
    f"D = {np.mean(metrics[1]):.2f}\n",
    f"I = {np.mean(metrics[2]):.2f}\n",
    f"D_st = {np.mean(Dz_terms[0]):.2f}\n",
    f"D_ct = {np.mean(Dz_terms[1]):.2f}\n"
    )

  return nwr 
