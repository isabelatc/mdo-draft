from md_network import *
from config import *
from itertools import product
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.sparse as sp

# ===== Network Helper =====

def random_parameters(
    c=[2, 5], nc=[100, 200], p_in=None, p_out=None,
    p_func=None, p_args=(), mu_func=None, mu_args=(),
    lb=-1, ub=1, max_m=10, tol=1e-5
    ):
  """
  Generates random variables to create a network (sizes, probs, means, std and m).

  Args:
    c (list(int)): Range of communities to generate.
    nc (list(int)): Range of participants per community.
    p_in (list(float)): Range of the in-community edge probabilities.
    p_out (list(float)): Range of the outer-community edge probabilities.
    p_func (callable): Function to generate probabilities.
    p_args (tuple): Arguments for p_func.
    mu_func (callable): Function to generate means.
    mu_args (tuple): Arguments for mu_func.
    lb (int): Minimum value of opinions.
    ub (int): Maximum value of opinions.
    max_m (int): Maximum amount of dimensions/topics for the network.
    tol (float): A tolerance value between probabilities.

  Returns:
    list(int): The sizes of the communities.
    list(list(float)): The edge probabilities.
    list(list(float)): The means of intrinsic opinions.
    float: The standard deviation.
    int: The dimension of the network.
  """
  rng = np.random.default_rng()

  # "sizes" (<c[0]> to <c[1]> communities, from <nc[0]> up to <nc[1]> participants each)
  comms = rng.integers(low=c[0], high=c[1] + 1)
  sizes = rng.integers(low=nc[0], high=nc[1] + 1, size=comms).tolist()
  n = sum(sizes)

  # "probs" (between communities must be lower than within communities)
  if p_func:
    p_args += (n, )
    res = p_func(*p_args)
    probs, mu, k = [np.round(p, 3) for p in res]
  else:
    p_ins = rng.uniform(p_in[0], p_in[1] + tol, comms)
    pre_probs = np.triu(rng.uniform(p_out[0], p_out[1] + tol, (comms, comms)))
    probs = pre_probs + pre_probs.T
    np.fill_diagonal(probs, p_ins)
    probs = np.round(probs, 3)

  # "m" (any integer between 2 and <max_m>)
  m = rng.integers(low=2, high=max_m + 1)

  # "means" (for <m> topics)
  if mu_func:
    res = mu_func(*mu_args)
    means, dc, dt = [np.round(mu, 3) for mu in res]
  else:
    means = np.round(rng.uniform(lb, ub, (comms, m)), 3)

  # "std" (any number between 0 and 1)
  std = np.round(np.random.rand(), 3)
  
  # Variable with parameters for specific case
  var_params = [] if not (p_func or mu_func) else [mu, k, dc, dt]

  return sizes, probs.tolist(), means.tolist(), std, m, var_params

# ===== 2D Example Helpers =====

def pretty_matrix(name: str, M: sp.csr_matrix):
  """
  Prints a pretty matrix.
  """
  print(f"\n{name} = ", end="")
  for i, row in enumerate(M.toarray()):
    space = "" if i == 0 else "    "
    print(f"{space}{row}")


def show_step_details(nw: MDNetwork, now: int, next: int):
  """
  Prints the detailed formula of a step in the network's social process.
  """
  se = lambda x: " " if x >= 0 else ""
  n = sum(nw.sizes)
  m = nw.m
  _, ops, _, _ = nw.update_process(save_z=True)
  for a, i in product(range(m), range(n)):
    m_terms = np.round([(nw.A[a * n + i, b * n + j], ops[now][b * n + j]) for b in range(m) for j in range(n) if i != j], 2)
    deg_term = 1 / (np.sum(m_terms, axis=0)[0] + 1)
    next_op = get_vector_topic(ops[next], a, m)[i]
    mult = [f"({w:.2f} x {se(o)}{o:.2f})" for w, o in m_terms]
    s_i_a = nw.s[a * n + i]
    print(f"z_{i + 1}[{a + 1}] = {deg_term:.3f} x [(1 x {se(s_i_a)}{s_i_a}) + {" + ".join(mult)}] = {se(next_op)}{next_op:.2f}")


# ===== Plot Helpers =====

def set_labels(
    ax: mpl.Axes, title: str, xlabel: str, ylabel: str, 
    titlesize: int=11, xsize: int=11, ysize: int=11):
  """
  Add formatted labels to a plot.
  """
  ax.set_title(title, fontdict={"fontsize":titlesize})
  ax.set_xlabel(xlabel, fontdict={"fontsize":xsize})
  ax.set_ylabel(ylabel, fontdict={"fontsize":ysize})


def save_and_show(plt: plt, filename: str, suptitle: str=None):
  """
  Saves a plot into a file and shows it with certain parameters.
  """
  if suptitle:
    plt.suptitle(suptitle, fontsize=12)
  plt.savefig(filename, bbox_inches="tight", pad_inches=0.05)
  plt.show()


def customize_bp(bp: dict, whisker: dict={}, box: dict={}, cap: dict={}, median: dict={}):
  """
  Personalizes a boxplot.
  """
  items = [("whiskers", whisker), ("boxes", box), ("caps", cap), ("medians", median)]
  for i, v in items:
    for x in bp[i]:
      x.set(**v)


def make_fig(
    nrows: int=1, ncols: int=1, 
    width: float=6.5, height: float=3, 
    pads: bool=False
    ) -> tuple[mpl.Figure, mpl.Axes]:
  """
  Creates a figure with specific parameters.
  """
  f, axs = plt.subplots(nrows, ncols, figsize=(width, height), constrained_layout=True)
  if pads:
    f.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1)
  return f, axs


def common_legend(f: mpl.Figure, handles: any=None, ncol: int=2):
  """
  Adds a shared legend for multiple subplots in one figure.
  """
  kwargs = dict(
    fontsize=10, loc="upper center", bbox_to_anchor=(0.52, 0),
    ncol=ncol, alignment="left", borderpad=0.6
    )
  f.legend(handles=handles, **kwargs) if handles else f.legend(**kwargs)

def conds_handles(extra: list=[]) -> list:
  """
  Creates legend handles for non-regularized and regularized cases (EXP3).
  """
  handles = [Line2D([0], [0], color=COLORS_2[c], label=COND_LABELS[c]) for c in COND_LABELS]
  return handles + extra


# ===== Checkpoint Helpers =====

def save_pkl(var: any, filename: str):
  """
  Saves a variable into a Pickle file.
  """
  with open(filename, "wb") as f:
    pickle.dump(var, f)


def load_pkl(filename: str) -> any:
  """
  Loads a variable from a Pickle file.
  """
  with open(filename, "rb") as f:
    return pickle.load(f)


def add_progress(msg: str, log_file: str="progress.log", end: any=None):
  """
  Logs a progress message to a file, and also prints it. If <end> is not None, it must
  print adding "\r" to the start to the message so it overwrites itself.
  """
  with open(log_file, "a") as f:
    f.write(msg + "\n")
  if end is None:
    print(msg)
  else:
    print("\r" + msg, end="", flush=True)


# ===== Data Modifiers (EXP3) =====

def data_by_key(
    data: dict, 
    avg: bool=False,
    order: str="m_c",
    conds: list=["NR", "RR"],
    metrics: list=["P", "D", "I", "D_st", "D_ct"]
    ) -> dict:
  """
  Transforms a dictionary of data into a more manageable organization.
  """
  nws_m = [nw["m"] for nw in data[conds[0]]]
  max_m = max(nws_m)

  outer_keys = metrics if order == "m_c" else conds
  inner_keys = metrics if order == "c_m" else conds
  final = {ok: {ik: [] for ik in inner_keys} for ok in outer_keys}

  for ok in outer_keys:
    for ik in inner_keys:
      met = ok if order == "m_c" else ik
      con = ok if order == "c_m" else ik
      if avg:
        for nw in data[con]:
          iters = len(nw[met])
          iter_data = [np.mean(nw[met][i]) for i in range(iters)]
          final[ok][ik].append(iter_data)
      else:
        for a in range(max_m):
          a_list = []
          for j, nw in enumerate(data[con]):
            if a < nws_m[j]:
              iters = len(nw[met])
              a_data = [nw[met][i][a] for i in range(iters)]
            else:
              a_data = []
            a_list.append(a_data)
          final[ok][ik].append(a_list)

  return final


def EXP3_get_delta(data: dict) -> list:
  """
  Given metric data, obtains the difference in percentage between the first and last 
  measurements, for each network.
  """
  deltas = []
  for nw in data:
    ini, fin = nw[0], nw[-1]
    delta_nw = (fin - ini) * 100 / ini
    deltas.append(delta_nw)
  return deltas