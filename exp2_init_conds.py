from md_network import *
from utils import *
from itertools import product
import numpy as np

# ===== Generation of Random Probabilities and Means =====

def EXP2_generate_probs(mu_range, k_range, n, max_iters=10):
  """
  Creates a random vector of probabilities for 2 communities, bounded by specific ranges.

  Args:
    mu_range (list(float)): Range of the expected degree ratio for nodes.
    k_range (list(float)): Range of the expected external degree for nodes.
    n (int): The amount of participants.
    max_iters (int): The amount of attempts to generate valid probabilities.

  Returns:
    list(list(float)): The probabilities of connections in the network.
    float: Expected degree ratio for nodes.
    float: Expected external degree for nodes.
  """
  # Define initial variables
  rng = np.random.default_rng()
  mu_min, mu_max = mu_range
  k_min, k_max   = k_range

  for _ in range(max_iters):
    # Obtain the parameters
    mu = rng.uniform(mu_min, mu_max)
    k  = rng.uniform(k_min, k_max)

    # Calculate variables
    nc_avg = n / 2
    p_in   = (k * (1 - mu)) / (nc_avg - 1)
    p_out  = (mu * k) / nc_avg

    if 0 <= p_in <= 1 and 0 <= p_out <= 1 and p_out < p_in:
      return [[p_in, p_out], [p_out, p_in]], mu, k

  else:
    raise ValueError("Couldn't define valid probabilities with the given constraints.")


def EXP2_generate_means(c, t, low, high, eps=0.1, max_iters=10):
  """
  Creates a random vector of means for 2 communities and 2 topics, bounded by specific ranges.

  Args:
    c (int): Current "means" case by community (0 = communities agree, 1 = communities disagree).
    t (int): Current "means" case by topic (0 = consistent topics, 1 = inconsistent topics).
    low (list(float)): Low range of mean differences.
    high (list(float)): High range of mean differences.
    eps (float): Parameter to add noise to the results.
    max_iters (int): The amount of attempts to generate valid means.

  Returns:
    list(list(float)): The means of intrinsic opinions.
    float: Community difference on one topic.
    float: Topic difference on one community.
  """
  # Initial variables
  rng   = np.random.default_rng()
  sign  = lambda: rng.choice([-1, 1])
  noise = lambda: sign() * rng.uniform(0, eps)

  for _ in range(max_iters):

    s1, s2, st = sign(), sign(), sign()
    dc1, dc2 = rng.uniform(*low), rng.uniform(*low)
    dt = 0
    nc = 1 - c
    dc = rng.uniform(*high)

    if t == 0:
      if c == 1: dc1 = dc2 = dc
      else: dc = rng.choice([dc1, dc2])
      s1 = s2
    else:
      dt = rng.uniform(*high)
      dc1, dc2 = dc * c, dc * nc

    xc1 = s1 * dc1 / 2
    xc2 = s2 * dc2 / 2
    xt  = st * dt  / 2

    # Case within bounds
    if dc1 + dc2 + dt <= 2:
      
      # Create noise and mean base
      boom = [noise() for _ in range(4)]
      lt = abs(xt)
      lc = max(abs(xc1), abs(xc2))
      lb = max(abs(b) for b in boom)
      limit = abs(1 - (lt + lc + lb))
      mu = rng.uniform(-limit, limit)

      # Calculate means
      means = np.clip(
        [
          [mu + xc1 + xc2 + xt + boom[0], mu + xc1 + xc2 + xt + boom[1]],
          [mu - xc1 + xc2 + xt + boom[2], mu + xc1 - xc2 + xt + boom[3]]
        ],
        -1, 1
        ).tolist()
      
      [[m11, m12], [m21, m22]] = means

      final_dc = abs(m11 - m21) + abs(m12 - m22)
      final_dt = abs(m11 - m12) + abs(m21 - m22)
      return means, final_dc, final_dt

  else:
    raise ValueError("Couldn't define valid means with the given constraints.")
  

# ===== Main Function =====

def EXP2_initial_conditions(
    mu_ranges=[[0.05, 0.2], [0.25, 0.4]],
    k_ranges=[[20, 45], [55, 80]],
    mean_ranges=[[0.1, 0.3], [0.7, 0.9]],
    networks=100
    ):
  """
  Creates the amount of networks given as parameter for each case of initial conditions,
  calculates their metrics at convergence and returns all of that data.

  Args:
    mu_ranges (list(list(float))): Matrix with low and high ranges for mu.
    k_ranges (list(list(int))): Matrix with low and high ranges for k.
    mean_ranges (list(list(float))): Matrix with low and high ranges for mean parameters.
    networks (int): The amount of networks to create.

  Returns:
    dict: A dictionary containing several lists with different metrics calculated for each network.
  """
  # Create dictionary to store the resulting data
  dict_labels = [
    "mu", "k", "dC", "dT", "P_G", "D_G", "I_G", "P_1", "P_2", "D_1", "D_2", "I_1", "I_2",
    "P_1_1d", "P_2_1d", "D_1_1d", "D_2_1d", "I_1_1d", "I_2_1d"
    ]
  data = {metric: np.empty((4, 3, networks), dtype=float) for metric in dict_labels}
  index = 0

  # Iterate through the 16 total cases, with <networks> networks each
  for mu, k, c, t in product(range(2), range(2), range(2), range(2)):
    if t == 1:
      if c == 0: continue
      if c == 1: c = int(np.random.default_rng().choice([0, 1]))
    index += 1
    case_str = f"Case {[mu, k, c, t]} ({index}/12) ---"

    for i in range(networks):
      print(f"\r{case_str} Processing Network {i + 1}/{networks}...", end="")

      # Generate parameters and network
      while True:
        try:
          m_low, m_high = mean_ranges
          sizes, probs, means, std, m, vars = random_parameters(
            c=[2, 2],
            p_func=EXP2_generate_probs,
            p_args=(mu_ranges[mu], k_ranges[k]),
            mu_func=EXP2_generate_means,
            mu_args=(c, t, m_low, m_high),
            max_m=2
            )
          N = MDNetwork(sizes=sizes, probs=probs, means=means, std=std, m=m)
          break
        except ValueError:
          print(f"\r{case_str} Network Creation Failed ({i + 1}/{networks}). Retrying...", end="")
      print(f"\r{case_str} Processing Network {i + 1}/{networks}...", end="")

      # Add results to data dictionary
      metrics_t = N.get_metrics()
      metrics_1d = N.get_metrics_1D()
      metrics_g = np.mean(metrics_t, axis=1)
      nw_values = np.concatenate((vars, metrics_g, metrics_t, metrics_1d), axis=None)
      p_case = (index - 1) // 3
      m_case = (index - 1) % 3
      for metric, value in zip(dict_labels, nw_values):
        data[metric][p_case][m_case][i] = value

  print("\rDone!")
  return data