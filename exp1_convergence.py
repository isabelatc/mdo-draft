from md_network import *
from utils import *
import numpy as np

# ===== Main Function =====

def EXP1_convergence(networks=100, max_iters=1000, tol=1e-5, p_in=[0.09, 0.3], p_out=[0.001, 0.08]):
  """
  Creates the amount of networks given as parameter, updates them until reaching equilibrium and returns
  data for all of them.

  Args:
    networks (int): The amount of networks to create.
    max_iters (int): The maximum iterations for a network to reach equilibrium.
    tol (float): Minimum difference of opinions tolerated between iterations.
    p_in (list(float)): Range of the in-community edge probabilities.
    p_out (list(float)): Range of the outer-community edge probabilities.

  Returns:
    dict: A dictionary containing several lists with different metrics calculated for each network.
  """
  # Define dictionary to be filled with the networks' information
  dict_labels = [
    ("n", int), ("c", int), ("std", float), ("m", int),
    ("p_in", object), ("p_out", object), ("means", object),
    ("p_in_min", float), ("p_out_min", float), ("means_min", float),
    ("p_in_max", float), ("p_out_max", float), ("means_max", float),
    ("z_eq", object), ("z", object)
    ]
  data = {label: np.empty(networks, dtype=label_type) for label, label_type in dict_labels}

  for i in range(networks):
    print(f"\rProcessing Network {i + 1}/{networks}...", end="")

    # Generate parameters and network
    while True:
      try:
        sizes, probs, means, std, m, _ = random_parameters(p_in=p_in, p_out=p_out)
        N = MDNetwork(sizes=sizes, probs=probs, means=means, std=std, m=m)
        break
      except ValueError:
        print(f"\rNetwork Creation Failed ({i + 1}/{networks}). Retrying...", end="")
    print(f"\rProcessing Network {i + 1}/{networks}...", end="")

    # Pre-processing
    probs = np.array(probs)
    p_in  = np.diagonal(probs)
    p_out = probs[np.triu_indices_from(probs, 1)]
    means = np.ravel(means)
    _, z, _, _ = N.update_process(save_z=True, tol=tol, max_iters=max_iters)

    # Add data to dictionary
    data["n"][i]     = sum(sizes)
    data["c"][i]     = len(sizes)
    data["p_in"][i]  = p_in
    data["p_out"][i] = p_out
    data["means"][i] = means
    data["std"][i]   = std
    data["m"][i]     = m
    data["z_eq"][i]  = N.get_z_eq()
    data["z"][i]     = z

    # Adding variables to reduce data
    for l, v in [("p_in", p_in), ("p_out", p_out), ("means", means)]:
      data[f"{l}_min"][i] = np.min(v)
      data[f"{l}_max"][i] = np.max(v)

  # Last data processing
  for l in ["p_in", "p_out", "means"]:
    data[l] = np.concatenate(data[l])
  data["means_max_min"] = data["means_max"] - data["means_min"]

  print("\rDone!")
  return data