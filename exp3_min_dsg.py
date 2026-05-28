from md_network import *
from utils import *
from itertools import product
import numpy as np
import random
import scipy.sparse as sp

# ===== Setup =====

# Random Networks Generation

def EXP3_create_nws(num, p_in=[0.09, 0.3], p_out=[0.001, 0.08], max_n=400, max_m=5):
  """
  Creates <num> random networks and saves them into a list.

  Args:
    num (int): The amount of networks to create.
    p_in (list(float)): Range of the in-community edge probabilities.
    p_out (list(float)): Range of the outer-community edge probabilities.
    max_n (int): Maximum amount of nodes.
    max_m (int): Maximum amount of topics.

  Returns:
    list(MDNetwork): A list with the generated networks.
  """
  nws = []
  for _ in range(num):
    # Create networks and save them to list
    nw_args = {"p_in": p_in, "p_out": p_out, "max_m": max_m}
    sizes, probs, means, std, m, _ = random_parameters(**nw_args)
    # Restrict sizes for a given maximum of nodes
    if sum(sizes) > max_n:
      factor = max_n / sum(sizes)
      sizes = [max(50, int(s * factor)) for s in sizes]
    nw = MDNetwork(sizes=sizes, probs=probs, means=means, std=std, m=m)
    nws.append(nw)

  # Save list to Pickle file and download it
 # if os.path.exists(filename):
  #  os.remove(filename)
  #save_pkl(nws, filename)
  #files.download(filename)
  #add_progress(f"Networks created, saved and downloaded! (file: {filename})")

  # Return the list
  return nws


# Calculation of Gradient
def EXP3_gradient_A(nw, z_eq, reg=False, gamma=0.0):
  """
  Calculates the gradient of the global disagreement at equilibrium on the given network, over A.
  The formula is the following: grad_A(Dz*) = diag(grad_L(Dz*)) * 1^T - grad_L(Dz*),
  where grad_L(Dz*) = (1 / m) * (2 * z_eq * (K * z_eq)^T - z_eq * z_eq^T), and K = (L + I)^(-1).
  If reg is True, a regularization term is added to the disagreement function.

  Args:
    nw (MDNetwork): The network to work on.
    z_eq (ndarray): The equilibrium opinions vector.
    reg (bool): Indicates if the regularization term is added or not.
    gamma (float): Parameter that controls the strength of the regularization.

  Returns:
    float: The gradient of the disagreement over A.
  """
  m = nw.m
  n = sum(nw.sizes)
  N = m * n

  Kz = sp.linalg.spsolve(nw.L + sp.eye(N, format="csr"), z_eq)
  grad_L = (1 / m) * (2 * np.outer(z_eq, Kz) - np.outer(z_eq, z_eq))
  diag_L = np.diag(grad_L)
  grad_A = np.outer(diag_L, np.ones(len(diag_L))) - grad_L

  if reg:
    grad_A += 2 * gamma * nw.A.toarray()

  return grad_A


# Projection onto Constraints

def EXP3_enforce_symmetry(A_proj: np.ndarray, n: int, m: int) -> np.ndarray:
  """
  Modifies the matrix A_proj to enforce symmetry by blocks and in general.
  """
  for a, b in product(range(m), range(m)):
    if a <= b:
      rs, re = a * n, a * n + n
      cs, ce = b * n, b * n + n
      temp = A_proj[rs:re, cs:ce]
      A_proj[rs:re, cs:ce] = (temp + temp.T) / 2
      A_proj[cs:ce, rs:re] = A_proj[rs:re, cs:ce]
  return A_proj


def EXP3_project_A(nw, A_proj, A_init, epsilon, deg_tol=1e-2):
  """
  Modifies the given A_proj to fit the constraints of the model (starting from A_init).

  Args:
    nw (MDNetwork): The network to work on.
    A_proj (ndarray): The matrix obtained from the minimization process.
    A_init (ndarray): The original modified adjacency matrix.
    epsilon (float): Constraint parameter representing how much weights can change.
    deg_tol (float): Constraint parameter representing the maximum change of nodes' degrees.

  Returns:
    ndarray: The projected A matrix (A_proj modified).
  """
  m = nw.m
  n = sum(nw.sizes)

  # Maintain original zeros in the matrix
  mask = (A_init != 0)
  A_proj *= mask

  # Enforce symmetry (total and by topic-block)
  A_proj = EXP3_enforce_symmetry(A_proj, n, m)

  # Clip weights to satisfy their limits
  A_proj = np.clip(A_proj, 0, 1 - 1e-10)

  # Update degrees to be close to the original ones
  for a in range(m):
    rs, re = a * n, (a + 1) * n
    deg_init = A_init[rs:re].sum(axis=1)
    deg_proj = A_proj[rs:re].sum(axis=1)

    delta = np.abs(deg_init - deg_proj)
    deg_check = (deg_init > 0) & (delta > deg_tol * deg_init)
    div_proj = np.where(deg_proj > 0, deg_proj, 1.0)
    deg_scale = np.where(deg_check, deg_init / div_proj, 1.0)

    A_proj[rs:re] *= np.reshape(deg_scale, (n, 1))
    A_proj[rs:re] *= mask[rs:re]
    A_proj[rs:re] = np.clip(A_proj[rs:re], 0, 1 - 1e-10)

  # Enforce symmetry again, in case the previous change broke it
  A_proj = EXP3_enforce_symmetry(A_proj, n, m)

  # Limit total weight changes (Frobenius ball)
  diff = A_proj - A_init
  norm_diff = np.linalg.norm(diff, "fro")
  norm_max = epsilon * np.linalg.norm(A_init, "fro")
  if norm_diff > norm_max:
    A_proj = A_init + (norm_max / norm_diff) * diff
    A_proj *= mask
    A_proj = np.clip(A_proj, 0, 1 - 1e-10)
    A_proj = EXP3_enforce_symmetry(A_proj, n, m)

  return A_proj


# ===== Main Function =====

def EXP3_min_disagreement(nw, lr_factor, epsilon, max_iters=100, tol=1e-3, reg=False, gamma=0.0, show=True):
  """
  Obtains the adjacency matrices that minimize disagreement for several consecutive social processes. The
  optimization is performed through the gradient descent method.

  Args:
    nw (MDNetwork): The network to work on.
    lr_factor (float): The learning rate will be <lr_factor> of A's norm.
    epsilon (float): Constraint parameter representing how much weights can change.
    max_iters (int): The maximum amount of iterations.
    tol (float): The tolerance set for the process to finish.
    reg (bool): Indicates if the process should be regularized or not.
    gamma (float): Parameter from the optimization function when the process is regularized.
    show (bool): True if intermediate disagreement values are displayed.

  Returns:
    dict: A dictionary containing per-topic metrics for each iteration.
    csr_matrix: The last calculated A.
  """
  A_init = nw.A.toarray()

  labels = ["P", "D", "I", "D_st", "D_ct"]
  data = {l: [] for l in labels}

  data["f"] = lr_factor
  data["e"] = epsilon
  data["i"] = 0
  data["m"] = nw.m
  data["n"] = sum(nw.sizes)

  z_eq = nw.get_z_eq()
  metrics = nw.get_metrics(z=z_eq)
  Dz_terms = nw.get_Dz_terms(z=z_eq)
  for m, l in zip(metrics + Dz_terms, labels):
    data[l].append(m)

  Dz_prev = np.mean(metrics[1])
  if show:
    print(f"[0] Dz* = {Dz_prev:.4f}")

  for i in range(1, max_iters + 1):
    # Obtain gradient
    grad = EXP3_gradient_A(nw, z_eq, reg=reg, gamma=gamma)

    # Calculate learning rate to be proportional to A and the gradient
    A_prev = nw.A.toarray()
    lr = (lr_factor * np.linalg.norm(A_prev, "fro")) / np.linalg.norm(grad, "fro")

    # Calculate gradient step and subtract it from previous A
    A_new = A_prev - (grad * lr)

    # Project onto constraints
    A_proj = EXP3_project_A(nw, A_new, A_init, epsilon)

    # Update network with obtained matrix
    A_csr = sp.csr_matrix(A_proj)
    nw.update_MDN(A_csr)

    # Get new equilibrium vector
    z_eq = nw.get_z_eq()

    # Save new metrics
    metrics = nw.get_metrics(z=z_eq)
    Dz_terms = nw.get_Dz_terms(z=z_eq)
    for m, l in zip(metrics + Dz_terms, labels):
      data[l].append(m)

    Dz_new = np.mean(metrics[1])
    if show:
      print(f"[{i}] Dz* = {Dz_new:.4f}{"  |  (higher!)" if Dz_new > Dz_prev else ""}")

    # Check for convergence
    delta = abs(Dz_prev - Dz_new)
    if delta < tol:
      data["i"] = i
      print(f"Convergence reached at iteration {i}!")
      break

    # Update the last disagreement
    Dz_prev = Dz_new

  else:
    print(f"Maximum iterations reached! Network didn't converge.")

  # Reset network with original matrices
  nw.update_MDN(sp.csr_matrix(A_init))
  return data, A_csr


# ===== Gamma Sensitivity Check =====

def EXP3_gamma_check(nws, gammas, lr_factor, epsilon, s_size):
  """
  It generates a sample of networks, optimizes them for each possible gamma
  value, saving the disagreement results for each case.

  Args:
    nws (list(MDNetwork)): The list with all of the networks.
    gammas (list(float)): The list with all possible gammas.
    lr_factor (float): A fixed learning rate factor.
    epsilon (float): A fixed epsilon.
    s_size (int): The sample size.

  Returns:
    list(list(float)): A list containing the minimization ratio for
                       disagreement, for each network and gamma.
  """
  # Obtain sample and save it
  gam_nws = random.sample(nws, s_size)
  print(f"Sample of size {s_size} created!")
  res_nws = []

  # Optimize for every case
  for i, nw in enumerate(gam_nws):
    print(f"Processing Network {i + 1}/{s_size}...")
    res_gam = []
    for gam in gammas:
      print(f"[$\\gamma$ = {gam}]")
      rr = EXP3_min_disagreement(nw, lr_factor, epsilon, reg=True, gamma=gam, show=False)
      D = rr["D"]
      d_i, d_f = np.mean(D[0]), np.mean(D[-1])
      res_gam.append(d_f / d_i)
    res_nws.append(res_gam)

  # Save list to Pickle file and download it
  #if os.path.exists(filename):
   # os.remove(filename)
  #save_pkl(res_nws, filename)
  #files.download(filename)
  print(f"Sample optimization complete!")

  # Return the list
  return res_nws


# ===== Apply Recommendation Algorithm =====
def EXP3_apply_rec_alg(nws, lr_factor, epsilon, gamma):
  """
  Applies the recommendation algorithm on the given networks.

  Args:
    nws (list(MDNetwork)): The full list of networks.
    lr_factor (float): A fixed learning rate factor.
    epsilon (float): A fixed epsilon parameter.
    gamma (float): A fixed gamma parameter.
  """
  # Create variables to save results in
  nrs = []
  rrs  = []

  for i, nw in enumerate(nws):
    print(f"Processing Network {i + 1}/{len(nws)}...")
    nr = EXP3_min_disagreement(nw, lr_factor, epsilon, show=False)
    print(f"> Non-Regularized Process: Done!")
    rr = EXP3_min_disagreement(nw, lr_factor, epsilon, reg=True, gamma=gamma, show=False)
    print(f"> Regularized Process: Done!")
    nrs.append(nr), rrs.append(rr)

  # Save lists to Pickle files and download them
  #if os.path.exists(nrs_file): os.remove(nrs_file)
  #if os.path.exists(rs_file):  os.remove(rs_file)
  #save_pkl(nrs, nrs_file), save_pkl(rs, rs_file)
  #files.download(nrs_file), files.download(rs_file)
  print(f"Minimization processes done!")

  # Return lists
  return nrs, rrs


def EXP3_setup(num, nws_file, gam_file, gammas, lr_factor, epsilon, s_size, load_func, save_func):
  q_nws = input("Do you have a file with networks? (y/n) ")
  if q_nws == "n":
    nws = EXP3_create_nws(num)
    save_func(nws, nws_file)
  elif q_nws == "y":
    nws = load_func(nws_file)
  else:
    print("Try again!")
    return
  print("Networks saved!")

  q_gam = input("Do you have a file with the gamma check? (y/n) ")
  if q_gam == "n":
    min_gam = EXP3_gamma_check(nws, gammas, lr_factor, epsilon, s_size)
    save_func(min_gam, gam_file)
  elif q_gam == "y":
    min_gam = load_func(gam_file)
  else:
    print("Try again!")
    return
  print("Gamma information saved!")

  print("Ready to continue!")
  return nws, min_gam