import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp

# ===== Topic Functions =====

def get_vector_topic(z: np.ndarray, t: int, m: int=2) -> np.ndarray:
  """
  Returns the opinions of z on topic t.
  """
  n = len(z) // m
  return z[t * n:(t * n) + n]


def get_matrix_topics(M: np.ndarray, t1: int, t2: int, m: int=2) -> sp.csr_matrix:
  """
  Generates a submatrix of M (which is A, D or L) for topics t1 and t2.
  """
  n = M.shape[0] // m
  row_start, row_end = t1 * n, t1 * n + n
  col_start, col_end = t2 * n, t2 * n + n
  return M[row_start:row_end, col_start:col_end]


# ===== Main Class and Methods =====

class MDNetwork:
  """
  Class representing a multidimensional network with communities/groups, where every node/participant is 
  represented by a vector of opinions in more than one topic.

  Attributes:
    sizes (list(int)): The sizes of the communities.
    probs (list(list(float))): Element [a,b] is the probability of edges between community a and b. It 
                               must be symmetric.
    means (list(list(float))): Element [a,b] is the mean of the intrinsic opinions from community a on 
                               topic b.
    std (float): Standard deviation of the opinion distribution, it must be non-negative.
    m (int): The amount of topics (or dimensions) in the network.
    lb (float): Minimum value of opinions.
    ub (float): Maximum value of opinions.
    s (ndarray): The intrinsic opinions of the participants in each topic.
    A (csr_matrix): A modified adjacency matrix of the network.
    D (csr_matrix): A modified degree matrix of the network.
    L (csr_matrix): A modified Laplacian matrix of the network.
  """

  def __init__(self, sizes=None, probs=None, means=None, std=0.5, m=2, lb=-1.0, ub=1.0):
    """
    Initializes an MDNetwork object.
    """
    # Attributes from parameters
    self.sizes, self.probs, self.means = sizes, probs, means
    self.std, self.m, self.lb, self.ub = std, m, lb, ub

    # Attributes to set, checking if the initialization had values or not
    if (self.sizes and self.probs and self.means):
      self.s = self.random_s()
      self.A = self.random_A()
      self.D = sp.diags(np.asarray(self.A.sum(axis=1)).ravel(), format="csr")
      self.L = self.D - self.A

    # If initialization is empty, set all attributes as None
    else:
      self.s = self.A = self.D = self.L = None


  def random_s(self) -> np.ndarray:
    """
    Defines a random set of intrinsic opinions, using a normal distribution.
    """
    if not self.means:
      s = None
    else:
      means = np.repeat(self.means, self.sizes, axis=0).flatten("F")
      s = np.clip(np.random.default_rng().normal(loc=means, scale=self.std), self.lb, self.ub)
    return s


  def random_A(self, A_1s: np.ndarray=None, max_iters: int=10) -> sp.csr_matrix:
    """
    Defines a random network using the stochastic block model, and returns a modified adjacency matrix.
    """
    # If no matrix is provided, create network
    if A_1s is None:
      # Make sure every node has at least one connection
      for _ in range(max_iters):
        G = nx.stochastic_block_model(self.sizes, self.probs)
        adj = nx.to_numpy_array(G)
        if 0 not in [np.sum(row) for row in adj]:
          break
      else:
        raise ValueError("Couldn't define a valid network with the given constraints.")
    # If A_1s was given, set it as the current adjacency matrix
    else:
      adj = A_1s

    rng = np.random.default_rng()

    # Get indices where the matrix is non-zero
    rows, cols = np.nonzero(np.triu(adj))
    t = len(rows)

    # Lists to store indices and values for csr_matrix construction
    r = []
    c = []
    v = []
    min_diags = []
    m = self.m
    n = sum(self.sizes)

    # Fill diagonal blocks (a = b)
    for a in range(m):
      d = np.round(rng.uniform(low=2e-3, high=1.0, size=t), 3)
      min_diags.append(d.min())

      r_block = rows + a * n
      c_block = cols + a * n

      r.extend(np.concatenate([r_block, c_block]))
      c.extend(np.concatenate([c_block, r_block]))
      v.extend(np.concatenate([d, d]))

    # Fill non-diagonal blocks (a != b)
    for a in range(m):
      for b in range(a + 1, m):
        min_d = min(min_diags[a], min_diags[b])
        nd = np.round(rng.uniform(low=1e-3, high=min_d, size=t), 3)

        r_block_ab = rows + a * n
        c_block_ab = cols + b * n
        r_block_ba = rows + b * n
        c_block_ba = cols + a * n

        r.extend(np.concatenate([r_block_ab, c_block_ab, r_block_ba, c_block_ba]))
        c.extend(np.concatenate([c_block_ab, r_block_ab, c_block_ba, r_block_ba]))
        v.extend(np.concatenate([nd, nd, nd, nd]))

    # Construct and return the final matrix
    N = m * n
    r = np.asarray(r, dtype=int)
    c = np.asarray(c, dtype=int)
    v = np.asarray(v, dtype=float)
    A = sp.csr_matrix((v, (r, c)), shape=(N, N))
    return A


  def update_MDN(self, A: sp.csr_matrix):
    """
    Given a new adjacency matrix, it modifies the internal matrices of the network.
    """
    self.A = A
    self.D = sp.diags(np.asarray(self.A.sum(axis=1)).ravel(), format="csr")
    self.L = self.D - self.A


  def get_z_eq(self) -> np.ndarray:
    """
    Returns the equilibrium set of opinions for every topic.
    """
    N = sum(self.sizes) * self.m
    z_eq = sp.linalg.spsolve(self.L + sp.eye(N, format="csr"), self.s)
    return z_eq


  def get_metrics(self, z: np.ndarray=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns polarization, disagreement and internal conflict of the network using z.
    """
    m = self.m
    s_topics = [get_vector_topic(self.s, a, m) for a in range(m)]
    if z is None:
      z = self.get_z_eq()
    z_topics = [get_vector_topic(z, a, m) for a in range(m)]

    # Initialize metric vectors
    Pz = np.zeros(m)
    Dz = np.zeros(m)
    Iz = np.zeros(m)

    # Iterate through topics
    for a in range(m):
      z_a = z_topics[a]

      # Polarization
      z_bar_a = z_a - np.mean(z_a)
      Pz[a] = np.sum(z_bar_a ** 2)

      # Disagreement
      for b in range(m):
        L_ab = get_matrix_topics(self.L, a, b, m)
        Dz[a] += z_a @ (L_ab @ z_topics[b])

      # Internal conflict
      diff_a = z_a - s_topics[a]
      Iz[a] = np.sum(diff_a ** 2)

    return Pz, Dz, Iz


  def get_Dz_terms(self, z: np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the same and cross topic terms contributing to disagreement, for a given z.
    """
    m = self.m
    if z is None:
      z = self.get_z_eq()

    # Initalize metric vectors
    Dz_same = np.empty(m)
    Dz_cross = np.empty(m)

    # Iterate through topics
    for a in range(m):
      z_a = get_vector_topic(z, a, m)
      for b in range(m):
        l_ab = get_matrix_topics(self.L, a, b, m)
        ct_a = 0
        if a == b:
          st_a = z_a @ l_ab @ z_a
          Dz_same[a] = st_a
        else:
          z_b = get_vector_topic(z, b, m)
          ct_a += z_a @ l_ab @ z_b
      Dz_cross[a] = ct_a

    return Dz_same, Dz_cross


  def get_metrics_1D(self, z: np.ndarray=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the polarization, disagreement and internal conflict of the network as if it was one-dimensional at equilibrium.
    """
    n = sum(self.sizes)
    m = self.m
    s_topics = [get_vector_topic(self.s, a, m) for a in range(m)]
    L_topics = [get_matrix_topics(self.L, a, a, m) for a in range(m)]
    I = sp.eye(n, format="csr")
    if z is None:
      z_topics = [sp.linalg.spsolve(L_topics[a] + I, s_topics[a]) for a in range(m)]
    else:
      z_topics = [get_vector_topic(z, a, m) for a in range(m)]

    # Initialize metric vectors
    Pz_1D = np.zeros(m)
    Dz_1D = np.zeros(m)
    Iz_1D = np.zeros(m)

    # Iterate through topics
    for a in range(m):
      z_a = z_topics[a]

      # Polarization
      z_centered = z_a - np.mean(z_a)
      Pz_1D[a] = z_centered @ z_centered

      # Disagreement
      Dz_1D[a] = z_a @ L_topics[a] @ z_a

      # Internal Conflict
      diff_a = z_a - s_topics[a]
      Iz_1D[a] = diff_a @ diff_a

    return Pz_1D, Dz_1D, Iz_1D


  def update_z(self, z_prev: np.ndarray, d_factor: sp.csr_matrix=None) -> np.ndarray:
    """
    Makes one update to the set of opinions from the network, given the previous set.
    """
    if d_factor is None:
      N = sum(self.sizes) * self.m
      d_factor = self.D + sp.eye(N, format="csr")
    z_factor = self.A @ z_prev + self.s
    z_next = sp.linalg.spsolve(d_factor, z_factor)
    return z_next


  def update_process(self, save_z=False, save_norms=False, tol=1e-5, max_iters=1000):
    """
    Simulates the whole social process on the network.

    Args:
      save_z (bool): True if the list of all intermediate opinions sets must be returned.
      save_norms (bool): True if the lists of norms of opinion differences must be returned.
      tol (float): The minimum difference in steps to stop the process.
      max_iters (int): The maximum amount of iterations before declaring equilibrium.

    Returns:
      int: The amount of rounds until equilibrium.
      list(ndarray): A list of all intermediate opinions sets (empty list if save_z=False).
      list(float): A list of norms of sequential opinions differences (empty list if save_norms=False).
      list(float): A list of norms of differences between intermediate and equilibrium sets (empty 
                   list if save_norms=False).
    """
    z_prev = self.s
    z_eq = self.get_z_eq()
    N = sum(self.sizes) * self.m
    d_factor = self.D + sp.eye(N, format="csr")
    
    # Initialize variables
    rounds = 0
    checks = 0
    ops, norms_seq, norms_eq = [], [], []

    # Update rounds until convergence or maximum iterations
    while rounds < max_iters:
      if save_z:
        ops.append(z_prev)
      z_next = self.update_z(z_prev, d_factor)
      delta_seq = np.linalg.norm(z_next - z_prev)
      if save_norms:
        delta_eq = np.linalg.norm(z_eq - z_prev)
        norms_seq.append(delta_seq)
        norms_eq.append(delta_eq)
      if delta_seq < tol:
        checks += 1
        if checks > 5:
          break
      z_prev = z_next
      rounds += 1

    return rounds, ops, norms_seq, norms_eq


  def copy_MDN(self) -> "MDNetwork":
    """
    Creates a copy of the MDNetwork object.
    """
    # Create new network using the parameters from the original
    new_MDN = MDNetwork(
        sizes = copy.deepcopy(self.sizes),
        probs = [copy.deepcopy(row) for row in self.probs] if self.probs is not None else None,
        means = [copy.deepcopy(row) for row in self.means] if self.means is not None else None,
        std   = self.std,
        m     = self.m,
        lb    = self.lb,
        ub    = self.ub
        )

    # Copy the rest of the elements
    new_MDN.s = copy.deepcopy(self.s)
    new_MDN.A = copy.deepcopy(self.A)
    new_MDN.D = copy.deepcopy(self.D)
    new_MDN.L = copy.deepcopy(self.L)

    return new_MDN