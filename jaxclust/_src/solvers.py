import jax
import jax.numpy as jnp
from jaxclust._src.prims import prims, prims_cc
from typing import Tuple, Callable
from jaxclust._src.forests import build_forest, build_mnn_forest

def kruskals(S : jax.Array, ncc : int) -> Tuple[jax.Array, jax.Array]:
    """Calculates the adjacency matrix and cluster connectivity matrix
    of the minimum weight ncc-spanning forest using Kruskal's algorithm.

    Args:
        S (jax.Array): similarity matrix.
        ncc (int): number of connected components. 

    Returns:
        Tuple[jax.Array, jax.Array]: A, M

    :math:`A_{ij} = 1` if the edge (i, j) is in the forest.

    :math:`M_{ij} = 1` if i and j are in the same connected component of the forest.

    """

    D = -S # uses legacy distance matrix code (before similarity matrix)
    n = D.shape[0]
    triu_ids = jnp.triu_indices_from(D, k=1)
    triu_rows, triu_cols = triu_ids

    # sort by distance
    perm = jnp.argsort(D[triu_ids])
    # (rows, cols) of upper triangular are sorted by distance
    triu_rows = triu_rows[perm]
    triu_cols = triu_cols[perm]
    triu_ids = (triu_rows, triu_cols)

    H = build_forest(triu_ids, ncc, D)
    children = H['children']
    root = H['root']
    M = children[root[jnp.arange(n)]][:, :n]
    A = H['adjacency'][:n, :n]
    return A, M


def kruskals_prims_pre(S : jax.Array, ncc : int) -> Tuple[jax.Array, jax.Array]:
    """Calculates the adjacency matrix and cluster connectivity matrix
    of the minimum weight ncc-spanning forest. Uses Prim's algorithm
    to construct the full spanning tree, then applies Kruskal's algorithm
    to the edges in the spanning tree in order to calculate the forest.

    Args:
        S (jax.Array): similarity matrix.
        ncc (int): number of connected components. 

    Returns:
        Tuple[jax.Array, jax.Array]: A, M

    :math:`A_{ij} = 1` if the edge (i, j) is in the forest.

    :math:`M_{ij} = 1` if i and j are in the same connected component of the forest.
    """
    D = -S # uses legacy distance matrix code (before similarity matrix)
    n = D.shape[0]
    triu_ids = jnp.triu_indices_from(D, k=1)
    triu_rows, triu_cols = triu_ids

    A = prims(D)
    D_masked = jnp.where(A - jnp.eye(n) == 1, D, jnp.inf)
    perm = jnp.argsort(D_masked[triu_rows, triu_cols])

    edges = (triu_rows[perm], triu_cols[perm])
    H = build_forest(edges, ncc, D)
    children = H['children']
    root = H['root']
    M = children[root[jnp.arange(n)]][:, :n]
    A = H['adjacency'][:n, :n]
    return A, M


def ckruskals(S : jax.Array,
              ncc : int,
              C : jax.Array) ->  Tuple[jax.Array, jax.Array]:
    """Calculates the adjacency matrix and cluster connectivity matrix
    of the minimum weight ncc-spanning forest. Uses a biased heuristic
    based on kruskals algorithm to create the forest.

    Args:
        S (jax.Array): similarity matrix.
        ncc (int): number of connected components. 
        C (jax.Array) : constraint matrix.

    Returns:
        Tuple[jax.Array, jax.Array]: A, M

    :math:`A_{ij} = 1` if the edge (i, j) is in the forest.

    :math:`M_{ij} = 1` if i and j are in the same connected component of the forest.

    :math:`C_{ij}=1` if (i, j) has a must-link (ml) constraint. 

    :math:`C_{ij}=-1` if (i, j) has a must-not-link (mnl) constraint. 
    
    :math:`C_{ij}=0` if (i, j) has no constraints.
    """

    D = -S # uses legacy distance matrix code (before similarity matrix)
    n = D.shape[0]
    triu_ids = jnp.triu_indices_from(D, k=1)
    triu_rows, triu_cols = triu_ids

    mnl = jnp.where(C == -1, 1, 0)
    ml_mask = jnp.where(C == 1, 0, 1)
    D_biased = D + 2 * ml_mask * (D.max() - D.min())
    D_biased = jnp.where(mnl == 1, jnp.inf, D_biased)

    # sort by distance
    perm = jnp.argsort(D_biased[triu_ids])
    # (rows, cols) of upper triangular are sorted by distance
    triu_rows = triu_rows[perm]
    triu_cols = triu_cols[perm]
    triu_ids = (triu_rows, triu_cols)

    H =  build_mnn_forest(triu_ids, ncc, mnl)
    children = H['children']
    root = H['root']
    M = children[root[jnp.arange(n)]][:, :n]
    A = H['adjacency'][:n, :n]
    return A, M


def ckruskals_prims_post(S : jax.Array,
                         ncc : int,
                         C : jax.Array) ->  Tuple[jax.Array, jax.Array]:

    """Calculates the adjacency matrix and cluster connectivity matrix
    of the minimum weight ncc-spanning forest. Uses a biased heuristic
    based on kruskals algorithm to create the forest. Afterwards applies
    prims algorithm to recalculate the spanning tree of each connected
    component in the forest (hence guarenteed to obtain a solution
    at least as good if not better than `ckruskals`).

    Args:
        S (jax.Array): similarity matrix.
        ncc (int): number of connected components. 
        C (jax.Array) : constraint matrix.

    Returns:
        Tuple[jax.Array, jax.Array]: A, M

    :math:`A_{ij} = 1` if the edge (i, j) is in the forest.

    :math:`M_{ij} = 1` if i and j are in the same connected component of the forest.

    :math:`C_{ij}=1` if (i, j) has a must-link (ml) constraint. 

    :math:`C_{ij}=-1` if (i, j) has a must-not-link (mnl) constraint. 
    
    :math:`C_{ij}=0` if (i, j) has no constraints.
    """

    D = -S # uses legacy distance matrix code (before similarity matrix)
    n = D.shape[0]
    triu_ids = jnp.triu_indices_from(D, k=1)
    triu_rows, triu_cols = triu_ids

    mnl = jnp.where(C == -1, 1, 0)
    ml_mask = jnp.where(C == 1, 0, 1)
    D_biased = D + 2 * ml_mask * (D.max() - D.min())
    D_biased = jnp.where(mnl == 1, jnp.inf, D_biased)

    # sort by distance
    perm = jnp.argsort(D_biased[triu_ids])
    # (rows, cols) of upper triangular are sorted by distance
    triu_rows = triu_rows[perm]
    triu_cols = triu_cols[perm]
    triu_ids = (triu_rows, triu_cols)

    H =  build_mnn_forest(triu_ids, ncc, mnl)
    children = H['children']
    root = H['root']
    M = children[root[jnp.arange(n)]][:, :n]
    A = H['adjacency'][:n, :n]
    size = H['size']


    root_mask = (root == jnp.arange(len(root))).reshape(-1, 1)
    cc_mask = jnp.where(root_mask, children, 0)[:, :n]

    A_cc = jax.vmap(prims_cc, in_axes=(None, 0))(D, cc_mask)
    A_heur = jnp.sum(A_cc, axis=0) + jnp.eye(n)

    return A_heur, M



def get_flp_solver(constrained : bool=False,
                   use_prims : bool=False
                  ) -> Callable:
    """Returns a solver of the maximum weight k-connected component
    forest lp (flp).

    Args:
        constrained (bool, optional): option for solver to take constraints. Defaults to False.
        use_prims (bool, optional): option for solver to use prims pre/post-processing. Defaults to False.

    Returns:
        Callable: a solver which takes in as inputs S,
        a similarity matrix (jax.Array), ncc (int) and a constraint matrix C
        (jax.Array) if the constrained variable is set to True.
    """


    if not constrained:
        if use_prims:
            return lambda S, ncc : kruskals_prims_pre(S, ncc)
        else:
            return lambda S, ncc : kruskals(S, ncc)
    elif constrained:
        if use_prims:
            return lambda S, ncc, C : ckruskals_prims_post(S, ncc, C)
        else:
            return lambda S, ncc, C : ckruskals(S, ncc, C)
