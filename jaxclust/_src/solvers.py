import jax.numpy as jnp
import jax
from typing import Tuple, Callable, Dict, Any, Union
from jaxclust._src.prims import prims, prims_cc
from jaxclust._src.forests import build_forest, build_mnn_forest

# Types of different solvers (constrained vs unconstrained), (perturbed vs classic)
# Solver = Callable[
#     Tuple[jax.Array, int],
#     Tuple[jax.Array, jax.Array]
# ]
# CSolver = Callable[
#     Tuple[jax.Array, int, jax.Array],
#     Tuple[jax.Array, jax.Array]
# ]


def kruskals(D : jax.Array, ncc : int) -> Tuple[jax.Array, jax.Array]:
    '''
    Performs kruskals algorithm on the fully connected graph with
    weights given by D, constructing the minimum weight spanning
    ncc-forest.

    Inputs:
        D : jax.Array, distance matrix.
        ncc : int, number of connected components.
    Outputs:
        A : jax.Array, adjacency matrix of forest.
        M : jax.Array, cluster equivalence matrix of forest.

    A_ij = 1 if the edge (i, j) is in the forest.
    M_ij = 1 if (i, j) are in the same connected component of the forest.
    '''


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


def kruskals_prims_pre(D : jax.Array, ncc : int) -> Tuple[jax.Array, jax.Array]:
    '''
    Performs kruskals algorithm on the fully connected graph with
    weights given by D, constructing the minimum weight spanning
    ncc-forest. First calculates the full-spanning tree using
    prims algorithm, in order to extract the edges in order to
    caculate the spanning-forest (with a second pass of kruskals).

    Inputs:
        D : jax.Array, distance matrix.
        ncc : int, number of connected components.
    Outputs:
        A : jax.Array, adjacency matrix of forest.
        M : jax.Array, cluster equivalence matrix of forest.

    A_ij = 1 if the edge (i, j) is in the forest.
    M_ij = 1 if (i, j) are in the same connected component of the forest.
    '''

    n = D.shape[0]
    nsteps = n - ncc
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


def ckruskals(D : jax.Array,
              ncc : int,
              C : jax.Array) ->  Tuple[jax.Array, jax.Array]:

    '''
    Performs kruskals algorithm on the fully connected graph with
    weights given by D, respecting must-link and must-not-link
    constraints given by the matrix C. Note this problem is no
    longer a matroid structure like kruskals algorithm, and is
    hence only a heuristic. The must-link edges are considered
    first by the algorithm, and before any edge is added
    there is a check to ensure no must-not-link constraint
    is violated.

    Inputs:
        D : jax.Array, distance matrix.
        ncc : int, number of connected components.
        C : jax.Array, constraint matrix (see-below for more info).
    Outputs:
        A : jax.Array, adjacency matrix of forest.
        M : jax.Array, cluster equivalence matrix of forest.

    A_ij = 1 if the edge (i, j) is in the forest.
    M_ij = 1 if (i, j) are in the same connected component of the forest.
    C_ij = 1 if (i, j) has a must-link constraint.
    C_ij = -1 if (i, j) has a must-not-link constraint.
    C_ij = 0 if (i, j) has no constraint.
    '''

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


def ckruskals_prims_post(D : jax.Array,
                         ncc : int,
                         C : jax.Array) ->  Tuple[jax.Array, jax.Array]:

    '''
    Performs kruskals algorithm on the fully connected graph with
    weights given by D, respecting must-link and must-not-link
    constraints given by the matrix C. Note this problem is no
    longer a matroid structure like kruskals algorithm, and is
    hence only a heuristic. The must-link edges are considered
    first by the algorithm, and before any edge is added
    there is a check to ensure no must-not-link constraint
    is violated. Afterwards, prims algorithm is ran on each connected
    component of the forest, which will guarentee a more better
    (or at worst equal) solution.

    Inputs:
        D : jax.Array, distance matrix.
        ncc : int, number of connected components.
        C : jax.Array, constraint matrix (see-below for more info).
    Outputs:
        A : jax.Array, adjacency matrix of forest.
        M : jax.Array, cluster equivalence matrix of forest.

    A_ij = 1 if the edge (i, j) is in the forest.
    M_ij = 1 if (i, j) are in the same connected component of the forest.
    C_ij = 1 if (i, j) has a must-link constraint.
    C_ij = -1 if (i, j) has a must-not-link constraint.
    C_ij = 0 if (i, j) has no constraint.
    '''



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

    """
    Returns a solver for the tree linear program (flp).
    """


    if not constrained:
        if prims:
            fn = lambda S, ncc : kruskals_prims_pre(-S, ncc)
        else:
            fn = lambda S, ncc : kruskals(-S, ncc)
    elif constrained:
        if prims:
            fn = lambda S, ncc, C : ckruskals_prims_post(-S, ncc, C)
        else:
            fn = lambda S, ncc, C : ckruskals(-S, ncc, C)
    return fn

