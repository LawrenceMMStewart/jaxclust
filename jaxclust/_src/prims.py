import jax
import jax.numpy as jnp
from typing import Tuple, Any, Dict


def prims_update(vals : Dict, _ : int) -> Tuple[Dict, int]:
    '''
    scan body function for prims
    '''
    triu_rows, triu_cols = vals['triu_ids']
    in_tree = vals['in_tree']
    D = vals['D']
    adjacency = vals['adjacency']
    I = vals['I']
    n = D.shape[0]

    out_tree = jnp.logical_not(in_tree)
    mask = jnp.outer(in_tree, in_tree) + jnp.outer(out_tree, out_tree) + I
    D_masked = jnp.where(mask, jnp.inf , D)
    k = jnp.argmin(D_masked[triu_rows, triu_cols]) # argmin from upper-triangular (flattened)
    i, j = triu_rows[k], triu_cols[k] # argmin position for 2d array

    adjacency = adjacency.at[i, j].set(1)
    adjacency = adjacency.at[j, i].set(1)
    in_tree = in_tree.at[i].set(1)
    in_tree = in_tree.at[j].set(1)

    vals['in_tree'] = in_tree
    vals['adjacency'] = adjacency
    return vals, _


def prims(D : jax.Array) -> jax.Array:
    n = D.shape[0]
    vals = {}
    vals['triu_ids'] = jnp.tril_indices(n, k=1)
    vals['adjacency'] = jnp.eye(n)
    vals['D'] = D
    vals['I'] = jnp.eye(n)
    vals['in_tree'] = jnp.zeros(n).at[0].add(1)

    vals, _  = jax.lax.scan(f=prims_update, init=vals, xs=jnp.arange(n-1))
    return vals['adjacency']



def prims_cc_cond(vals):
    return vals['count'] < vals['m'] - 1


def prims_cc_body(vals):
    triu_rows, triu_cols = vals['triu_ids']
    in_tree = vals['in_tree']
    D = vals['D']
    adjacency = vals['adjacency']
    I = vals['I']
    n = D.shape[0]

    out_tree = jnp.logical_not(in_tree)
    mask = jnp.outer(in_tree, in_tree) + jnp.outer(out_tree, out_tree) + I
    D_masked = jnp.where(mask, jnp.inf , D)
    k = jnp.argmin(D_masked[triu_rows, triu_cols]) # argmin from upper-triangular (flattened)
    i, j = triu_rows[k], triu_cols[k] # argmin position for 2d array

    adjacency = adjacency.at[i, j].set(1)
    adjacency = adjacency.at[j, i].set(1)
    in_tree = in_tree.at[i].set(1)
    in_tree = in_tree.at[j].set(1)

    vals['in_tree'] = in_tree
    vals['adjacency'] = adjacency
    vals['count'] += 1
    return vals

def prims_cc(D : jax.Array, cc_mask : jax.Array) -> jax.Array:
    '''
    prims on a connected component
    '''

    n = D.shape[0]
    vals = {}

    D_masked = jnp.where(jnp.outer(cc_mask, cc_mask), D, jnp.inf)
    perm = jnp.argsort(jnp.logical_not(cc_mask))

    vals['triu_ids'] = jnp.tril_indices(n, k=1)
    vals['adjacency'] = jnp.zeros((n, n))
    vals['D'] = D_masked
    vals['I'] = jnp.eye(n)
    vals['in_tree'] = jnp.zeros(n).at[perm[0]].add(1)
    vals['m'] = jnp.sum(cc_mask)
    vals['count'] = 0


    vals = jax.lax.while_loop(prims_cc_cond, prims_cc_body, vals)
    return vals['adjacency']





