import jax.numpy as jnp
import jax.lax
from typing import Tuple, Callable, Dict, Any


def initialize_forest(nleafs : int) -> Dict:
    H = {}
    m = 2 * nleafs - 1
    H['size'] = nleafs
    H['root'] = jnp.arange(m)
    H['children'] = jnp.eye(m)
    H['adjacency'] = jnp.eye(nleafs)
    H['nconnected'] = nleafs
    H['t'] = 0
    return H


def first(*args : Tuple[Any]) -> Any:
    return args[0]


def link_adjacency(adjacency, x, y):
    adjacency = adjacency.at[x, y].add(1.0)
    adjacency = adjacency.at[y, x].add(1.0)
    return adjacency


def link_children(children, root, size, rx, ry):
    # all the children of rx and ry
    union_children = children[rx] + children[ry]
    # update children of the new node
    children = children.at[size].add(union_children)
    return children


def link_root(root, children, size, rx, ry):
    '''
    assumes link_children has been called first
    so the current node in question has all its
    children already calculated
    '''

    return jnp.where(children[size] == 1, size, root)


def link_or_ignore(H, x : int, y : int):

    size = H['size']
    root = H['root']
    children = H['children']
    adjacency = H['adjacency']
    nconnected = H['nconnected']
    t  = H['t']

    # find the roots of the two nodes
    rx = root[x]
    ry = root[y]

    # if the roots are not the same the link in the binary tree
    to_link = rx != ry

    # update adjacency matrix
    adjacency = jax.lax.select(to_link,
                               link_adjacency(adjacency, x, y),
                               first(adjacency, x, y)
                              )

    children = jax.lax.select(to_link,
                              link_children(children, root, size, rx, ry),
                              first(children, root, size, rx, ry)
                             )

    root = jax.lax.select(to_link,
                          link_root(root, children, size, rx, ry),
                          first(root, children, size, rx, ry)
                          )

    nconnected -= to_link
    size += to_link
    t += 1

    H['size'] = size
    H['root'] = root
    H['children'] = children
    H['adjacency'] = adjacency
    H['nconnected'] = nconnected
    H['t'] = t
    return H


def constrained_link_or_ignore(H : Dict, mnl : jax.Array, x : int, y : int):

    size = H['size']
    root = H['root']
    children = H['children']
    adjacency = H['adjacency']
    nconnected = H['nconnected']
    t  = H['t']

    # find the roots of the two nodes
    rx = root[x]
    ry = root[y]

    # if the roots are not the same the link in the binary tree
    n = mnl.shape[0]

    no_cycle = rx != ry
    no_mnl_violation = jnp.logical_not(jnp.any(
        jnp.outer(children[rx, :n], children[ry, :n]) * mnl))
    to_link = jnp.logical_and(no_cycle, no_mnl_violation)

    # update adjacency matrix
    adjacency = jax.lax.select(to_link,
                               link_adjacency(adjacency, x, y),
                               first(adjacency, x, y)
                              )

    children = jax.lax.select(to_link,
                              link_children(children, root, size, rx, ry),
                              first(children, root, size, rx, ry)
                             )

    root = jax.lax.select(to_link,
                          link_root(root, children, size, rx, ry),
                          first(root, children, size, rx, ry)
                          )

    nconnected -= to_link
    size += to_link
    t += 1

    H['size'] = size
    H['root'] = root
    H['children'] = children
    H['adjacency'] = adjacency
    H['nconnected'] = nconnected
    H['t'] = t
    return H


def cond_fn(vals : Dict) -> bool:
    H = vals['H']
    cond1 = vals['H']['t'] < len(vals['edges'][0])
    cond2 = vals['H']['nconnected'] != vals['ncc']
    return jnp.logical_and(cond1, cond2)


def unconstrained_loop_body(vals : Dict) -> Dict:

    t = vals['H']['t']
    x, y = (vals['edges'][0][t], vals['edges'][1][t])
    H = link_or_ignore(vals['H'], x, y)
    vals['H'] = H
    return vals

def constrained_loop_body(vals : Dict) -> Dict:
    t = vals['H']['t']
    x, y = (vals['edges'][0][t], vals['edges'][1][t])
    H = constrained_link_or_ignore(vals['H'], vals['mnl'], x, y)
    vals['H'] = H
    return vals


def build_forest(edges : Tuple[jax.Array, jax.Array],
                 ncc : int,
                 D_placeholder : jax.Array) -> Dict:
    n = D_placeholder.shape[0]
    H = initialize_forest(n)
    vals = {
        'H' : H,
        'edges' : edges,
        'ncc' : ncc
    }
    vals = jax.lax.while_loop(cond_fn, unconstrained_loop_body, vals)
    return vals['H']



def build_mnn_forest(edges : Tuple[jax.Array, jax.Array],
                     ncc : int,
                     mnl : jax.Array) -> Dict:
    n = mnl.shape[0]
    H = initialize_forest(n)
    vals = {
        'H' : H,
        'edges' : edges,
        'ncc' : ncc,
        'mnl' : mnl,
    }

    vals = jax.lax.while_loop(cond_fn, constrained_loop_body, vals)
    return vals['H']


