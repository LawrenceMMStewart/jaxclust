import jax
import jax.numpy as jnp


def pairwise_square_distance(X : jax.Array) -> jax.Array:
    """
    Pair-wise Euclidean square distance D_ij = \| x_i - x_j \|_2^2

    Inputs:
        X : jax.Array
    Outputs:
        D : jax.Array
    """
    n = X.shape[0]
    G = jnp.dot(X, X.T)
    g = jnp.diag(G).reshape(n, 1)
    o = jnp.ones_like(g)
    return jnp.dot(g, o.T) + jnp.dot(o, g.T) - 2 * G


# def argsort_first_k(x : jax.Array, k : int) -> jax.Array:
#     '''
#     A function to perform an partial argsort, where the first k
#     elements are argsorted. The rest of the indicies from k onwards
#     will be the last value of the argsort repeated. For this reason
#     the array size of the output of argsort_first_k will be the same
#     as the input array, and hence it is jit compatible.
#     Inputs:
#         x : array, to be sorted.
#         k : int, number of smallest elements to find.
#     Outputs:
#         ids : array, of indicies of smallest k elements same shape as x.
#     '''

#     ap = jnp.argpartition(x, k - 1)
#     return jnp.where(jnp.arange(ap.shape[0]) < k, ap, ap[k-1])


