from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp

from jaxclust._src import test_util
from jaxclust._src.utils import pairwise_square_distance
from jaxclust._src.prims import prims, prims_cc


prims = jax.jit(prims)
prims_cc = jax.jit(prims_cc)

class Prims_test(test_util.JAXTestCase):

    def test_one_element_prims(self):
        X = jnp.ones((1, 1))
        D = pairwise_square_distance(X)
        o = jnp.ones((1, 1))
        A = prims(D)
        self.assertArraysEqual(A, o)


    def test_two_element_prims(self):
        X = jnp.array([[1.0], [2.0]])
        D = pairwise_square_distance(X)
        o = jnp.ones((2, 2))
        A = prims(D)
        self.assertArraysEqual(A, o)


    def test_simple_example(self):
        X = jnp.array([[1.0], [2.0], [5.0]])
        D = pairwise_square_distance(X)
        A_true = jnp.array([
            [1., 1., 0.],
            [1., 1., 1.],
            [0., 1., 1.]
        ])
        A = prims(D)
        self.assertArraysEqual(A, A_true)

    def test_nx_example_1(self):
        X = jnp.array([1., 3., 7., 12]).reshape(-1, 1)
        D = pairwise_square_distance(X)

        A_true = jnp.array([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1]],
            dtype=jnp.float32)

        A = prims(D)
        self.assertArraysEqual(A, A_true)


    def test_nx_examples_2(self):
        X = jnp.array([-1.0, 2., 7., 8., 15.]).reshape(-1, 1)
        D = pairwise_square_distance(X)

        A_true = jnp.array([
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1]
        ], dtype=jnp.float32)

        A = prims(D)
        self.assertArraysEqual(A, A_true)


class Prims_cc_test(test_util.JAXTestCase):

    def setUp(self):
        super().setUp()
        self.rng = jax.random.PRNGKey(0)


    def test_no_cc(self):
        n = 100
        X = jax.random.normal(self.rng, shape=(n, 2))
        D = pairwise_square_distance(X)
        cc_mask = jnp.zeros(n)
        A_true = jnp.zeros_like(D)
        A = prims_cc(D, cc_mask)
        self.assertArraysEqual(A, A_true)


    def test_cc_1element(self):
        n = 100
        X = jax.random.normal(self.rng, shape=(n, 2))
        D = pairwise_square_distance(X)
        cc_mask = jnp.zeros(n).at[0].set(1.)
        A_true = jnp.zeros_like(D)
        A = prims_cc(D, cc_mask)
        self.assertArraysEqual(A, A_true)


    def test_cc_2elements(self):
        n = 100
        X = jax.random.normal(self.rng, shape=(n, 2))
        D = pairwise_square_distance(X)
        cc_mask = jnp.zeros(n).at[:2].set(1.)

        A_true = jnp.zeros_like(D)
        A_true = A_true.at[0, 1].set(1.)
        A_true = A_true.at[1, 0].set(1.)

        A = prims_cc(D, cc_mask)
        self.assertArraysEqual(A, A_true)


    def test_cc_3elements(self):
        X = jnp.array([1, 3, 6, 10, 15], dtype=jnp.float32).reshape(-1, 1)
        n = X.shape[0]
        D = pairwise_square_distance(X)

        cc_mask = jnp.zeros(n).at[jnp.array([0, -2, -1])].set(1.)
        A_true = jnp.zeros((n, n))

        A_true = A_true.at[0, -2].set(1.)
        A_true = A_true.at[-2, 0].set(1.)

        A_true = A_true.at[-1, -2].set(1.)
        A_true = A_true.at[-2, -1].set(1.)

        A = prims_cc(D, cc_mask)
        self.assertArraysEqual(A, A_true)


    def test_cc_3elements_(self):
        X = jnp.array([1, 3, 6, 10, 15], dtype=jnp.float32).reshape(-1, 1)
        n = X.shape[0]
        D = pairwise_square_distance(X)

        cc_mask = jnp.zeros(n).at[jnp.array([0, 1, -1])].set(1.)
        A_true = jnp.zeros((n, n))

        A_true = A_true.at[0, 1].set(1.)
        A_true = A_true.at[1, 0].set(1.)

        A_true = A_true.at[1, -1].set(1.)
        A_true = A_true.at[-1, 1].set(1.)

        A = prims_cc(D, cc_mask)
        self.assertArraysEqual(A, A_true)


    def cc_full(self, D):
        n = D.shape[0]
        cc_mask = jnp.ones(n)
        return prims_cc(D, cc_mask) + jnp.eye(n)

    def test_cc_full_same_as_prims(self):
        '''
        performing prims_cc with all elements is same as prims
        '''
        key1, key2 = jax.random.split(self.rng)

        X1 = jax.random.normal(key1, shape=(15, 2))
        D1 = pairwise_square_distance(X1)

        self.assertArraysEqual(
            prims(D1),
            self.cc_full(D1)
        )

        X2 = jax.random.normal(key2, shape=(25, 3))
        D2 = pairwise_square_distance(X2)

        self.assertArraysEqual(
            prims(D2),
            self.cc_full(D2)
        )


