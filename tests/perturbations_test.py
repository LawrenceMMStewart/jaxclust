from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp

from jaxclust._src import test_util
from jaxclust._src.utils import pairwise_square_distance
from jaxclust._src.solvers import get_flp_solver
from jaxclust._src.perturbations import make_pert_flp_solver

pairwise_square_distance = jax.jit(pairwise_square_distance)


class NoNoisePerturbationsTest(test_util.JAXTestCase):

    def setUp(self):
        super().setUp()
        self.rng = jax.random.PRNGKey(0)
        self.X = jax.random.normal(self.rng, (32, 3))
        self.D = pairwise_square_distance(self.X)
        self.S = - self.D

    def test_nonoise_kruskals(self):

        constrained = False
        use_prims = False
        ncc = 10
        eps = 0.0
        fn = jax.jit(get_flp_solver(constrained=constrained, use_prims=use_prims))
        pert_fn = make_pert_flp_solver(fn, constrained, num_samples=100, control_variate=False)

        A, M = fn(self.S, ncc)
        pert_A, pert_F, pert_M = pert_fn(self.S, ncc, eps, jax.random.PRNGKey(1))

        self.assertArraysEqual(A, pert_A)
        self.assertArraysEqual(M, pert_M)


    def test_nonoise_kruskals_prims(self):

        constrained = False
        use_prims = True
        ncc = 10
        eps = 0.0
        fn = jax.jit(get_flp_solver(constrained=constrained, use_prims=use_prims))
        pert_fn = make_pert_flp_solver(fn, constrained, num_samples=100, control_variate=False)

        A, M = fn(self.S, ncc)
        pert_A, pert_F, pert_M = pert_fn(self.S, ncc, eps, jax.random.PRNGKey(1))

        self.assertArraysEqual(A, pert_A)
        self.assertArraysEqual(M, pert_M)

    def test_nonoise_ckruskals(self):

        constrained = True
        use_prims = False
        C = jnp.zeros_like(self.S)
        ncc = 10
        eps = 0.0
        fn = jax.jit(get_flp_solver(constrained=constrained, use_prims=use_prims))
        pert_fn = make_pert_flp_solver(fn, constrained, num_samples=100, control_variate=False)

        A, M = fn(self.S, ncc, C)
        pert_A, pert_F, pert_M = pert_fn(self.S, ncc, C, eps, jax.random.PRNGKey(1))

        self.assertArraysEqual(A, pert_A)
        self.assertArraysEqual(M, pert_M)


    def test_nonoise_ckruskals_prims(self):

        constrained = True
        use_prims = True
        C = jnp.zeros_like(self.S)
        ncc = 10
        eps = 0.0

        fn = jax.jit(get_flp_solver(constrained=constrained, use_prims=use_prims))
        pert_fn = make_pert_flp_solver(fn, constrained, num_samples=100, control_variate=False)

        A, M = fn(self.S, ncc, C)
        pert_A, pert_F, pert_M = pert_fn(self.S, ncc, C, eps, jax.random.PRNGKey(1))

        self.assertArraysEqual(A, pert_A)
        self.assertArraysEqual(M, pert_M)





class GradientofMaxisArgmax_test(test_util.JAXTestCase):
    def setUp(self):
        super().setUp()
        self.rng = jax.random.PRNGKey(1)
        self.X = jax.random.normal(self.rng, (32, 3))
        self.D = pairwise_square_distance(self.X)
        self.ncc = 10
        self.eps = 0.1
        self.S = - self.D
        self.C = jnp.zeros_like(self.S)

    def test_gradF(self):
        constrained = False
        B = [False, True]
        for control_variate in B:
            for use_prims in B:
                fn = jax.jit(get_flp_solver(constrained=constrained, use_prims=use_prims))
                pert_fn = make_pert_flp_solver(fn, constrained, num_samples=100, control_variate=control_variate)
                pert_F = lambda *args : pert_fn(*args)[1]
                A, F, M = pert_fn(self.S, self.ncc, self.eps, self.rng)
                G = jax.grad(pert_F)(self.S, self.ncc, self.eps, self.rng)
                self.assertArraysEqual(A, G)

    def test_gradF_constrained(self):
        constrained = True
        B = [False, True]
        for control_variate in B:
            for use_prims in B:
                fn = jax.jit(get_flp_solver(constrained=constrained, use_prims=use_prims))
                pert_fn = make_pert_flp_solver(fn, constrained, num_samples=100, control_variate=control_variate)
                pert_F = lambda *args : pert_fn(*args)[1]
                A, F, M = pert_fn(self.S, self.ncc, self.C, self.eps, self.rng)
                G = jax.grad(pert_F)(self.S, self.ncc, self.C, self.eps, self.rng)
                self.assertArraysEqual(A, G)
