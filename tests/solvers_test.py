import jax
import jax.numpy as jnp

from jaxclust._src import test_util
from jaxclust._src.utils import pairwise_square_distance
from jaxclust._src.solvers import kruskals, kruskals_prims_pre
from jaxclust._src.solvers import ckruskals, ckruskals_prims_post

kruskals = jax.jit(kruskals)
kruskals_prims_pre = jax.jit(kruskals_prims_pre)
ckruskals = jax.jit(ckruskals)
ckruskals_prims_post = jax.jit(ckruskals_prims_post)
pairwise_square_distance = jax.jit(pairwise_square_distance)

class UnconstrainedTest(test_util.JAXTestCase):


    def setUp(self):
        super().setUp()
        self.rng = jax.random.PRNGKey(0)

    def test_trivial_two_points_1cc(self):
        X = jax.random.normal(self.rng, (2, 3))
        D = pairwise_square_distance(X)
        S = - D 
        C = jnp.zeros_like(S)

        A, M = kruskals(S, 1)
        Ap, Mp = kruskals_prims_pre(S, 1)
        cA, cM = ckruskals(S, 1, C)
        cAp, cMp = ckruskals_prims_post(S, 1, C)

        T = jnp.ones_like(S)

        self.assertArraysEqual(A, T)
        self.assertArraysEqual(M, T)

        self.assertArraysEqual(cA, T)
        self.assertArraysEqual(cM, T)

        self.assertArraysEqual(Ap, T)
        self.assertArraysEqual(Mp, T)

        self.assertArraysEqual(cA, T)
        self.assertArraysEqual(cM, T)


    def test_trivial_two_points_2cc(self):
        X = jax.random.normal(self.rng, (2, 3))
        D = pairwise_square_distance(X)
        S = - D 
        C = jnp.zeros_like(D)

        A, M = kruskals(S, 2)
        Ap, Mp = kruskals_prims_pre(S, 2)
        cA, cM = ckruskals(S, 2, C)
        cAp, cMp = ckruskals_prims_post(S, 2, C)

        T = jnp.eye(2)

        self.assertArraysEqual(A, T)
        self.assertArraysEqual(M, T)

        self.assertArraysEqual(cA, T)
        self.assertArraysEqual(cM, T)

        self.assertArraysEqual(Ap, T)
        self.assertArraysEqual(Mp, T)

        self.assertArraysEqual(cA, T)
        self.assertArraysEqual(cM, T)



    def test_simple_example_all_ncc(self):

        X = jax.random.normal(self.rng, (4, 3))
        D = pairwise_square_distance(X)
        S = - D 

        kA1, kM1 = kruskals(S, 1)
        kA2, kM2 = kruskals(S, 2)
        kA3, kM3 = kruskals(S, 3)
        kA4, kM4 = kruskals(S, 4)

        kpA1, kpM1 = kruskals(S, 1)
        kpA2, kpM2 = kruskals(S, 2)
        kpA3, kpM3 = kruskals(S, 3)
        kpA4, kpM4 = kruskals(S, 4)

        # running the constrained versions with no contraints should give same results
        C = jnp.zeros_like(S)

        ckA1, ckM1 = ckruskals(S, 1, C)
        ckA2, ckM2 = ckruskals(S, 2, C)
        ckA3, ckM3 = ckruskals(S, 3, C)
        ckA4, ckM4 = ckruskals(S, 4, C)



        ckpA1, ckpM1 = ckruskals_prims_post(S, 1, C)
        ckpA2, ckpM2 = ckruskals_prims_post(S, 2, C)
        ckpA3, ckpM3 = ckruskals_prims_post(S, 3, C)
        ckpA4, ckpM4 = ckruskals_prims_post(S, 4, C)



        A4 = jnp.eye(4)
        M4 = jnp.eye(4)


        A3 = jnp.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=jnp.float32)

        M3 = jnp.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=jnp.float32)


        A2 = jnp.array([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=jnp.float32)

        M2 = jnp.array([
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 1]
        ], dtype=jnp.float32)

        A1 = jnp.array([
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ], dtype=jnp.float32)

        M1 = jnp.ones((4, 4))


        # check matrices are correct for unconstrained methods
        self.assertArraysEqual(A1, kA1)
        self.assertArraysEqual(A1, kpA1)
        self.assertArraysEqual(M1, kM1)
        self.assertArraysEqual(M1, kpM1)

        self.assertArraysEqual(A2, kA2)
        self.assertArraysEqual(A2, kpA2)
        self.assertArraysEqual(M2, kM2)
        self.assertArraysEqual(M2, kpM2)

        self.assertArraysEqual(A3, kA3)
        self.assertArraysEqual(A3, kpA3)
        self.assertArraysEqual(M3, kM3)
        self.assertArraysEqual(M3, kpM3)

        self.assertArraysEqual(A4, kA4)
        self.assertArraysEqual(A4, kpA4)
        self.assertArraysEqual(M4, kM4)
        self.assertArraysEqual(M4, kpM4)


        # check matrices are correct for constrained methods (with no constraints)


        self.assertArraysEqual(A1, ckA1)
        self.assertArraysEqual(A1, ckpA1)
        self.assertArraysEqual(M1, ckM1)
        self.assertArraysEqual(M1, ckpM1)

        self.assertArraysEqual(A2, ckA2)
        self.assertArraysEqual(A2, ckpA2)
        self.assertArraysEqual(M2, ckM2)
        self.assertArraysEqual(M2, ckpM2)

        self.assertArraysEqual(A3, ckA3)
        self.assertArraysEqual(A3, ckpA3)
        self.assertArraysEqual(M3, ckM3)
        self.assertArraysEqual(M3, ckpM3)

        self.assertArraysEqual(A4, ckA4)
        self.assertArraysEqual(A4, ckpA4)
        self.assertArraysEqual(M4, ckM4)
        self.assertArraysEqual(M4, ckpM4)


    def test_full_cluster_nx(self):
        X = jax.random.normal(self.rng, (6, 3))
        D = pairwise_square_distance(X)
        S = - D 
        C = jnp.zeros_like(S)
        A = jnp.array([
            [1, 0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [1, 1, 1, 0, 0, 1]
        ], dtype=jnp.float32)
        M = jnp.ones_like(A)


        kA, kM = kruskals(S, 1)
        ckA, ckM = ckruskals(S, 1, C)
        kpA, kpM = kruskals_prims_pre(S, 1)
        ckpA, ckpM = ckruskals_prims_post(S, 1, C)


        self.assertArraysEqual(kA, A)
        self.assertArraysEqual(kpA, A)
        self.assertArraysEqual(ckA, A)
        self.assertArraysEqual(ckpA, A)

        self.assertArraysEqual(kM, M)
        self.assertArraysEqual(kpM, M)
        self.assertArraysEqual(ckM, M)
        self.assertArraysEqual(ckpM, M)




class Constrained_clustering(test_util.JAXTestCase):

    def test_4_point_example_1cc_C2(self):

        X = jnp.array([[-1, -1], [1, -1], [4, 0], [-6, 0]], dtype=jnp.float32)
        D = pairwise_square_distance(X)
        S = - D 

        C = jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=jnp.float32)

        cA, _ = ckruskals(S, 1, C)
        cpA, _ = ckruskals_prims_post(S, 1, C)


        A1 = jnp.array([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1]
        ], dtype=jnp.float32)


        A2 = jnp.array([
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ], dtype=jnp.float32)


        self.assertArraysEqual(A1, cA)
        self.assertArraysEqual(A2, cpA)

        self.assertLessEqual(jnp.sum(A2 * D), jnp.sum(A1 * D))



    def test_4_point_example_2cc_C1(self):

        X = jnp.array([[-1, -1], [1, -1], [4, 0], [-6, 0]], dtype=jnp.float32)
        D = pairwise_square_distance(X)
        S = - D 

        C = jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=jnp.float32)

        cA, _ = ckruskals(S, 2, C)
        cpA, _ = ckruskals_prims_post(S, 2, C)


        A1 = jnp.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ], dtype=jnp.float32)



        self.assertArraysEqual(A1, cA)
        self.assertArraysEqual(A1, cpA)
        self.assertLessEqual(jnp.sum(cpA * D), jnp.sum(cA * D))




    def test_4_point_example_2cc_C2(self):

        X = jnp.array([[-1, -1], [1, -1], [4, 0], [-6, 0]], dtype=jnp.float32)
        D = pairwise_square_distance(X)
        S = - D 

        C = jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=jnp.float32)

        cA, _ = ckruskals(S, 2, C)
        cpA, _ = ckruskals_prims_post(S, 2, C)


        A1 = jnp.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ], dtype=jnp.float32)



        self.assertArraysEqual(A1, cA)
        self.assertArraysEqual(A1, cpA)
        self.assertLessEqual(jnp.sum(cpA * D), jnp.sum(cA * D))




class Postprocessing_with_prims_leads_to_better_tree(test_util.JAXTestCase):

    def setUp(self):
        super().setUp()
        self.rng = jax.random.PRNGKey(0)

    def find_best_and_worst_edge(self, D):
        triu_ids = jnp.triu_indices_from(D, k=1)
        k_best = jnp.argmin(D[triu_ids])
        k_worst = jnp.argmax(D[triu_ids])

        e_best = (triu_ids[0][k_best], triu_ids[1][k_best])
        e_worst = (triu_ids[0][k_worst], triu_ids[1][k_worst])
        return e_best, e_worst


    def test_weight_of_constrained_forest(self):
        X = jax.random.normal(self.rng, shape=(100, 3))
        D  = pairwise_square_distance(X)
        S = - D 

        e_best, e_worst = self.find_best_and_worst_edge(D)
        C = jnp.zeros_like(D)
        i, j = e_best
        k, l = e_worst

        # mnl for first edge
        C = C.at[i, j].set(-1)
        C = C.at[j, i].set(-1)

        # must link for worst edge
        C = C.at[k, l].set(1)
        C = C.at[l, k].set(1)


        A, M = ckruskals(S, 10, C)
        Ap, Mp = ckruskals_prims_post(S, 10, C)


        F = jnp.sum(A * D)
        Fp = jnp.sum(Ap * D)


        self.assertLessEqual(Fp, F, msg='Prims post-processing should lead to a better or identical solution')
        self.assertArraysEqual(M, Mp)



    def test_weight_of_constrained_forest_2(self):
        X = jax.random.normal(self.rng, shape=(50, 3))
        D  = pairwise_square_distance(X)
        S = - D 

        e_best, e_worst = self.find_best_and_worst_edge(D)
        C = jnp.zeros_like(D)
        i, j = e_best
        k, l = e_worst

        # mnl for first edge
        C = C.at[i, j].set(-1)
        C = C.at[j, i].set(-1)

        # must link for worst edge
        C = C.at[k, l].set(1)
        C = C.at[l, k].set(1)


        A, M = ckruskals(S, 15, C)
        Ap, Mp = ckruskals_prims_post(S, 15, C)


        F = jnp.sum(A * D)
        Fp = jnp.sum(Ap * D)


        self.assertLessEqual(Fp, F, msg='Prims post-processing should lead to a better or identical solution')
        self.assertArraysEqual(M, Mp)



class Valid_coincidence_matrices_and_postprocessing(test_util.JAXTestCase):



    def Assert_valid_coincidence_matrix(self, M, C):

        # check for mnl violations
        self.assertArraysEqual(jnp.where(C == -1, 1, 0) * M, jnp.zeros_like(M), check_dtypes=False)

        # check for ml violations
        self.assertArraysEqual(jnp.where(C == 1, 1, 0), jnp.where(C == 1, M, 0), check_dtypes=False)



    def Assert_improved_weight(self, Ap, A, D):
        self.assertLessEqual(jnp.sum(D * Ap), jnp.sum(D * A))


    def generate_D_and_C(self, key, n=100, nclasses=10, p=0.9):
        key1, key2, key3 = jax.random.split(key, 3)
        X = jax.random.normal(key1, shape=(n, 3))
        Y = jax.random.randint(key2, shape=(n, ), minval=0, maxval=nclasses)
        mask = jax.random.bernoulli(key3, p, (n, 1))
        mask_hot = mask @ mask.T
        Yhot = jax.nn.one_hot(Y, nclasses)

        C  = jnp.where(Yhot @ Yhot.T == 1, 1, -1) * mask_hot
        return pairwise_square_distance(X), C


    def test_valid_coincidence_and_postprocessing(self):
        nclasses = 5
        keys = jax.random.split(jax.random.PRNGKey(0), 5)
        ps = [0.3, 0.4, 0.5, 0.6, 0.7]
        for i in range(5):
            key = keys[i]
            p = ps[i]
            D, C = self.generate_D_and_C(key, nclasses=nclasses, p=p)
            S = -D
            A, M = ckruskals(S, nclasses, C)
            Ap, Mp = ckruskals_prims_post(S, nclasses, C)

            self.Assert_improved_weight(Ap, A, D)
            self.Assert_valid_coincidence_matrix(M, C)
            self.Assert_valid_coincidence_matrix(Mp, C)
