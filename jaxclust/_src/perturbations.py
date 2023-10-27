import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Any, Union

class Normal:
  """Normal distribution."""

  def sample(self,
             seed: jax.random.PRNGKey,
             sample_shape: Tuple[int]) -> jax.Array:
    return jax.random.normal(seed, sample_shape)

  def log_prob(self, inputs: jax.Array) -> jax.Array:
    return -0.5 * inputs ** 2


def make_pert_flp_solver(flp_solver : Callable,
                         constrained : bool,
                         num_samples: int = 1000,
                         noise=Normal(),
                         control_variate : bool = False) -> Callable:
    """Creates a perturbed solver of the maximum weight k-connected-component
    forest lp (flp).

    Args:
        flp_solver (Callable): an flp solver from `jaxclust.solvers`.
        constrained (bool): indicates if `flp_solver` takes constraints.
        num_samples (int, optional):  number of samples for MC estimator. Defaults to 1000.
        noise (Class, optional): noise distribution. Defaults to Normal().
        control_variate (bool) : use a control variate for jacobians of adjacency and connectivity matrix.

    Returns:
        Callable: an flp solver taking the same args as `flp_solver`
        as well the additional arguments (following those of `flp_solver`):
            sigma (float): magnitude of noise. 
            rng (jax.random.PRNGKey).
    """

    if not constrained:


        @jax.custom_jvp
        def forward_pert(S, ncc, sigma, rng):

            samples = noise.sample(seed=rng,
                                   sample_shape=(num_samples,) + S.shape)

            Ak_z, Mk_z = jax.vmap(flp_solver,
                                  in_axes=(0, None))(S + sigma * samples, ncc)

            # perturbed argmax and its corresponding coincidence / cluster connectivity matrix
            Akeps = jnp.mean(Ak_z, axis=0)
            Mkeps = jnp.mean(Mk_z, axis=0)

            # perturbed value
            max_values = jnp.einsum('nd,nd->n',
                                    jnp.reshape(S + sigma * samples,
                                                (num_samples, -1)),
                                    jnp.reshape(Ak_z, (num_samples, -1)))
            Fkeps = jnp.mean(max_values)

            return Akeps, Fkeps, Mkeps


        def pert_jvp(tangent, _, S, ncc, sigma, rng):

            samples = noise.sample(seed=rng,
                                   sample_shape=(num_samples,) + S.shape)


            Ak_z, Mk_z = jax.vmap(flp_solver,
                                  in_axes=(0, None))(S + sigma * samples, ncc)


            nabla_z_flat = -jax.vmap(jax.grad(noise.log_prob))(samples.reshape([-1]))

            tangent_Ak_z_flat = 1.0 / (num_samples * sigma) * jnp.einsum(
                'nd,ne,e->d',
                jnp.reshape(Ak_z, (num_samples, -1)),
                jnp.reshape(nabla_z_flat, (num_samples, -1)),
                jnp.reshape(tangent, (-1,)))

            tangent_Mk_z_flat = 1.0 / (num_samples * sigma) * jnp.einsum(
                'nd,ne,e->d',
                jnp.reshape(Mk_z, (num_samples, -1)),
                jnp.reshape(nabla_z_flat, (num_samples, -1)),
                jnp.reshape(tangent, (-1,)))

            jvp_Ak_z = jnp.reshape(tangent_Ak_z_flat, S.shape)
            jvp_Mk_z = jnp.reshape(tangent_Mk_z_flat, S.shape)

            # max jvp
            pert_argmax = jnp.mean(Ak_z, axis=0)
            jvp_Fk_z = jnp.sum(pert_argmax * tangent)
            return jvp_Ak_z, jvp_Fk_z, jvp_Mk_z

        def pert_jvp_control_variate(tangent, _, S, ncc, sigma, rng):

            samples = noise.sample(seed=rng,
                                   sample_shape=(num_samples,) + S.shape)


            Ak_z, Mk_z = jax.vmap(flp_solver,
                                  in_axes=(0, None))(S + sigma * samples, ncc)

            Ak, Mk = flp_solver(S, ncc)

            #argmax jvp
            nabla_z_flat = -jax.vmap(jax.grad(noise.log_prob))(samples.reshape([-1]))

            tangent_Ak_z_flat = 1.0 / (num_samples * sigma) * jnp.einsum(
                'nd,ne,e->d',
                jnp.reshape(Ak_z - Ak, (num_samples, -1)),
                jnp.reshape(nabla_z_flat, (num_samples, -1)),
                jnp.reshape(tangent, (-1,)))

            tangent_Mk_z_flat = 1.0 / (num_samples * sigma) * jnp.einsum(
                'nd,ne,e->d',
                jnp.reshape(Mk_z - Mk, (num_samples, -1)),
                jnp.reshape(nabla_z_flat, (num_samples, -1)),
                jnp.reshape(tangent, (-1,)))

            jvp_Ak_z = jnp.reshape(tangent_Ak_z_flat, S.shape)
            jvp_Mk_z = jnp.reshape(tangent_Mk_z_flat, S.shape)

            # max jvp
            pert_argmax = jnp.mean(Ak_z, axis=0)
            jvp_Fk_z = jnp.sum(pert_argmax * tangent)
            return jvp_Ak_z, jvp_Fk_z, jvp_Mk_z


        if not control_variate:
            forward_pert.defjvps(pert_jvp, None, None, None)
        elif control_variate:
            forward_pert.defjvps(pert_jvp_control_variate, None, None, None)

        return forward_pert


    else:

        @jax.custom_jvp
        def forward_pert(S, ncc, C, sigma, rng):
            samples = noise.sample(seed=rng, sample_shape=(num_samples,) + S.shape)
            AkM_z, MkM_z = jax.vmap(flp_solver, in_axes=(0, None, None))(S + sigma * samples, ncc, C)

            AkMeps = jnp.mean(AkM_z, axis=0)
            MkMeps = jnp.mean(MkM_z, axis=0)

            max_values = jnp.einsum('nd,nd->n',
                                    jnp.reshape(S + sigma * samples,
                                                (num_samples, -1)),
                                    jnp.reshape(AkM_z, (num_samples, -1)))

            FkMeps = jnp.mean(max_values)
            return AkMeps, FkMeps, MkMeps


        def pert_jvp(tangent, _, S, ncc, C, sigma, rng):
            samples = noise.sample(seed=rng, sample_shape=(num_samples,) + S.shape)
            Ak_z, Mk_z = jax.vmap(flp_solver, in_axes=(0, None, None))(S + sigma * samples, ncc, C)

            nabla_z_flat = -jax.vmap(jax.grad(noise.log_prob))(samples.reshape([-1]))

            tangent_Ak_z_flat = 1.0 / (num_samples * sigma) * jnp.einsum(
                'nd,ne,e->d',
                jnp.reshape(Ak_z, (num_samples, -1)),
                jnp.reshape(nabla_z_flat, (num_samples, -1)),
                jnp.reshape(tangent, (-1,)))

            tangent_Mk_z_flat = 1.0 / (num_samples * sigma) * jnp.einsum(
                'nd,ne,e->d',
                jnp.reshape(Mk_z, (num_samples, -1)),
                jnp.reshape(nabla_z_flat, (num_samples, -1)),
                jnp.reshape(tangent, (-1,)))

            jvp_Ak_z = jnp.reshape(tangent_Ak_z_flat, S.shape)
            jvp_Mk_z = jnp.reshape(tangent_Mk_z_flat, S.shape)

            # max jvp
            pert_argmax = jnp.mean(Ak_z, axis=0)
            jvp_Fk_z = jnp.sum(pert_argmax * tangent)
            return jvp_Ak_z, jvp_Fk_z, jvp_Mk_z

        def pert_jvp_control_variate(tangent, _, S, ncc, C, sigma, rng):
            samples = noise.sample(seed=rng, sample_shape=(num_samples,) + S.shape)
            Ak_z, Mk_z = jax.vmap(flp_solver, in_axes=(0, None, None))(S + sigma * samples, ncc, C)
            Ak, Mk = flp_solver(S, ncc, C)

            nabla_z_flat = -jax.vmap(jax.grad(noise.log_prob))(samples.reshape([-1]))

            tangent_Ak_z_flat = 1.0 / (num_samples * sigma) * jnp.einsum(
                'nd,ne,e->d',
                jnp.reshape(Ak_z - Ak, (num_samples, -1)),
                jnp.reshape(nabla_z_flat, (num_samples, -1)),
                jnp.reshape(tangent, (-1,)))

            tangent_Mk_z_flat = 1.0 / (num_samples * sigma) * jnp.einsum(
                'nd,ne,e->d',
                jnp.reshape(Mk_z - Mk, (num_samples, -1)),
                jnp.reshape(nabla_z_flat, (num_samples, -1)),
                jnp.reshape(tangent, (-1,)))

            jvp_Ak_z = jnp.reshape(tangent_Ak_z_flat, S.shape)
            jvp_Mk_z = jnp.reshape(tangent_Mk_z_flat, S.shape)

            # max jvp
            pert_argmax = jnp.mean(Ak_z, axis=0)
            jvp_Fk_z = jnp.sum(pert_argmax * tangent)
            return jvp_Ak_z, jvp_Fk_z, jvp_Mk_z

        if not control_variate:
            forward_pert.defjvps(pert_jvp, None, None, None, None)
        elif control_variate:
            forward_pert.defjvps(pert_jvp_control_variate, None, None, None, None)

        return forward_pert






