"""Test utilities."""

from absl.testing import absltest
from absl.testing import parameterized

import functools

import jax
import jax.numpy as jnp

import numpy as onp

# Test utilities copied from JAX core so we don't depend on their private API.

_dtype_to_32bit_dtype = {
    onp.dtype('int64'): onp.dtype('int32'),
    onp.dtype('uint64'): onp.dtype('uint32'),
    onp.dtype('float64'): onp.dtype('float32'),
    onp.dtype('complex128'): onp.dtype('complex64'),
}

@functools.lru_cache(maxsize=None)
def _canonicalize_dtype(x64_enabled, dtype):
  """Convert from a dtype to a canonical dtype based on config.x64_enabled."""
  try:
    dtype = onp.dtype(dtype)
  except TypeError as e:
    raise TypeError(f'dtype {dtype!r} not understood') from e

  if x64_enabled:
    return dtype
  else:
    return _dtype_to_32bit_dtype.get(dtype, dtype)


def canonicalize_dtype(dtype):
  return _canonicalize_dtype(jax.config.x64_enabled, dtype)


python_scalar_dtypes : dict = {
  bool: onp.dtype('bool'),
  int: onp.dtype('int64'),
  float: onp.dtype('float64'),
  complex: onp.dtype('complex128'),
}


def _dtype(x):
  return (getattr(x, 'dtype', None) or
          onp.dtype(python_scalar_dtypes.get(type(x), None)) or
          onp.asarray(x).dtype)


float0: onp.dtype = onp.dtype([('float0', onp.void, 0)])


_default_tolerance = {
  float0: 0,
  onp.dtype(onp.bool_): 0,
  onp.dtype(onp.int8): 0,
  onp.dtype(onp.int16): 0,
  onp.dtype(onp.int32): 0,
  onp.dtype(onp.int64): 0,
  onp.dtype(onp.uint8): 0,
  onp.dtype(onp.uint16): 0,
  onp.dtype(onp.uint32): 0,
  onp.dtype(onp.uint64): 0,
  #onp.dtype(_dtypes.bfloat16): 1e-2,
  onp.dtype(onp.float16): 1e-3,
  onp.dtype(onp.float32): 1e-6,
  onp.dtype(onp.float64): 1e-15,
  onp.dtype(onp.complex64): 1e-6,
  onp.dtype(onp.complex128): 1e-15,
}


def default_tolerance():
  if device_under_test() != "tpu":
    return _default_tolerance
  tol = _default_tolerance.copy()
  tol[np.dtype(np.float32)] = 1e-3
  tol[np.dtype(np.complex64)] = 1e-3
  return


def tolerance(dtype, tol=None):
  tol = {} if tol is None else tol
  if not isinstance(tol, dict):
    return tol
  tol = {onp.dtype(key): value for key, value in tol.items()}
  dtype = canonicalize_dtype(onp.dtype(dtype))
  return tol.get(dtype, default_tolerance()[dtype])


def device_under_test():
  return jax.lib.xla_bridge.get_backend().platform


def _assert_numpy_allclose(a, b, atol=None, rtol=None, err_msg=''):
  if a.dtype == b.dtype == float0:
    np.testing.assert_array_equal(a, b, err_msg=err_msg)
    return
  #a = a.astype(np.float32) if a.dtype == _dtypes.bfloat16 else a
  #b = b.astype(np.float32) if b.dtype == _dtypes.bfloat16 else b
  kw = {}
  if atol: kw["atol"] = atol
  if rtol: kw["rtol"] = rtol
  with onp.errstate(invalid='ignore'):
    # TODO(phawkins): surprisingly, assert_allclose sometimes reports invalid
    # value errors. It should not do that.
    onp.testing.assert_allclose(a, b, **kw, err_msg=err_msg)


def is_sequence(x):
  try:
    iter(x)
  except TypeError:
    return False
  else:
    return True


class JAXTestCase(parameterized.TestCase):
  """Base class for tests including numerical checks."""

  def assertArraysEqual(self, x, y, *, check_dtypes=True, err_msg=''):
    """Assert that x and y arrays are exactly equal."""
    if check_dtypes:
      self.assertDtypesMatch(x, y)
    # Work around https://github.com/numpy/numpy/issues/18992
    with onp.errstate(over='ignore'):
      onp.testing.assert_array_equal(x, y, err_msg=err_msg)

  def assertArraysAllClose(self, x, y, *, check_dtypes=True, atol=None,
                           rtol=None, err_msg=''):
    """Assert that x and y are close (up to numerical tolerances)."""
    self.assertEqual(x.shape, y.shape)
    atol = max(tolerance(_dtype(x), atol), tolerance(_dtype(y), atol))
    rtol = max(tolerance(_dtype(x), rtol), tolerance(_dtype(y), rtol))

    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)

    if check_dtypes:
      self.assertDtypesMatch(x, y)

  def assertDtypesMatch(self, x, y, *, canonicalize_dtypes=True):
    if not jax.config.x64_enabled and canonicalize_dtypes:
      self.assertEqual(canonicalize_dtype(_dtype(x)),
                       canonicalize_dtype(_dtype(y)))
    else:
      self.assertEqual(_dtype(x), _dtype(y))

  def assertAllClose(self, x, y, *, check_dtypes=True, atol=None, rtol=None,
                     canonicalize_dtypes=True, err_msg=''):
    """Assert that x and y, either arrays or nested tuples/lists, are close."""
    if isinstance(x, dict):
      self.assertIsInstance(y, dict, msg=err_msg)
      self.assertEqual(set(x.keys()), set(y.keys()), msg=err_msg)
      for k in x.keys():
        self.assertAllClose(x[k], y[k], check_dtypes=check_dtypes, atol=atol,
                            rtol=rtol, canonicalize_dtypes=canonicalize_dtypes,
                            err_msg=err_msg)
    elif is_sequence(x) and not hasattr(x, '__array__'):
      self.assertTrue(
          is_sequence(y) and not hasattr(y, '__array__'), msg=err_msg
      )
      self.assertEqual(len(x), len(y), msg=err_msg)
      for x_elt, y_elt in zip(x, y):
        self.assertAllClose(x_elt, y_elt, check_dtypes=check_dtypes, atol=atol,
                            rtol=rtol, canonicalize_dtypes=canonicalize_dtypes,
                            err_msg=err_msg)
    elif hasattr(x, '__array__') or onp.isscalar(x):
      self.assertTrue(
          hasattr(y, '__array__') or onp.isscalar(y),
          msg=f'{err_msg}: {x} is an array but {y} is not.',
      )
      if check_dtypes:
        self.assertDtypesMatch(x, y, canonicalize_dtypes=canonicalize_dtypes)
      x = onp.asarray(x)
      y = onp.asarray(y)
      self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
                                err_msg=err_msg)
    elif x == y:
      return
    else:
      raise TypeError((type(x), type(y)))
