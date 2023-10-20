
JAXClust
====================================

Hardware accelerated, batchable, differentiable clustering in `JAX <https://github.com/google/jax>`_.

- **Hardware accelerated:** implementations run on CPU, GPU and TPU.
  to CPU.
- **JIT-Compatible:** all clustering algorithms are `jax.jit` compatible.


To install the latest release of JAXClust, use the following command::

    pip install jaxclust

Alternatively, it can be be installed from sources with the following command::

    python setup.py install


.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   api.md


.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/cnn-mnist.ipynb



.. toctree::
   :maxdepth: 1
   :caption: About

   Authors <https://github.com/LawrenceMMStewart/jaxclust/graphs/contributors>
   Source code <https://github.com/LawrenceMMStewart/jaxclust>
   Issue tracker <https://github.com/LawrenceMMStewart/jaxclust/issues>


Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/LawrenceMMStewart/jaxclust/issues>`_.

License
-------

JAXClust is licensed under the Apache 2.0 License.


Citing
------

If this software is useful for you, please consider citing :cite:t:`2023:stewart`:

.. bibliography::

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
