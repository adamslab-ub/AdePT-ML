.. AdePT-ML documentation master file

Welcome to AdePT-ML's documentation!
=====================================

AdePT-ML is designed to streamline the training of hybrid physics-informed
neural networks by seamlessly combining PyTorch modules with non-differentiable
physics solvers written in any framework (NumPy, JAX, SciPy, etc.).

Key features:

- **Three physics backprop modes** — full Jacobian, manual VJP, or split-VJP
  (pullback closure from ``jax.vjp``) — choose based on your solver's cost.
- **Flexible input routing** — serial pipelines or arbitrary fan-in/fan-out
  between sub-models via ``HybridConfig.model_inputs``.
- **Gradient clipping** — built into the training loop for stable
  physics-informed training.
- **TensorBoard integration** — per-epoch train/test losses logged
  automatically with auto-incrementing run directories.

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  api


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
