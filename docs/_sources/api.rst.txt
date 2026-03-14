API Reference
=============

============
Hybrid Model
============
.. automodule:: adeptml.ensemble
  :members:

=======
Models
=======

AdePT-ML provides three autograd-compatible wrappers for non-differentiable
physics functions, differentiated by how gradients are computed during the
backward pass:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Class
     - ``forward_func`` signature
     - ``jacobian_func`` signature
   * - :class:`~adeptml.models.Physics`
     - ``(x, *args) -> ndarray``
     - ``(x, *args) -> ndarray`` shape ``(batch, out, in)``
   * - :class:`~adeptml.models.Physics_VJP`
     - ``(x, *args) -> ndarray``
     - ``(x, grad_out, *args) -> ndarray`` shape ``(batch, in)``
   * - :class:`~adeptml.models.Physics_SplitVJP`
     - ``(x, *args) -> (ndarray, pullback_fn)``
     - not used

**When to use each:**

- Use :class:`~adeptml.models.Physics` when the full Jacobian is cheap or
  already available (e.g. small output dimension, analytical Jacobian).
- Use :class:`~adeptml.models.Physics_VJP` when a manual VJP avoids
  materialising the full Jacobian (e.g. adjoint method, large output dim).
- Use :class:`~adeptml.models.Physics_SplitVJP` when the physics solver
  already produces a pullback closure during the forward pass (e.g. via
  ``jax.vjp``).  This is the most memory-efficient option because no part of
  the physics forward is re-run during backpropagation.

  Example with a JAX solver::

      from Physics_Funcs import pp_model_gen_forward
      # pp_model_gen_forward(mesh_params, bc, ic, sim_time)
      #   -> (T_normalised, pullback_fn)

      phy_cfg = PhysicsConfig(
          forward_func=pp_model_gen_forward,
          use_split_vjp=True,
      )

.. automodule:: adeptml.models
  :members:

=======
Configs
=======
.. automodule:: adeptml.configs
  :members:

==============
Training Utils
==============
.. automodule:: adeptml.train_utils
  :members:
