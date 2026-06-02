Compute with physical units
===========================

``brainevent`` is fully compatible with `BrainUnit <https://github.com/chaobrain/brainunit>`_,
so synaptic weights and currents can carry physical units (e.g. millisiemens, millivolts)
and be dimensionally checked throughout a simulation. This guide shows how units flow through
event-driven operations.


Attach units to weights
------------------------

Give a connectivity weight a unit and the result of the matrix product inherits it:

.. code-block:: python

   import brainevent
   import brainunit as u
   import jax.numpy as jnp

   spikes = brainevent.BinaryArray(jnp.array([1, 0, 1, 0, 1], dtype=jnp.float32))

   conn = brainevent.JITCScalarR(
       num_pre=5, num_post=3, prob=0.5,
       weight=1.62 * u.mS,        # weight carries a unit
       seed=0,
   )

   current = spikes @ conn        # current is a unit-aware quantity (mS-scaled)


Why this matters
----------------

- **Dimensional safety** — multiplying a conductance by a voltage and adding it to a current
  raises an error if the dimensions do not line up, catching modeling bugs early.
- **Readable models** — parameters read as ``20. * u.ms`` or ``-50. * u.mV`` instead of bare
  floats whose units you must remember.
- **No performance cost** — units are tracked at trace time and erased before the kernel
  runs, so event-driven kernels stay just as fast.


Mixing units and plain arrays
------------------------------

Unit-aware quantities and plain JAX arrays interoperate: a ``BinaryArray`` of spikes is
dimensionless, and it picks up the weight's unit through the product. Everything continues to
work inside ``jax.jit`` and ``jax.grad``.

.. seealso::

   The `BrainUnit documentation <https://brainx.chaobrain.com/brainunit/>`_ covers the full
   unit system. Most ``brainevent`` constructors that accept a ``weight`` accept a unit-aware
   weight in its place.
