Introduction
=====================

.. contents::
    :depth: 2
    :local:


This library aims to provide XLA/JAX based Python implementations for
discrete and continuous wavelets.

Bulk of this library is built using functional programming techniques
which is critical for the generation of efficient numerical codes for CPU
and GPU architectures.

Open Source Credits
-----------------------------

Major parts of this library are directly influenced by existing projects.
While the implementation in CR-Sparse is fresh (based on JAX), it has been
possible thanks to the extensive study of existing implementations. We list
here some of the major existing projects which have influenced the implementation
in CR-Sparse. Let us know if we missed anything. 

* `JAX <https://github.com/google/jax>`_ The overall project structure is heavily
  influenced by the conventions followed in JAX. We learned the functional programming
  techniques as applicable for linear algebra work by reading the source code of JAX.
* `SciPy <https://github.com/scipy/scipy>`_ JAX doesn't have all parts of SciPy ported
  yet. Some parts of SciPy have been adapted and re-written (in functional manner) 
  as per the needs of CR-Sparse. E.g. ``cr.sparse.dsp.signals``. The :cite:`torrence1998practical` version
  of CWT in ``cr.sparse.wt``.
* `PyWavelets <https://github.com/PyWavelets/pywt>`_: The DWT and CWT implementations
  in ``cr.sparse.wt`` are largely derived from it. The filter coefficients for discrete
  wavelets have been ported from C to Python from here.
* `WaveLab <https://github.com/gregfreeman/wavelab850>`_ This MATLAB package helped a lot in
  initial understanding of DWT implementation.
* `aaren/wavelets <https://github.com/aaren/wavelets>`_ is a decent CWT implementation following
  :cite:`torrence1998practical`. Influenced: ``cr.sparse.wt``.
  

Further Reading
------------------
* `Functional programming <https://en.wikipedia.org/wiki/Functional_programming>`_
* `How to Think in JAX <https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html>`_
* `JAX - The Sharp Bits <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_


.. bibliography::
   :filter: docname in docnames

`Documentation <https://carnotresearch.github.io/cr-wavelets>`_ | 
`Code <https://github.com/carnotresearch/cr-wavelets>`_ | 
`Issues <https://github.com/carnotresearch/cr-wavelets/issues>`_ | 
`Discussions <https://github.com/carnotresearch/cr-wavelets/discussions>`_ |
`Sparse-Plex <https://sparse-plex.readthedocs.io>`_ 
