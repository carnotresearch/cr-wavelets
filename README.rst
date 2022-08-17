Wavelets with JAX
==================================================================


|pypi| |license| |zenodo| |docs| |unit_tests| |coverage|


Introduction
-------------------


CR-Wavelets is a Python library that enables efficiently solving
a wide variety of sparse representation based signal processing problems.
It is a cohesive collection of sub-libraries working together. Individual
sub-libraries provide functionalities for:
wavelets, linear operators, greedy and convex optimization 
based sparse recovery algorithms, subspace clustering, 
standard signal processing transforms,
and linear algebra subroutines for solving sparse linear systems. 
It has been built using `Google JAX <https://jax.readthedocs.io/en/latest/>`_, 
which enables the same high level
Python code to get efficiently compiled on CPU, GPU and TPU architectures
using `XLA <https://www.tensorflow.org/xla>`_. 

For detailed documentation and usage, please visit `online docs <https://cr-wavelets.readthedocs.io/en/latest>`_.

For theoretical background, please check online notes at `Topics in Signal Processing <https://tisp.indigits.com>`_
and references therein (still under development).

Supported Platforms
----------------------

``CR-Wavelets`` can run on any platform supported by ``JAX``. 
We have tested ``CR-Wavelets`` on Mac and Linux platforms and Google Colaboratory.

* The latest code in the library has been tested against JAX 0.3.14.

``JAX`` is not officially supported on Windows platforms at the moment. 
Although, it is possible to build it from source using Windows Subsystems for Linux.
Alternatively, you can check out the community supported Windows build for JAX
available from https://github.com/cloudhan/jax-windows-builder.
This seems to work well and all the unit tests in the library have passed
on Windows also. 

Installation
-------------------------------

Installation from PyPI:

.. code:: shell

    python -m pip install cr-wavelets

Directly from our GITHUB repository:

.. code:: shell

    python -m pip install git+https://github.com/carnotresearch/cr-wavelets.git



Examples/Usage
----------------

See the `examples gallery <https://cr-wavelets.readthedocs.io/en/latest/gallery/index.html>`_ in the documentation.


Contribution Guidelines/Code of Conduct
----------------------------------------

* `Contribution Guidelines <CONTRIBUTING.md>`_
* `Code of Conduct <CODE_OF_CONDUCT.md>`_


`Documentation <https://carnotresearch.github.io/cr-wavelets>`_ | 
`Code <https://github.com/carnotresearch/cr-wavelets>`_ | 
`Issues <https://github.com/carnotresearch/cr-wavelets/issues>`_ | 
`Discussions <https://github.com/carnotresearch/cr-wavelets/discussions>`_ |


.. |docs| image:: https://readthedocs.org/projects/cr-wavelets/badge/?version=latest
    :target: https://cr-wavelets.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    :scale: 100%

.. |unit_tests| image:: https://github.com/carnotresearch/cr-wavelets/actions/workflows/ci.yml/badge.svg
    :alt: Unit Tests
    :scale: 100%
    :target: https://github.com/carnotresearch/cr-wavelets/actions/workflows/ci.yml


.. |pypi| image:: https://badge.fury.io/py/cr-wavelets.svg
    :alt: PyPI cr-wavelets
    :scale: 100%
    :target: https://badge.fury.io/py/cr-wavelets

.. |coverage| image:: https://codecov.io/gh/carnotresearch/cr-wavelets/branch/master/graph/badge.svg?token=JZQW6QU3S4
    :alt: Coverage
    :scale: 100%
    :target: https://codecov.io/gh/carnotresearch/cr-wavelets


.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :alt: License
    :scale: 100%
    :target: https://opensource.org/licenses/Apache-2.0

.. |codacy| image:: https://app.codacy.com/project/badge/Grade/36905009377e4a968124dabb6cd24aae
    :alt: Codacy Badge
    :scale: 100%
    :target: https://www.codacy.com/gh/carnotresearch/cr-wavelets/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=carnotresearch/cr-wavelets&amp;utm_campaign=Badge_Grade

.. |zenodo| image:: https://zenodo.org/badge/525693334.svg
    :alt: DOI
    :scale: 100%
    :target: https://zenodo.org/badge/latestdoi/525693334
