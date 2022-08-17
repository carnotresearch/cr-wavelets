Quick Start
===================

|pypi| |license| |zenodo| |docs| |unit_tests| |coverage| 


Platform Support
----------------------

``cr-wavelets`` can run on any platform supported by ``JAX``. 
We have tested ``cr-wavelets`` on Mac and Linux platforms and Google Colaboratory.

``JAX`` is not officially supported on Windows platforms at the moment. 
Although, it is possible to build it from source using Windows Subsystems for Linux.


Installation
-------------------------------

Installation from PyPI:

.. code:: shell

    python -m pip install cr-wavelets



Directly from our GITHUB repository:

.. code:: shell

    python -m pip install git+https://github.com/carnotresearch/cr-wavelets.git



Examples
----------------

* See the :ref:`examples gallery <gallery>`.

.. note::

    ``cr-wavelets`` depends on its sister library `cr-nimble <https://github.com/carnotresearch/cr-nimble>`_.
    Normally, it would be installed automatically as a dependency. 
    You may want to install it directly from GITHUB if you need access to the latest code.

    .. code:: shell

        python -m pip install git+https://github.com/carnotresearch/cr-nimble.git


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
