========
flowpost
========


.. image:: https://img.shields.io/pypi/v/flowpost.svg
        :target: https://pypi.python.org/pypi/flowpost

.. image:: https://img.shields.io/travis/awaldm/flowpost.svg
        :target: https://travis-ci.com/awaldm/flowpost

.. image:: https://readthedocs.org/projects/flowpost/badge/?version=latest
        :target: https://flowpost.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/awaldm/flowpost/shield.svg
     :target: https://pyup.io/repos/github/awaldm/flowpost/
     :alt: Updates



Unsteady Flow Field Analysis for CFD Data


* Free software: MIT license
* Documentation: https://flowpost.readthedocs.io.

Wake analysis
-------------
This is a collection of scripts for computation and analysis of wake quantities. It is aimed at data produced by the TAU flow solver, however anyone can plug different reading and   writing routines in order to carry out the same analysis. Presently, the input is assumed to be 2D planar and unstructured. Adapt as necessary if you use different inputs.

Installation
------------
Add the top level directory (i.e. TAU\_processing) to PYTHONPATH

IO
--
the pyTecIO folder contains tecreader.py, which is a module for bindary Tecplot I/O. It is a link to the tec\_series repo, i.e. linked via git submodule add https://github.com/      awaldm/tec\_series pyTecIO



Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
