### Wake analysis
This is a collection of scripts for computation and analysis of wake quantities. It is aimed at data produced by the TAU flow solver, however anyone can plug different reading and writing routines in order to carry out the same analysis. Presently, the input is assumed to be 2D planar and unstructured. Adapt as necessary if you use different inputs.

## Installation
Add the top level directory (i.e. TAU\_processing) to PYTHONPATH

## IO
the pyTecIO folder contains tecreader.py, which is a module for bindary Tecplot I/O. It is a link to the tec\_series repo, i.e. linked via git submodule add https://github.com/awaldm/tec\_series pyTecIO
