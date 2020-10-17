###
These directories contain scripts for accessing basic functionalities, mostly aimed at computing statistics or other quantities from planar Tecplot-formatted flow solution slices. The inputs are time series named following the convention:
case\_i=XXX\_t=YYY.plt

With XXX the iteration number and YYY the physical time. The datasets are read using the tecreader module (https://github.com/awaldm/tec\_series), which requires a working installation of Tecplot and the pytecplot module. The selection of computation cases is controlled via a WakeCaseParams object described in wake\_config.py. Check that file for  
