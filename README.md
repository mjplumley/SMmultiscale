# SMmultiscale

This code runs the multiscale HMM stratgey for solving the single mode equations as written in:

Plumley, Meredith, et al. "Self-consistent single mode investigations of the quasi-geostrophic convection-driven dynamo model." Journal of Plasma Physics 84.4 (2018).

and derived from:

Calkins, Michael A., et al. "A multiscale dynamo model driven by quasi-geostrophic convection." Journal of Fluid Mechanics 780 (2015): 143-166.


The main time stepping loop and control of the parameters is located in SMdynamo_multiscale.py and all the background matrix creation and transfers are contained in SMdynamo_tools.py.  The control parameters are explained in the file and correspond with the control parameters in paper above.
