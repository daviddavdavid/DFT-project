
# Import statements
import DFT_code
import os
import matplotlib.pyplot as plt
import numpy as np
from gpaw import GPAW, PW
from ase.optimize import BFGS
from ase.lattice.hexagonal import Graphene
from ase.visualize import view
from ase.io import read
from gpaw.spinorbit import soc_eigenstates
from ase.build import add_adsorbate
from ase.io import Trajectory
from scipy.optimize import curve_fit


CC_dist = 1.424
a = CC_dist * np.sqrt(3)
c =10
graphene = graphene = Graphene("C", size=(1, 1, 1), latticeconstant={"a": a, "c": c})
k, ecut, a, c, E0 = DFT_code.converge(graphene, "graphene")
print(k, ecut, a, c, E0)
band_gap, x = DFT_code.run_structure(a, k, c, ecut, "PBE", (1, 1), "graphene")
print(f"The band gap is {band_gap}")