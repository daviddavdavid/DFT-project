
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
from gpaw.response.df import DielectricFunction
from matplotlib import rc


CC_dist = 1.424
a = CC_dist * np.sqrt(3)
c =10
graphene = graphene = Graphene("C", size=(1, 1, 1), latticeconstant={"a": a, "c": c})
# k, ecut, a, c, E0 = DFT_code.converge(graphene, "graphene")
k = 5
ecut = 500
a = 2.481
c = 9
xc = "PBE"
band_gap, _ = DFT_code.run_structure(a, k, c, ecut, xc, (1, 1), "graphene")
print(f"The band gap is {band_gap}")



# Part 2 : Spectrum calculation
# DF: dielectric function object
# Ground state gpw file (with wavefunction) as input
df = DielectricFunction(
    calc="graphene.gpw",
    frequencies={"type": "nonlinear",
                 "domega0": 0.05,
                 "omegamax": 100},  # using nonlinear frequency grid
    rate="eta")
# By default, a file called "df.csv" is generated
df.get_dielectric_function()

rc("figure", figsize=(4.0, 4.0), dpi=800)

data = np.loadtxt("df.csv", delimiter=",")

# Set plotting range
xmin = 0.1
xmax = 100.0
inds_w = (data[:, 0] >= xmin) & (data[:, 0] <= xmax)

plt.plot(data[inds_w, 0], 4 * np.pi * data[inds_w, 4])
plt.xlabel("$\\omega / [eV]$")
plt.ylabel("$\\mathrm{Im} \\epsilon(\\omega)$")
plt.axvline(92.0, linestyle="--", color="red", linewidth=0.5)
plt.savefig("graphene_abs.png", bbox_inches="tight")
print(f"Max omega in data: {data[:,0].max():.2f} eV")