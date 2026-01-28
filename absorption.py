from gpaw.response.df import DielectricFunction
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib import rc
from gpaw.spinorbit import soc_eigenstates
from ase.io import read
from ase.visualize import view
from gpaw import GPAW, PW
from ase.lattice.hexagonal import Graphene
import os

# Part 1: Ground state calculation
a = 2.514
c_conv = 8
ecut_conv = 800
k_conv = 10
k = k_conv
ecut = ecut_conv
xc = 'PBE'
c = c_conv
POSCAR_file = "BN.poscar"
atoms = read(filename=POSCAR_file)
hBN = atoms[1,2]
calc = GPAW(mode=PW(ecut),
                           xc=xc,
                           kpts=(k, k, 1),
                           txt=f'out/h_BN_xc{xc}_ecut{ecut}_k{k}.txt',
                           spinpol=True)

atoms.calc = calc
atoms.get_potential_energy()  # ground state calculation is performed
calc.write('hBN.gpw', 'all')  # use 'all' option to write wavefunction

# Part 2 : Spectrum calculation
# DF: dielectric function object
# Ground state gpw file (with wavefunction) as input
df = DielectricFunction(
    calc='hBN.gpw',
    frequencies={'type': 'nonlinear',
                 'domega0': 0.05,
                 'omegamax': 100},  # using nonlinear frequency grid
    rate='eta')
# By default, a file called 'df.csv' is generated
df.get_dielectric_function()

rc('figure', figsize=(4.0, 4.0), dpi=800)

data = np.loadtxt('df.csv', delimiter=',')

# Set plotting range
xmin = 0.1
xmax = 100.0
inds_w = (data[:, 0] >= xmin) & (data[:, 0] <= xmax)

plt.plot(data[inds_w, 0], 4 * pi * data[inds_w, 4])
plt.xlabel('$\\omega / [eV]$')
plt.ylabel('$\\mathrm{Im} \\epsilon(\\omega)$')
plt.axvline(92.0, linestyle='--', color='red', linewidth=0.5)
plt.savefig('si_abs.png', bbox_inches='tight')
print(f"Max omega in data: {data[:,0].max():.2f} eV")