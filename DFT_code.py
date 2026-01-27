# Import statements
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
from ase import Atoms, cell

def run_structure( a, k, vacuum, E_cut, xc, size_xy, structure_name, filepath = None):
    # first it selects the structure you want to model, currently we have 3 options
    structure = None
    if structure_name == "graphene": 
        structure = Graphene(symbol="C", latticeconstant={"a": a, "c": vacuum}, size=(size_xy[0], size_xy[1], 1))
    elif structure_name == "Si":
        a = 4.05
        # c = 10.99
        c = vacuum

        # variable to ensure z positions are dependent on a, instead of c
        z_dist_a = 0.16988097777 * a
        z_dist = z_dist_a / c 

        structure = Atoms(
            'Si2',
            cell= [a, a, c, 90, 90, 120],
            scaled_positions= [(0, 0, 0), (1/3, 2/3, z_dist)],
            pbc=[True, True, False]

        )
    elif structure_name == "h-BN":
        # Now has a two atom basis
        structure = Graphene(symbols=('B', 'N'), latticeconstant={'a': a, 'c': vacuum}, size=(size_xy[0], size_xy[1], 1))
    else:
        raise ValueError("Unsupported structure name and unable to read from file.")
    
    # now it either gets the calculator from a saved file or it generates a new one
    structure_calc = GPAW(mode=PW(E_cut), 
                            kpts=(k, k, 1), 
                            xc=xc, 
                            spinpol=True, 
                            txt=f"out/{structure_name}_xc{xc}_k_{k}_Ecut{E_cut}_vac{vacuum}.txt")
    structure.calc = structure_calc

    # gets the potential energy to make sure the other calculations work
    potential_energy = structure.get_potential_energy()
    fermi_level = structure.calc.get_fermi_level()
    soc = soc_eigenstates(structure.calc)
    eigenvalues = soc.eigenvalues().ravel() # converts the 2D array to 1D
    valence_energy = np.max(eigenvalues[eigenvalues < fermi_level]) # selects the highest energy below the fermi level
    conduction_energy = np.min(eigenvalues[eigenvalues > fermi_level]) # selects the lowest energy above the fermi level
    band_gap = conduction_energy - valence_energy
    return band_gap, potential_energy

def generate_file_path(structure_name, xc, k, E_cut, vacuum):
    return f"out/{structure_name}_xc{xc}_k_{k}_Ecut{E_cut}_vac{vacuum}"

filepath = generate_file_path("Si", "PBE", 30, 200, 20)
band_gap, pot_eng = run_structure( a=4.06, k = 30, E_cut = 200, vacuum = 20, xc = "PBE", size_xy = (1,1), structure_name = "Si", filepath = filepath)
print("Band gap:", band_gap)

# Band gap: 0.9970890608619065