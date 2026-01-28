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


def run_structure( a, k, vacuum, E_cut, xc, size_xy, structure_name):
    # first it selects the structure you want to model, currently we have 3 options
    structure = None
    if structure_name == "graphene": 
        structure = Graphene(symbol="C", latticeconstant={"a": a, "c": vacuum}, size=(size_xy[0], size_xy[1], 1))
    elif structure_name == "Si":
        # Works because silicene has the same lattice structure as graphene
        structure =  Graphene(symbol="Si", latticeconstant={"a": a, "c": vacuum}, size=(size_xy[0], size_xy[1], 1))
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
                            txt=generate_file_path(structure_name, xc, k, E_cut, vacuum) + ".txt")
    structure.calc = structure_calc

    # gets the potential energy to make sure the other calculations work
    potential_energy = structure.get_potential_energy()
    fermi_level = structure.calc.get_fermi_level()
    dos = structure_calc.dos()
    structure_energies = dos.get_energies()
    structure_l0_a0 = dos.raw_pdos(structure_energies, a=0, l=0)
    structure_l0_a1 = dos.raw_pdos(structure_energies, a=1, l=0)
    structure_l0 = structure_l0_a0 + structure_l0_a1
    structure_l1_a0 = dos.raw_pdos(structure_energies, a=0, l=1)
    structure_l1_a1 = dos.raw_pdos(structure_energies, a=1, l=1)
    structure_l1 = structure_l1_a0 + structure_l1_a1
    structure_total = structure_l1 + structure_l0

    valence_energy = None
    conduction_energy = None

    for energy in structure_energies:
        if energy < fermi_level and (valence_energy is None or energy > valence_energy):
            valence_energy = energy
        elif energy > fermi_level and (conduction_energy is None or energy < conduction_energy):
            conduction_energy = energy

    band_gap = conduction_energy - valence_energy

    plt.figure(figsize=(8, 6))
    plt.plot(structure_energies - fermi_level, structure_total, label='Total DOS', color='black', marker='o')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Density of States (states/eV)')
    plt.title(f'Density of States for {structure_name} (xc={xc}, k={k}, E_cut={E_cut} eV, vacuum={vacuum} Å)')
    plt.axvline(x=0, color='red', linestyle='--', label='Fermi Level')
    plt.legend()
    plt.grid()
    plt.savefig(f'out/{structure_name}_DOS_xc{xc}_k{k}_Ecut{E_cut}_vac{vacuum}.png')
    plt.show()

    
    return band_gap, potential_energy

def generate_file_path(structure_name, xc, k, E_cut, vacuum):
    return f"out/{structure_name}_xc{xc}_k_{k}_Ecut{E_cut}_vac{vacuum}"
band_gap, pot_eng = run_structure( a=1.424 * np.sqrt(3), k = 30, E_cut = 500, vacuum = 10.0, xc = "PBE", size_xy = (1,1), structure_name = "graphene")
print("Band gap:", band_gap)