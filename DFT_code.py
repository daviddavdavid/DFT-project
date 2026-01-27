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


def parabola(x, a, b, c):
    """parabola polynomial function"""
    return a + b * x + c * x ** 2


def fit_E_vs_V(traj_filename: str) -> float:
    """
    Fits energy vs. volume data from an ASE trajectory to a parabola (E = aV² + bV + c)
    and returns the equilibrium volume (V₀) corresponding to the energy minimum.
    """
    # Load trajectory file
    traj_path = os.path.join('out', traj_filename)
    traj = Trajectory(traj_path)

    volumes = []
    energies = []

    for atoms in traj:
        volumes.append(atoms.get_volume())
        energies.append(atoms.get_potential_energy())

    volumes = np.array(volumes)
    energies = np.array(energies)

    # Fit data to parabola
    popt, _ = curve_fit(parabola, volumes, energies, p0=[energies.min(), 1, 1])

    # Extract polynomial coefficients
    a, b, c = popt

    # Get optimized volume
    V0 = - b / (2 * c)

    plt.figure(figsize=(8, 6))

    # Plotting
    v_fit = np.linspace(min(volumes), max(volumes), 50)
    e_fit = parabola(v_fit, a, b, c)
    plt.plot(v_fit, e_fit, '-', label='ADD APPROPRIATE LABEL')

    # Print fitted parameters
    print(f"parabola fit parameters:")
    print(f"a = {a:.3f}, b = {b:.3f}, c = {c:.3f}, V0 = {V0:.3f} Å³")

    plt.plot(volumes, energies, 'o', label='Data', color='black')
    plt.xlabel('Volume (Å³)')
    plt.ylabel('Energy (eV)')
    plt.title('E vs V parabolic fit')
    plt.legend()
    plt.show()

    E0 = e_fit.min()

    return V0, E0


def converge(atoms, atom_name, CCdist=1.41, c=10, Ediff=0.08, xc='PBE', ecut=300, E_range = [100, 600, 50], c_list = [1,2,3,4,5,6,7,8,9,10]):
    # The lattice constant of graphene is equal to the bond length times sqrt(3)
    a = CCdist * np.sqrt(3)

    # To lazy to name it differently
    graphene = atoms

    energies = []
    kpoints=[]

    # Run calculations at different k-point values
    for k in range(2, 12, 1):
        graphene.calc = GPAW(mode=PW(ecut),
                                xc=xc,
                                kpts=(k, k, 1),
                                txt=f'out/{atom_name}_xc{xc}_ecut{ecut}_k{k}.txt',
                                spinpol=True)
        energy = graphene.get_potential_energy()
        energies.append(energy)
        kpoints.append(k)

    delta_energies = []
    for i in range(1,len(energies)):
        delta_energies.append(np.absolute(energies[i]-energies[i-1]))

    kpoints_edit = kpoints
    kpoints_edit.pop(0)

    for i in range(len(delta_energies)):
        if delta_energies[i] <= Ediff:
            converged_kvalue = kpoints[i]
            break

    print(converged_kvalue)

    k=converged_kvalue

    # ecut convergence
    energies = []
    encuts=[]

    E_start = E_range[0]
    E_end = E_range[1]
    E_step = E_range[2]

    for ecut in range(E_start, E_end, E_step):
        graphene.calc = GPAW(mode=PW(ecut),
                                xc=xc,
                                kpts=(k, k, 1),
                                txt=f'out/{atom_name}_xc{xc}_ecut{ecut}_k{k}.txt',
                                spinpol=True)
        energy = graphene.get_potential_energy()
        energies.append(energy)
        encuts.append(ecut)

    delta_energies = []
    for i in range(1,len(encuts)):
        delta_energies.append(np.absolute(energies[i]-energies[i-1]))

    encuts_edit = encuts
    encuts_edit.pop(0)

    for i in range(len(delta_energies)):
        if delta_energies[i] <= Ediff:
            converged_Evalue = encuts_edit[i]
            break

    print(converged_Evalue)

    ecut = converged_Evalue

    for c in c_list:
        # Set up the initial guess for structure and lattice constants for a 1x1x1 supercell of graphene
        graphene = Graphene('C', size=(1, 1, 1), latticeconstant={'a': a, 'c': c})

        graphene.calc = GPAW(mode=PW(ecut),
                            xc=xc,
                            kpts=(k, k, 1),
                            txt=f'out/{atom_name}_xc{xc}_ecut{ecut}_k{k}.txt',
                            spinpol=True)
        energy = graphene.get_potential_energy()
        energies.append(energy)

    delta_energies = []
    for i in range(1,len(c_list)):
        delta_energies.append(np.absolute(energies[i]-energies[i-1]))

    c_edit = c_list
    c_edit.pop(0)

    for i in range(len(delta_energies)):
        if delta_energies[i] <= Ediff:
            converged_cvalue = c_edit[i]
            break

    print(converged_cvalue)

    c = converged_cvalue

    energies = []

    atoms = Graphene('C', size=(1, 1, 1), latticeconstant={'a': a, 'c': c})

    # Now I will perform the volume fit
    traj = Trajectory(os.path.join('out', f'{atom_name}.traj'), 'w')

    energies = []
    scales = np.arange(0.98, 1.02, 0.01)

    for s in scales:
        atoms_scaled = atoms.copy()
        atoms_scaled.set_cell(atoms.cell * s, scale_atoms=True)
        atoms_scaled.calc = GPAW(mode=PW(ecut),
                            xc=xc,
                            kpts=(k, k, 1),
                            txt=f'out/{atom_name}_xc{xc}_ecut{ecut}_k{k}.txt',
                            spinpol=True)
        energies.append(atoms_scaled.get_potential_energy())

        traj.write(atoms_scaled)

    V0, E0 = fit_E_vs_V(f'{atom_name}.traj') if os.path.exists(os.path.join('out', f'{atom_name}.traj')) else None

    A0 = V0 / c
    a = np.sqrt(2 * A0 / np.sqrt(3))
    print(a)
    CC = a / np.sqrt(3)
    print(CC)
        
    return k, ecut, a, c, E0


def run_structure( a, k, vacuum, E_cut, xc, size_xy, structure_name, filepath = None):
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
                            txt=f"out/{structure_name}_xc{xc}_k_{k}_Ecut{E_cut}_vac{vacuum}.txt")
    structure.calc = structure_calc

    # gets the potential energy to make sure the other calculations work
    potential_energy = structure.get_potential_energy()
    fermi_level = structure.calc.get_fermi_level()
    soc = soc_eigenstates(structure.calc)
    eigenvalues = soc.eigenvalues().ravel() # converts the 2D array to 1D
    valence_energy = None
    conduction_energy = None
    for k_point_energy_array in eigenvalues:
        for energy in k_point_energy_array:
            if energy < fermi_level and (valence_energy is None or energy > valence_energy):
                valence_energy = energy
            elif energy > fermi_level and (conduction_energy is None or energy < conduction_energy):
                conduction_energy = energy
    band_gap = conduction_energy - valence_energy
    return band_gap, potential_energy

def generate_file_path(structure_name, xc, k, E_cut, vacuum):
    return f"out/{structure_name}_xc{xc}_k_{k}_Ecut{E_cut}_vac{vacuum}"
filepath = generate_file_path("graphene", "PBE", 3, 500, 10.0)
band_gap, pot_eng = run_structure( a=2.5, k = 3, E_cut = 500, vacuum = 10.0, xc = "PBE", size_xy = (1,1), structure_name = "graphene", filepath = filepath)
print("Band gap:", band_gap)