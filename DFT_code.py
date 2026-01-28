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

# Note that l_max is up to l_max so not including
def run_structure( a, k, vacuum, E_cut, xc, size_xy, l_max, structure_name):
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
    structure_calc.write(f"{structure_name}.gpw", "all")
    fermi_level = structure.calc.get_fermi_level()
    dos = structure_calc.dos()
    structure_energies = dos.get_energies()

    structure_total = 0
    for l in range(l_max):
        structure_a0 = dos.raw_pdos(structure_energies, a=0, l=l)
        structure_a1 = dos.raw_pdos(structure_energies, a=1, l=l)
        structure_current_l = structure_a0 + structure_a1
        structure_total += structure_current_l

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
if __name__ == "__name__":
    band_gap, pot_eng = run_structure( a=1.424 * np.sqrt(3), k = 30, E_cut = 500, vacuum = 10.0, xc = "PBE", size_xy = (1,1), structure_name = "graphene")
    print("Band gap:", band_gap)