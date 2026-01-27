import numpy as np
import os
from ase.io import Trajectory
from gpaw import GPAW, PW
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from ase import Atoms

os.makedirs("figs", exist_ok=True)

def parabola(x, a, b, c):
    """parabola polynomial function"""
    return a + b * x + c * x ** 2

def fit_E_vs_V(traj_filename: str) -> float:
    """
    Fits energy vs. volume data from an ASE trajectory to a parabola (E = aV² + bV + c)
    and returns the equilibrium volume (V₀) corresponding to the energy minimum.
    """
    # Load trajectory file
    traj_path = traj_filename
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

    v_fit = np.linspace(min(volumes), max(volumes), 50)
    e_fit = parabola(v_fit, a, b, c)

    plt.figure()
    plt.plot(volumes, energies, 'o', label="DFT data")
    plt.plot(v_fit, e_fit, '-', label="Parabolic fit")
    plt.xlabel("Volume (Å³)")
    plt.ylabel("Total energy (eV)")
    plt.title("Energy vs Volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/E_vs_V.png")
    plt.show()

    # Print fitted parameters
    print(f"parabola fit parameters:")
    print(f"a = {a:.3f}, b = {b:.3f}, c = {c:.3f}, V0 = {V0:.3f} Å³")

    E0 = e_fit.min()

    return V0, E0

def find_converged_value(values, energies, Ediff):
    delta_energies = []
    for i in range(1, len(energies)):
        delta_energies.append(abs(energies[i] - energies[i - 1]))

    values_edit = values.copy()
    values_edit.pop(0)

    for i in range(len(delta_energies)):
        if delta_energies[i] <= Ediff:
            return values_edit[i]

    return values_edit[-1]


def kpoint_convergence(atoms, atom_name, ecut, xc, Ediff):
    energies = []
    kpoints = []

    for k in range(2, 12, 1):
        atoms_copy = atoms.copy()
        atoms_copy.calc = GPAW(
            mode=PW(ecut),
            xc=xc,
            kpts=(k, k, 1),
            txt=f'out/{atom_name}_xc{xc}_ecut{ecut}_k{k}.txt',
            spinpol=True
        )
        energies.append(atoms_copy.get_potential_energy())
        kpoints.append(k)

    plt.figure()
    plt.plot(kpoints, energies, marker='o')
    plt.xlabel("k-point mesh")
    plt.ylabel("Total energy (eV)")
    plt.title(f"{atom_name} k-point convergence")
    plt.tight_layout()
    plt.savefig(f"figs/{atom_name}_kpoint_convergence.png")
    plt.show()

    k_conv = find_converged_value(kpoints, energies, Ediff)
    print(k_conv)
    return k_conv


def ecut_convergence(atoms, atom_name, k, xc, Ediff, E_range):
    energies = []
    encuts = []

    E_start, E_end, E_step = E_range

    for ecut in range(E_start, E_end, E_step):
        atoms_copy = atoms.copy()
        atoms_copy.calc = GPAW(
            mode=PW(ecut),
            xc=xc,
            kpts=(k, k, 1),
            txt=f'out/{atom_name}_xc{xc}_ecut{ecut}_k{k}.txt',
            spinpol=True
        )
        energies.append(atoms_copy.get_potential_energy())
        encuts.append(ecut)

    plt.figure()
    plt.plot(encuts, energies, marker='o')
    plt.xlabel("Plane-wave cutoff (eV)")
    plt.ylabel("Total energy (eV)")
    plt.title(f"{atom_name} ecut convergence")
    plt.tight_layout()
    plt.savefig(f"figs/{atom_name}_ecut_convergence.png")
    plt.show()

    ecut_conv = find_converged_value(encuts, energies, Ediff)
    print(ecut_conv)
    return ecut_conv

def vacuum_convergence(atoms, atom_name, k, ecut, xc, Ediff, c_list):
    energies = []

    for c in c_list:
        atoms_copy = atoms.copy()
        cell = atoms_copy.cell.copy()
        cell[2, 2] = c
        atoms_copy.set_cell(cell, scale_atoms=False)

        atoms_copy.calc = GPAW(
            mode=PW(ecut),
            xc=xc,
            kpts=(k, k, 1),
            txt=f'out/{atom_name}_xc{xc}_ecut{ecut}_k{k}.txt',
            spinpol=True
        )
        energies.append(atoms_copy.get_potential_energy())

    plt.figure()
    plt.plot(c_list, energies, marker='o')
    plt.xlabel("Vacuum size (Å)")
    plt.ylabel("Total energy (eV)")
    plt.title(f"{atom_name} vacuum convergence")
    plt.tight_layout()
    plt.savefig(f"figs/{atom_name}_vacuum_convergence.png")
    plt.show()

    c_conv = find_converged_value(c_list, energies, Ediff)
    print(c_conv)
    return c_conv

def volume_fit(atoms, atom_name, k, ecut, xc):
    traj_path = os.path.join('out', f'{atom_name}.traj')
    traj = Trajectory(traj_path, 'w')

    energies = []
    scales = np.arange(0.98, 1.02, 0.01)

    for s in scales:
        atoms_scaled = atoms.copy()
        atoms_scaled.set_cell(atoms.cell * s, scale_atoms=True)

        atoms_scaled.calc = GPAW(
            mode=PW(ecut),
            xc=xc,
            kpts=(k, k, 1),
            txt=f'out/{atom_name}_xc{xc}_ecut{ecut}_k{k}.txt',
            spinpol=True
        )

        energies.append(atoms_scaled.get_potential_energy())
        traj.write(atoms_scaled)

    if os.path.exists(traj_path):
        V0, E0 = fit_E_vs_V(traj_path)
        return V0, E0

    return None, None

def converge(atoms,
             atom_name,
             Ediff=0.08,
             xc='PBE',
             ecut=300,
             E_range=[100, 600, 50],
             c_list=[1,2,3,4,5,6,7,8,9,10]):

    atoms_base = atoms.copy()

    # k-point convergence
    k = kpoint_convergence(atoms_base, atom_name, ecut, xc, Ediff)

    # ecut convergence
    ecut = ecut_convergence(atoms_base, atom_name, k, xc, Ediff, E_range)

    # vacuum convergence
    c = vacuum_convergence(atoms_base, atom_name, k, ecut, xc, Ediff, c_list)

    # volume fit
    V0, E0 = volume_fit(atoms_base, atom_name, k, ecut, xc)

    A0 = V0 / c
    a = np.sqrt(2 * A0 / np.sqrt(3))
    # CC = a / np.sqrt(3)
    print(a)
    # print(CC)

    return k, ecut, a, c, E0

a = 4.05
c = 10.99
z_dist_a = 0.16988097777 * a
z_dist = z_dist_a / c

silicene = Atoms(
    'Si2',
    cell=[a, a, c, 90, 90, 120],
    scaled_positions=[(0, 0, 0), (1/3, 2/3, z_dist)],
    pbc=[True, True, False]
)

k, ecut, a_opt, c_opt, E0 = converge(
    silicene,
    atom_name="Si",
    Ediff=0.08,
    xc="PBE"
)

# Results: k = 5, ecut = 200, c = 8, a = 4.062544476939418

# a = -7.095, b = -0.043, c = 0.000, V0 = 114.345 Å³
# 2.3455111473558 CC?