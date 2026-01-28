import numpy as np
import os
import matplotlib.pyplot as plt

from ase import Atoms
from ase.optimize import BFGS
from ase.io import Trajectory
from gpaw import GPAW, PW

os.makedirs("figs", exist_ok=True)
os.makedirs("out", exist_ok=True)

def find_converged_value(values, energies, Ediff):
    for i in range(1, len(energies)):
        if abs(energies[i] - energies[i - 1]) <= Ediff:
            return values[i]
    return values[-1]

def vacuum_convergence(atoms, atom_name, k, ecut, xc, c_list, Ediff):
    energies = []

    for c in c_list:
        atoms_c = atoms.copy()
        cell = atoms_c.cell.copy()
        cell[2, 2] = c
        atoms_c.set_cell(cell, scale_atoms=False)

        atoms_c.calc = GPAW(
            mode=PW(ecut),
            xc=xc,
            kpts={'size': (k, k, 1), 'gamma': True},
            txt=f'out/{atom_name}_vac_c{c}.txt'
        )

        E = atoms_c.get_potential_energy() / len(atoms_c)
        energies.append(E)

    plt.figure()
    plt.plot(c_list, energies, 'o-')
    plt.xlabel("Vacuum height c (Å)")
    plt.ylabel("Energy per atom (eV)")
    plt.title("Vacuum convergence")
    plt.tight_layout()
    plt.savefig("figs/vacuum_convergence.png")
    plt.show()

    return find_converged_value(c_list, energies, Ediff)

def kpoint_convergence(atoms, atom_name, ecut, xc, k_list, Ediff):
    energies = []

    for k in k_list:
        atoms_k = atoms.copy()
        atoms_k.calc = GPAW(
            mode=PW(ecut),
            xc=xc,
            kpts={'size': (k, k, 1), 'gamma': True},
            txt=f'out/{atom_name}_k{k}.txt'
        )

        E = atoms_k.get_potential_energy() / len(atoms_k)
        energies.append(E)

    plt.figure()
    plt.plot(k_list, energies, 'o-')
    plt.xlabel("k-point grid (k × k × 1)")
    plt.ylabel("Energy per atom (eV)")
    plt.title("k-point convergence")
    plt.tight_layout()
    plt.savefig("figs/kpoint_convergence.png")
    plt.show()

    return find_converged_value(k_list, energies, Ediff)

def ecut_convergence(atoms, atom_name, k, xc, ecut_list, Ediff):
    energies = []

    for ecut in ecut_list:
        atoms_e = atoms.copy()
        atoms_e.calc = GPAW(
            mode=PW(ecut),
            xc=xc,
            kpts={'size': (k, k, 1), 'gamma': True},
            txt=f'out/{atom_name}_ecut{ecut}.txt'
        )

        E = atoms_e.get_potential_energy() / len(atoms_e)
        energies.append(E)

    plt.figure()
    plt.plot(ecut_list, energies, 'o-')
    plt.xlabel("Plane-wave cutoff (eV)")
    plt.ylabel("Energy per atom (eV)")
    plt.title("ecut convergence")
    plt.tight_layout()
    plt.savefig("figs/ecut_convergence.png")
    plt.show()

    return find_converged_value(ecut_list, energies, Ediff)

def relax_buckling(atoms, k, ecut, xc, fmax=0.01):
    atoms_r = atoms.copy()
    atoms_r.calc = GPAW(
        mode=PW(ecut),
        xc=xc,
        kpts={'size': (k, k, 1), 'gamma': True},
        txt="out/relax_buckling.txt"
    )

    opt = BFGS(atoms_r, trajectory="out/buckling.traj")
    opt.run(fmax=fmax)

    return atoms_r

def optimize_lattice_constant(atoms, k, ecut, xc, a_factors):
    energies = []
    a_values = []

    for f in a_factors:
        atoms_a = atoms.copy()
        cell = atoms_a.cell.copy()
        cell[0, 0] *= f
        cell[1, 1] *= f
        atoms_a.set_cell(cell, scale_atoms=True)

        atoms_a.calc = GPAW(
            mode=PW(ecut),
            xc=xc,
            kpts={'size': (k, k, 1), 'gamma': True},
            txt=f"out/a_scan_{f:.3f}.txt"
        )

        E = atoms_a.get_potential_energy()
        energies.append(E)
        a_values.append(cell[0, 0])

    i_min = np.argmin(energies)
    return a_values[i_min], energies[i_min]

def converge_silicene(
    atoms,
    atom_name="Si",
    xc="PBE",
    Ediff=1e-3,
    k_list=range(6, 16, 2),
    ecut_list=range(300, 600, 50),
    c_list=range(12, 26, 2)
):

    print("Vacuum convergence")
    c_opt = vacuum_convergence(atoms, atom_name, k=6, ecut=400, xc=xc,
                               c_list=c_list, Ediff=Ediff)

    atoms.cell[2, 2] = c_opt

    print("k-point convergence")
    k_opt = kpoint_convergence(atoms, atom_name, ecut=400, xc=xc,
                               k_list=k_list, Ediff=Ediff)

    print("ecut convergence")
    ecut_opt = ecut_convergence(atoms, atom_name, k_opt, xc,
                                ecut_list, Ediff)

    print("Relaxing buckling")
    atoms_relaxed = relax_buckling(atoms, k_opt, ecut_opt, xc)

    print("Optimizing lattice constant a")
    a_opt, E0 = optimize_lattice_constant(
        atoms_relaxed, k_opt, ecut_opt, xc,
        a_factors=np.linspace(0.97, 1.03, 9)
    )

    return {
        "k": k_opt,
        "ecut": ecut_opt,
        "c": c_opt,
        "a": a_opt,
        "E0": E0
    }


a0 = 3.86 
c0 = 18.0  
buckling = 0.44 

silicene = Atoms(
    'Si2',
    cell=[a0, a0, c0, 90, 90, 120],
    positions=[
        (0.0, 0.0, c0 / 2 - buckling / 2),
        (a0 / 3, 2 * a0 / 3, c0 / 2 + buckling / 2)
    ],
    pbc=[True, True, False]
)

results = converge_silicene(silicene)
print(results)

# Results: k = 5, ecut = 200, c = 8, a = 4.062544476939418

# a = -7.095, b = -0.043, c = 0.000, V0 = 114.345 Å³
# 2.3455111473558 CC?

#       Step     Time          Energy          fmax
# BFGS:    0 10:41:25       34.942221      137.325991
# BFGS:    1 10:42:51        1.582530       41.708869
# BFGS:    2 10:44:07       -3.879991       21.427051
# BFGS:    3 10:45:43       -6.742202        8.614357
# BFGS:    4 10:46:39       -7.561097        3.973742
# BFGS:    5 10:47:30       -8.136620        3.554758
# BFGS:    6 10:48:40       -9.064657        3.089436
# BFGS:    7 10:49:52       -9.127238        1.900088
# BFGS:    8 10:51:01       -9.263605        1.985926
# BFGS:    9 10:51:38       -9.305953        1.827343
# BFGS:   10 10:52:23       -9.434621        1.119953
# BFGS:   11 10:52:54       -9.470824        0.604775
# BFGS:   12 10:53:26       -9.484560        0.511869
# BFGS:   13 10:53:54       -9.499929        0.446647
# BFGS:   14 10:54:26       -9.515465        0.216692
# BFGS:   15 10:54:56       -9.518316        0.023620
# BFGS:   16 10:55:17       -9.518369        0.002184
# {'k': 14, 'ecut': 450, 'c': 14, 'a': np.float64(3.86), 'E0': np.float64(-9.516275091914718)}