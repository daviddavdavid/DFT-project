from ase import Atoms, cell

a = 4.05
c = 10.99

# variable to ensure z positions are dependent on a, instead of c
z_dist_a = 0.16988097777 * a
z_dist = z_dist_a / c 

silicene = Atoms(
    'Si2',
    cell= [a, a, c, 90, 90, 120],
    scaled_positions= [(0, 0, 0), (1/3, 2/3, z_dist)]

)