import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


max_O_H_length = 1.2 # angstrom


"""
A, B, C:        Lattice parameters (in Angstrom)
coords:         List of coordinates for all atoms.
atom_list:      Labels for all atoms defined in 'coords'.
data_oxygen:    Coordinates of all oxygen atoms in electrolyte.
data_hydrogen:  Coordinates of all hydrogen atoms in electrolyte.
data_water_molecule:  List of list of coordinates in each water molecule unit.
data_com:       Coordinates of center of mass for each water molecule unit.
"""

def get_distance(x0, x1, A, B, C):
    """
    Get distance between two points taking into account periodic boundary
    condition.

    x0: coordinates of point 1 as numpy array
    x1: coordinates of point 2 as numpy array
    A, B, C: dimensions of periodic box
    return: distance
    """
    # ref: https://stackoverflow.com/questions/11108869/optimizing-python-\
    # distance-calculation-while-accounting-for-periodic-boundary-co
    dimensions = np.array([A, B, C])
    delta = np.abs(x0 - x1)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))

def apply_period_boundary(coord, A, B, C):
    """
    If a point lies outside of periodic box, update coordinates to lie inside of it.
    Only works in x and y directions.

    coord: coordinates of point
    A, B, C: dimensions of periodic box
    return: updated coordinates
    """
    new_coord = [ i for i in coord]
    # Translate x-coordinate.
    while new_coord[0] < 0:
        new_coord[0] = new_coord[0] + A
    while new_coord[0] > A:
        new_coord[0] = new_coord[0] - A
    # Translate y-coordinate.
    while new_coord[1] < 0:
        new_coord[1] = new_coord[1] + B
    while new_coord[1] > B:
        new_coord[1] = new_coord[1] - B
    return new_coord

def get_H2O_from_O_and_H(data_oxygen, data_hydrogen):
    """
    Arrange O and H coordinates in lists of [O,H,H] - corresponding to each H2O.
    If number of H in 1.2 A radius of O is not equal to 2, this means it is H3O+ or OH-.
    In this case, the whole set it discarded. TL;DR: only 3-atom H2O molecules are
    returned.

    data_oxygen: numpy array of O coordinates
    data_hydrogen: numpy array of H coordinates
    return: numpy array of H2O molecules listed as [O,H,H],
               numpy array of center of mass of each H2O molecule
    """
    # For each O, get H within a raduis of 1.2 A and
    #   save all the atoms as units of one H2O.
    data_water_molecule = []
    data_com = []
    for oxygen in data_oxygen:
        # Add a H2O.
        data_water_molecule.append([])
        data_water_molecule[-1].append(oxygen)

        # Find all H in the H2O
        for hydrogen in data_hydrogen:
            dist = get_distance(oxygen, hydrogen, A, B, C)
            if dist <= max_O_H_length:
                data_water_molecule[-1].append(hydrogen)

    # final list of water molecules with length = 3
    data_coords = []
    for mol in data_water_molecule:
        if len(mol) != 3:
            #print("Hydronium or hydroxide ion detected.")
            #print("Change max_O_H_length in line 10 and try again.")
            continue
        else:
            data_coords.append(mol)
            # Compute center of mass for each H2O.
            # Center of mass = sum [ position of atom * weight of atom ]
            # Mass of O = 16
            com_o = mol[0] * 16
            # Mass of H = 1
            list_h = [i * 1 for i in mol[1:]]
            com_h = sum(list_h)
            center_of_mass = (com_o + com_h) / (16 + (len(list_h) * 1))
            data_com.append(center_of_mass)

    data_water_molecule = np.array(data_coords)
    data_com = np.array(data_com)
    return data_water_molecule, data_com

def get_water_molecules_and_com(coords, atom_list, z_bot, z_top):
    """ Get H2O molecules and compute center of mass for each molecule. """
    # Isolate all H2O molecules in electrolyte based on their O.
    # Get all O and H present in the electrolyte.
    data_oxygen = []
    data_hydrogen = []
    for i, atom in enumerate(coords):
        if atom_list[i] == "O":
            # Check if atom lies in electrolyte region.
            if z_bot < atom[2] < z_top:
                data_oxygen.append(atom)
        if atom_list[i] == "H":
            # Check if atom lies in electrolyte region.
            # Add 1.2 ang to the search region, to account for H2O
            #   that lie on the edge of the (z_top - z_bot) region.
            # The final H2O in this region are determined by O coords,
            #   so extra counting of H is okay.
            if z_bot - 1.2 < atom[2] < z_top + 1.2:
                data_hydrogen.append(atom)
    data_oxygen = np.array(data_oxygen)
    data_hydrogen = np.array(data_hydrogen)

    data_water_molecule, data_com = get_H2O_from_O_and_H(data_oxygen, data_hydrogen)

    return data_water_molecule, data_com

def get_atoms_pos(atom_type, coords, atom_list, z_bot, z_top):
    """ Get coords of all atoms of type atom_type in region (z_top - z_bot).
    E.g., "O", "H", etc """
    data = []
    for i, atom in enumerate(coords):
        if atom_list[i] == atom_type:
            # check if atom lies in electrolyte region
            if z_bot < atom[2] < z_top:
                data.append(atom)
    data = np.array(data)

    # include data_com for consistency with get_water_molecules_and_com method
    # for atoms, center of mass is at the position coordinate
    data_com = np.array(data)
    return data, data_com

def get_water_cuboid(coords, atom_list, x1, x2, y1, y2, z1, z2, A, B, C):
    """ Get coords of all H2O in region bounded in x1, x2, y1, y2, z1, z2."""
    data_oxygen = []
    data_hydrogen = []
    for i, atom in enumerate(coords):
        # Check if atom lies in box region.
        if atom_list[i] == "O":
            pbc_atom = apply_period_boundary(atom, A, B, C)
            # (xi - A) and (xi + A) regions are checked to account for cuboid box
            #   crossing periodic boundary.
            in_x = ((x1 < pbc_atom[0] < x2)
                or (x1 - A < pbc_atom[0] < x2 - A)
                or (x1 + A < pbc_atom[0] < x2 + A))
            in_y = ((y1 < pbc_atom[1] < y2)
                or (y1 - B < pbc_atom[1] < y2 - B)
                or (y1 + B < pbc_atom[1] < y2 + B))
            in_z = (z1 < pbc_atom[2] < z2)
            if in_x and in_y and in_z:
                data_oxygen.append(atom)
        if atom_list[i] == "H":
            pbc_atom = apply_period_boundary(atom, A, B, C)
            # Add 1.2 ang to the search region, to account for H2O
            #   that lie on the edge of the region.
            # The final H2O in this region are determined by O coords,
            #   so extra counting of H is okay.
            in_x = ((x1 - 1.2 < pbc_atom[0] < x2 + 1.2)
                or (x1 - A - 1.2 < pbc_atom[0] < x2 - A + 1.2)
                or (x1 + A - 1.2 < pbc_atom[0] < x2 + A + 1.2))
            in_y = ((y1 - 1.2 < pbc_atom[1] < y2 + 1.2)
                or (y1 - B - 1.2 < pbc_atom[1] < y2 - B + 1.2)
                or (y1 + B - 1.2 < pbc_atom[1] < y2 + B + 1.2))
            in_z = (z1 - 1.2 < pbc_atom[2] < z2 + 1.2)
            if in_x and in_y and in_z:
                data_hydrogen.append(atom)
    data_oxygen = np.array(data_oxygen)
    data_hydrogen = np.array(data_hydrogen)

    data_water_molecule, data_com = get_H2O_from_O_and_H(data_oxygen, data_hydrogen)
    return data_water_molecule, data_com

def get_water_sphere(coords, atom_list, p, r):
    """ Get coords of all H2O in region bounded in sphere centered at p0 with radius r."""
    data_oxygen = []
    data_hydrogen = []
    for i, atom in enumerate(coords):
        if atom_list[i] == "O":
            # check if atom lies in spherical region
            dist = get_distance(p, atom, A, B, C)
            if 0.0 < dist < r:
                data_oxygen.append(atom)
        if atom_list[i] == "H":
            # check if atom lies in spherical region
            # Add 1.2 ang to the search region, to account for H2O
            #   that lie on the edge of the region.
            # The final H2O in this region are determined by O coords,
            #   so extra counting of H is okay.
            dist = get_distance(p, atom, A, B, C)
            if dist < r + 1.2:
                data_hydrogen.append(atom)
    data_oxygen = np.array(data_oxygen)
    data_hydrogen = np.array(data_hydrogen)

    data_water_molecule, data_com = get_H2O_from_O_and_H(data_oxygen, data_hydrogen)
    return data_water_molecule, data_com


class Trajectory:
    def __init__(self, filename, file_type, skip, z_bot_interface, z_top_interface,
                 start_timestep, stop_timestep, interface_offset=0.0, resolution=201,
                 d_angle=5, z_range_min=0.0, z_range_max=5.0, x_min=0.0,
                 x_max=0.0, y_min=0.0, y_max=0.0, cuboid_resolution=11,
                 center_atom_list=[0], r=0.0, spherical_resolution=11):
        """
        filename         : Path to the trajectory file.
        file_type         : xyz or vasp
        skip             : Number of snapshots to be skipped between two
                           configurations that are evaluated (for example,
                           if trajectory is 9000 steps long, and skip = 10,
                           every tenth step is evaluated, 900 steps in
                           total; use skip=1 to take every step of the MD).
        z_bot_interface  : Average vertical coordinate for interface below
                           water layer in Angstrom.
        z_top_interface  : Average vertical coordinate for interface above
                           water layer in Angstrom.
        interface_offset : Distance between interface and region of water
                           with bulk-like properties.
        resolution       : Number of points in the final radial
                           distribution function.
        z_range_min      : The distibution of bond angles are calculated
                           in the volume between z_range_min and z_range_max
                           vertical coordinates.
                           (w.r.t z_bot)
        z_range_max      : The distibution of bond angles are calculated
                           in the volume between z_range_min and z_range_max
                           vertical coordinates.
                           (w.r.t z_bot)
        x_min, x_max, y_min, y_max      : Together with z_range_min and
                           z_range_max, these coordinates describe a cuboid region
                           for analysis.
        center_atom_list, r     : Defines a sperical region centered at each atom in
                           center_atom_list with radius r. The center_atom_list keyword
                           stores atom numbers, e.g., of all step sites, and coordinates
                           of the atom are determined from trajectory, and stored in list p0.
        spherical_resolution   : Number of points in the analysis of spherical
                           region
                           """

        self.filename = filename
        self.file_type = file_type
        self.skip = skip
        self.z_top = z_top_interface - interface_offset
        self.z_bot = z_bot_interface + interface_offset
        self.start_timestep = start_timestep
        self.stop_timestep = stop_timestep
        self.resolution = resolution

        self.surface_normal = np.array([0, 0, 1])
        self.d_angle = d_angle
        # Uncomment this line for considering full electrolyte volume for angle distributions.
        #z_range_max = self.z_top - self.z_bot
        self.z_range = np.array([z_range_min, z_range_max])
        self.x_range = np.array([x_min, x_max])
        self.y_range = np.array([y_min, y_max])
        self.cuboid_resolution = cuboid_resolution
        self.center_atom_list = center_atom_list
        self.r = r
        self.spherical_resolution = spherical_resolution

        self.parse_input()

    def parse_input(self):
        with open(self.filename, 'r') as f:
            data = f.readlines()

        if self.file_type == 'xyz':
            # Get total number of atoms.
            self.n_atoms = int(data[0].strip())
            # Get total number of steps.
            self.n_steps_all = len(data) / (self.n_atoms+2)

            # Get list of labels for all atoms.
            data_first_step = data[2:self.n_atoms+2]
            data_first_step = np.array([ i.split() for i in data_first_step ])
            self.atom_list = data_first_step[:,0]
            self.atoms = list(set(self.atom_list))

            # Get indices of lines containing atomic positions.
            #   from the time range defined by user.
            data_start_index = (self.start_timestep-1) * (self.n_atoms+2)
            data_stop_index = self.stop_timestep * (self.n_atoms+2)
            data = data[data_start_index:data_stop_index]

            # Calculate number of total steps.
            self.n_steps_total = int(len(data) / (self.n_atoms+2))
            self.n_steps = self.n_steps_total // self.skip

            # Initialize list of coordinates.
            self.coordinates = np.zeros((self.n_steps, self.n_atoms, 3))
            for step in range(self.n_steps):
                coords = np.zeros((self.n_atoms, 3))

                # Get coordinates.
                i = step * self.skip * (self.n_atoms+2)
                for j, line in enumerate(data[i+2 : i+self.n_atoms+2]):
                    coords[j,:] = [ float(value) for value in line.split()[1:4] ]

                self.coordinates[step] = coords
            print('parse done')
        elif self.file_type == 'vasp':
            self.atoms = data[5].split()
            self.atoms_count = list(map(int,data[6].split()))
            # Get list of labels for all atoms
            self.atom_list = []
            for i in range(len(self.atoms)):
                self.atom_list += [self.atoms[i]] * self.atoms_count[i]

            # Get total number of atoms
            self.n_atoms = sum(self.atoms_count)
            # Get total number of steps
            self.n_steps_all = int((len(data)-7) / (self.n_atoms+1))

            # Get indices of lines containing atomic positions
            # from the time range defined by user.
            data_start_index = ((self.start_timestep-1) * (self.n_atoms+1)) + 7
            data_stop_index = (self.stop_timestep * (self.n_atoms+1)) + 7
            data = data[data_start_index:data_stop_index]

            # Calculate number of total steps
            self.n_steps_total = int(len(data) / (self.n_atoms+1))
            self.n_steps = self.n_steps_total // self.skip

            # Initialize list of coordinates.
            self.coordinates = np.zeros((self.n_steps, self.n_atoms, 3))
            for step in range(self.n_steps):
                coords = np.zeros((self.n_atoms, 3))

                # Get coordinates
                i = step * self.skip * (self.n_atoms+1)
                for j, line in enumerate(data[i+1 : i+self.n_atoms+1]):
                    coords[j,:] = [float(value) for value in line.split()]

                # Convert to cartesian coordinates
                # A, B, C are lattice parameters defined below
                for i in range(len(coords)):
                    coords[i] = [coords[i][0]*A, coords[i][1]*B, coords[i][2]*C]

                self.coordinates[step] = coords
            print('parse done')

    def compute_cuboid_density_profile(self, atom_type):
        # Define step size, dz.
        self.dz = (self.z_range[1] - self.z_range[0]) / self.cuboid_resolution
        # Divide electrolyte length into units of dz.
        self.z_step = np.linspace(0.0, self.cuboid_resolution * self.dz, self.cuboid_resolution)

        # Initialize list for storing density profile, rho_of_z.
        self.rho_of_z = np.zeros(self.cuboid_resolution)

        for step in range(self.n_steps):
            if atom_type == "H2O":
                # Get water molecules and centers of mass.
                data_coords, data_com = get_water_cuboid(
                    self.coordinates[step], self.atom_list,
                    self.x_range[0], self.x_range[1],
                    self.y_range[0], self.y_range[1],
                    self.z_bot + self.z_range[0], self.z_bot + self.z_range[1],
                    A, B, C)
            else:
                # Only works with H2O for now.
                print("Incorrect atom_type. Please choose 'H2O'.")
                return

            # Sweep over intervals of dz width from self.z_range[0] to self.z_range[1]
            #   and count all molecules/atoms (in case of H2O, check coords of O) that
            #   fall in the region at each step.
            for i in range(self.cuboid_resolution):
                z1 = self.z_bot + self.z_range[0] + (i * self.dz)
                z2 = z1 + self.dz
                for water in data_coords:
                    if z1 < water[0][2] <= z2:
                        self.rho_of_z[i] += 1

        # Normalize for number of steps.
        self.rho_of_z = self.rho_of_z / self.n_steps

        num_atoms = sum(self.rho_of_z)
        print('Total number:', num_atoms)

        # Convert to units of g/cm^3.
        if atom_type == "H2O":
            mass = 16+1+1
        else:
            # Only works with H2O for now.
            print("Incorrect atom_type. Please choose 'H2O'.")
            return
        volume = (
            (self.x_range[1] - self.x_range[0])
            * (self.y_range[1] - self.y_range[0])
            * self.dz * 10**(-24))
        #self.rho_of_z = (self.rho_of_z * (16+1+1)) / ((6.02214076 * 10**23) * volume)
        print('Total number density:', sum(self.rho_of_z) / self.resolution)

    def compute_cuboid_dipole_orientation(self):
        # Define step size, dz.
        self.dz = (self.z_range[1] - self.z_range[0]) / self.cuboid_resolution
        # Divide electrolyte length into units of dz.
        self.z_step = np.linspace(0.0, self.cuboid_resolution * self.dz, self.cuboid_resolution)

        # Initialize list for storing density profile, rho_of_z.
        self.rho_of_z = np.zeros(self.cuboid_resolution)

        cos_psi_all_steps = [ [] for _ in self.z_step]

        for step in range(self.n_steps):
            # Get water molecules and centers of mass.
            data_coords, data_com = get_water_cuboid(
                self.coordinates[step], self.atom_list,
                self.x_range[0], self.x_range[1],
                self.y_range[0], self.y_range[1],
                self.z_bot + self.z_range[0], self.z_bot + self.z_range[1],
                A, B, C)

            # Sweep over intervals of dz width from self.z_range[0] to self.z_range[1]
            for i in range(self.resolution):
                z1 = self.z_bot + self.z_range[0] + (i * self.dz)
                z2 = z1 + self.dz
                water_in_region = []
                for water in data_coords:
                    if z1 < water[0][2] <= z2:
                        water_in_region.append(water)

                if len(water_in_region) > 0:
                    cos_psi = np.full(len(water_in_region), -2.0)
                    for k, water in enumerate(water_in_region):
                        if len(water) == 3:
                            # Get O-H unit vectors.
                            vec1 = (
                                (water[1]-water[0]) / np.linalg.norm(water[1]-water[0]))
                            vec2 = (
                                (water[2]-water[0]) / np.linalg.norm(water[2]-water[0]))
                            # Compute angle-bisector unit vector.
                            bisector = (vec1 + vec2) / np.linalg.norm(vec1 + vec2)

                            # https://stackoverflow.com/a/13849249
                            # Get psi in radians.
                            if water[0][2] < (self.z_top - self.z_bot)/2:
                                psi = np.arccos(np.dot(bisector, self.surface_normal))
                            elif water[0][2] > (self.z_top - self.z_bot)/2:
                                # reverse direction of surface normal at opposite surface
                                psi = np.arccos(np.dot(bisector, -1*self.surface_normal))
                            cos_psi[k] = np.cos(psi)

                    # Sanity check, to ensure psi from all waters were counted.
                    if -2 in cos_psi:
                        print("cos(psi) for some H2O in region", z1, z2, "were not populated")

                    # Collect all cos_psi which occur within each incremental region.
                    for value in cos_psi:
                        cos_psi_all_steps[i].append(value)

            # Sum over all cos_psi within each incremental region.
            for i, value in enumerate(cos_psi_all_steps):
                self.rho_of_z[i] = np.sum(value)

        # Normalize for number of steps.
        self.rho_of_z = self.rho_of_z / self.n_steps

        # Convert to g/cm3.
        volume = (
            (self.x_range[1] - self.x_range[0])
            * (self.y_range[1] - self.y_range[0])
            * self.dz * 10**(-24))
        self.rho_of_z = (self.rho_of_z * (16+1+1)) / ((6.02214076 * 10**23) * volume)

    def compute_sphere_density_profile(self, atom_type):
        # Define step size, dr.
        self.dr = self.r / self.spherical_resolution
        # Divide radial length into units of dr.
        self.r_step = np.linspace(
            0.0, self.spherical_resolution * self.dr, self.spherical_resolution)

        # Initialize list for storing density profile, rho_of_r.
        self.rho_of_r = np.zeros(self.spherical_resolution)

        for step in range(self.n_steps):
            # Get coordinates of center atoms.
            p0 = [ self.coordinates[step][c-1] for c in self.center_atom_list ]

            # Iterate over each center_atom.
            for p in p0:
                if atom_type == "H2O":
                    # Get water molecules and centers of mass.
                    data_coords, data_com = get_water_sphere(
                        self.coordinates[step], self.atom_list, p, self.r)
                else:
                    # Only works with H2O for now.
                    print("Incorrect atom_type. Please choose 'H2O'.")
                    return

                # Sweep over intervals of dr width from 0 to self.r and
                #   count all molecules/atoms that fall in the spherical region
                #   at each step.
                for i in range(self.spherical_resolution):
                    r1 = 0.0001 + (i * self.dr)
                    r2 = r1 + self.dr
                    # Normalize with respect to volume of region, dv = d(volume).
                    #dv = 4 * np.pi * r1**2 * self.dr
                    dv = 1
                    for water in data_coords:
                        dist = get_distance(p, water[0], A, B, C)
                        if r1 < dist <= r2:
                            self.rho_of_r[i] += 1/dv

        # Normalize for number of steps.
        self.rho_of_r = self.rho_of_r / (self.n_steps)
        print("Average number of water molecules within", self.r, "Ang of center atom:", sum(self.rho_of_r))

    def compute_sphere_angle_distribution(self, atom_type):
        angle_resolution = int(180/self.d_angle)
        self.angle_step = np.linspace(0, 180, angle_resolution)

        # psi: Angle between dipole (bisector of H-O-H) and surface normal.
        # theta: Angle between O-H bond and surface normal.
        self.psi_dist = np.zeros(angle_resolution)
        self.theta_dist = np.zeros(angle_resolution)

        for step in range(self.n_steps):
            # Get coordinates of center atoms.
            p0 = [ self.coordinates[step][c-1] for c in self.center_atom_list ]

            # Iterate over each center_atom.
            for p in p0:
                if atom_type == "H2O":
                    # Get water molecules and centers of mass.
                    data_coords, data_com = get_water_sphere(
                        self.coordinates[step], self.atom_list, p, self.r)
                else:
                    # Only works with H2O for now.
                    print("Incorrect atom_type. Please choose 'H2O'.")
                    return

                # Get angle distributions
                for water in data_coords:
                    # Get  O-H unit vectors.
                    vec1 = (
                        (water[1]-water[0]) / np.linalg.norm(water[1]-water[0]))
                    vec2 = (
                        (water[2]-water[0]) / np.linalg.norm(water[2]-water[0]))

                    # Compute psi, dipole-normal angle.
                    bisector = (vec1 + vec2) / np.linalg.norm(vec1 + vec2)
                    # https://stackoverflow.com/a/13849249
                    # Convert radians to degrees.
                    psi = np.rad2deg(np.arccos(np.dot(bisector, self.surface_normal)))
                    index = int(psi / self.d_angle)
                    if 0 < index < angle_resolution:
                        self.psi_dist[index] += 1.0

                    # Compute theta, OH-normal angle.
                    # https://stackoverflow.com/a/13849249
                    theta = np.rad2deg(np.arccos(np.dot(vec1, self.surface_normal)))
                    index = int(theta / self.d_angle)
                    if 0 < index < angle_resolution:
                        self.theta_dist[index] += 1.0

                    theta = np.rad2deg(np.arccos(np.dot(vec2, self.surface_normal)))
                    index = int(theta / self.d_angle)
                    if 0 < index < angle_resolution:
                        self.theta_dist[index] += 1.0

        # Normalize for number of steps and number of center atoms.
        num_centers = len(p0)
        self.psi_dist = self.psi_dist / (self.n_steps * num_centers)
        self.theta_dist = self.theta_dist / (self.n_steps * num_centers)

    def compute_density_profile(self, atom_type):
        # Define step size, dz.
        self.dz = (self.z_top - self.z_bot) / self.resolution
        # Divide electrolyte length into units of dz.
        self.z_step = np.linspace(0.0, self.resolution * self.dz, self.resolution)

        # Initialize list for storing density profile, rho_of_z.
        self.rho_of_z = np.zeros(self.resolution)

        for step in range(self.n_steps):
            if atom_type == "H2O":
                # Get water molecules and centers of mass from both sides.
                data_coords, data_com = get_water_molecules_and_com(
                    self.coordinates[step], self.atom_list, self.z_bot, self.z_top)
            elif atom_type in self.atoms:
                # Get atoms of type atom_type. E.g., "O", "H", etc. from both sides.
                data_coords, data_com = get_atoms_pos(
                    atom_type,
                    self.coordinates[step], self.atom_list, self.z_bot, self.z_top)
            else:
                print("Incorrect atom_type. Please choose one of 'H2O' or", self.atoms)
                return

            # Sweep over intervals of dz width and count all molecules/atoms that
            #   fall in the region at each step.
            for i in range(self.resolution):
                z1 = self.z_bot + (i * self.dz)
                z2 = z1 + self.dz
                for water in data_coords:
                    if atom_type == "H2O":
                        # z-coordinate of O atom in H2O.
                        coord = water[0][2]
                    else:
                        # z-coordinate of atom. Here, "water" variable is not water
                        #   but an atom of atom_type.
                        coord = water[2]
                    if z1 < coord <= z2:
                        self.rho_of_z[i] += 1

        # Normalize for number of steps.
        self.rho_of_z = self.rho_of_z / self.n_steps

        # Sanity check:
        # sum(self.rho_of_z) should be equal to number of molecules/atoms in input file.
        # If it is not,
        # (1) Check that n_steps is correct.
        # (2) Check for formation of hydronuim ions (i.e. some molecules in
        #     data_water_molecules will have length != 3) which indicate that O from
        #     water may have adsorbed on surface, thus leaving the electrolyte region.
        num_atoms = sum(self.rho_of_z)
        print('Total number:', num_atoms)

        # Convert to units of g/cm^3.
        if atom_type == "H2O":
            mass = 16+1+1
        elif atom_type == "O":
            mass = 16
        elif atom_type == "H":
            mass = 1
        elif atom_type == "F":
            mass = 19
        elif atom_type == "Na":
            mass = 23
        else:
            print("Mass of atom", atom_type, "not defined. Please edit the script to include it.")
            return
        total_volume = A * B * self.dz * 10**(-24)
        #self.rho_of_z = (self.rho_of_z * mass) / ((6.02214076 * 10**23) * total_volume)
        print('Total number density:', sum(self.rho_of_z) / self.resolution)

    def compute_angle_distribution(self):
        angle_resolution = int(180/self.d_angle)
        self.angle_step = np.linspace(0, 180, angle_resolution)

        # psi: Angle between dipole (bisector of H-O-H) and surface normal.
        # theta: Angle between O-H bond and surface normal.
        self.psi_dist = np.zeros(angle_resolution)
        self.theta_dist = np.zeros(angle_resolution)

        for step in range(self.n_steps):

            # Get water molecules and centers of mass.
            data_coords, data_com = get_water_molecules_and_com(
                    self.coordinates[step], self.atom_list,
                    self.z_bot + self.z_range[0], self.z_bot + self.z_range[1])

            for water in data_coords:
                if len(water) == 3:
                    # Get O-H unit vectors.
                    vec1 = (
                        (water[1]-water[0]) / np.linalg.norm(water[1]-water[0]))
                    vec2 = (
                        (water[2]-water[0]) / np.linalg.norm(water[2]-water[0]))
                    # Compute angle-bisector unit vector.
                    bisector = (vec1 + vec2) / np.linalg.norm(vec1 + vec2)

                    # Compute psi, dipole-normal angle, and theta, OH-normal angle.
                    # Convert radians to degrees.
                    # https://stackoverflow.com/a/13849249
                    if water[0][2] < (self.z_top - self.z_bot)/2:
                        psi = np.rad2deg(np.arccos(np.dot(bisector, self.surface_normal)))
                        theta1 = np.rad2deg(np.arccos(np.dot(vec1, self.surface_normal)))
                        theta2 = np.rad2deg(np.arccos(np.dot(vec2, self.surface_normal)))
                    elif water[0][2] > (self.z_top - self.z_bot)/2:
                        # reverse direction of surface normal at opposite surface
                        psi = np.rad2deg(np.arccos(np.dot(bisector, -1*self.surface_normal)))
                        theta1 = np.rad2deg(np.arccos(np.dot(vec1, -1*self.surface_normal)))
                        theta2 = np.rad2deg(np.arccos(np.dot(vec2, -1*self.surface_normal)))

                    # Update distribution.
                    index_psi = int(psi / self.d_angle)
                    if 0 < index_psi < angle_resolution:
                        self.psi_dist[index_psi] += 1.0
                    index_theta1 = int(theta1 / self.d_angle)
                    if 0 < index_theta1 < angle_resolution:
                        self.theta_dist[index_theta1] += 1.0
                    index_theta2 = int(theta2 / self.d_angle)
                    if 0 < index_theta2 < angle_resolution:
                        self.theta_dist[index_theta2] += 1.0

        # Normalize.
        self.psi_dist = self.psi_dist / sum(self.psi_dist)
        self.theta_dist = self.theta_dist / sum(self.theta_dist)

    def compute_avg_cos_dipole(self):
        cos_psi_all_steps = []

        for step in range(self.n_steps):

            # Get water molecules and centers of mass.
            data_coords, data_com = get_water_molecules_and_com(
                    self.coordinates[step], self.atom_list,
                    self.z_bot + self.z_range[0], self.z_bot + self.z_range[1])

            cos_psi = np.full(len(data_coords), -2.0)

            for k, water in enumerate(data_coords):
                if len(water) == 3:
                    # Get O-H unit vectors.
                    vec1 = (
                        (water[1]-water[0]) / np.linalg.norm(water[1]-water[0]))
                    vec2 = (
                        (water[2]-water[0]) / np.linalg.norm(water[2]-water[0]))
                    # Compute angle-bisector unit vector.
                    bisector = (vec1 + vec2) / np.linalg.norm(vec1 + vec2)

                    # https://stackoverflow.com/a/13849249
                    # Get psi in radians.
                    if water[0][2] < (self.z_top - self.z_bot)/2:
                            psi = np.arccos(np.dot(bisector, self.surface_normal))
                    elif water[0][2] > (self.z_top - self.z_bot)/2:
                            # reverse direction of surface normal at opposite surface
                            psi = np.arccos(np.dot(bisector, -1*self.surface_normal))
                    cos_psi[k] = np.cos(psi)

            # Sanity check, to ensure psi from all waters were counted.
            if -2 in cos_psi:
                print("cos(psi) for some H2O were not counted")

            for i in cos_psi:
                cos_psi_all_steps.append(i)

        # Normalize
        cos_psi_all_steps = np.mean(cos_psi_all_steps)
        print('cos_psi_all_steps:', cos_psi_all_steps)

    def compute_dipole_orientation(self):
        # Define step size, dz.
        self.dz = (self.z_top - self.z_bot) / self.resolution
        # Divide electrolyte length into units of dz.
        self.z_step = np.linspace(0.0, self.resolution * self.dz, self.resolution)

        # Initialize list for storing density profile, rho_of_z.
        self.rho_of_z = np.zeros(self.resolution)

        cos_psi_all_steps = [ [] for _ in self.z_step]

        for step in range(self.n_steps):

            # Get water molecules and centers of mass.
            data_coords, data_com = get_water_molecules_and_com(
                    self.coordinates[step], self.atom_list, self.z_bot, self.z_top)

            # Sweep over intervals of dz width from z_bot to z_top.
            for i in range(self.resolution):
                z1 = self.z_bot + (i * self.dz)
                z2 = z1 + self.dz
                water_in_region = []
                for water in data_coords:
                    if z1 < water[0][2] <= z2:
                        water_in_region.append(water)

                if len(water_in_region) > 0:
                    cos_psi = np.full(len(water_in_region), -2.0)
                    for k, water in enumerate(water_in_region):
                        if len(water) == 3:
                            # Get O-H unit vectors.
                            vec1 = (
                                (water[1]-water[0]) / np.linalg.norm(water[1]-water[0]))
                            vec2 = (
                                (water[2]-water[0]) / np.linalg.norm(water[2]-water[0]))
                            # Compute angle-bisector unit vector.
                            bisector = (vec1 + vec2) / np.linalg.norm(vec1 + vec2)

                            # https://stackoverflow.com/a/13849249
                            # Get psi in radians.
                            if water[0][2] < (self.z_top - self.z_bot)/2:
                                psi = np.arccos(np.dot(bisector, self.surface_normal))
                            elif water[0][2] > (self.z_top - self.z_bot)/2:
                                # reverse direction of surface normal at opposite surface
                                psi = np.arccos(np.dot(bisector, -1*self.surface_normal))
                            cos_psi[k] = np.cos(psi)

                    # Sanity check, to ensure psi from all waters were counted.
                    if -2 in cos_psi:
                        print("cos(psi) for some H2O in region", z1, z2, "were not counted")

                    # Collect all cos_psi which occur within each incremental region.
                    for value in cos_psi:
                        cos_psi_all_steps[i].append(value)

            # Sum over all cos_psi within each incremental region.
            for i, value in enumerate(cos_psi_all_steps):
                self.rho_of_z[i] = np.sum(value)

        # Normalize for number of steps.
        self.rho_of_z = self.rho_of_z / self.n_steps

        # Convert to g/cm3.
        volume = A * B * self.dz * 10**(-24)
        self.rho_of_z = (self.rho_of_z * (16+1+1)) / ((6.02214076 * 10**23) * volume)

    def plot_cuboid_density_profile(self, atom_type):
        """ Plots density profile in a cuboidal region. """
        if not self.rho_of_z.any():
            print('compute the density profile first\n')
            return
        print('plotting density profile')
        # normalize so that each point corresponds to density in small box region
        #rho_dist = self.rho_of_z * self.resolution * 0.5
        rho_dist = self.rho_of_z
        # plot
        plt.plot(self.z_step, rho_dist, 'o-')
        # save data
        with open('cuboid_density_profile.dat', 'w') as f:
            for i in range(len(self.z_step)):
                f.write(str(self.z_step[i]) + '\t' + str(rho_dist[i]) + '\n')
        plt.xlabel('z (Å)')
        plt.ylabel('ρ$_{H_{2}O}$ (g/cm$^3$)')
        plt.title('H$_2$O density distribution')
        plt.xlim([0.0, (self.z_range[1] - self.z_range[0])])
        plt.tight_layout()
        plt.savefig('cuboid_density_profile.png', dpi=600, format='png')
        plt.close()

    def plot_cuboid_dipole_orientation(self):
        """ Plots the density orientation profile. """
        if not self.rho_of_z.any():
            print('compute the density orientation profile first\n')
            return
        print('plotting')
        # plot
        plt.plot(self.z_step, self.rho_of_z)
        # save data
        with open('cuboid_density_orientation.dat', 'w') as f:
            for i in range(len(self.z_step)):
                f.write(str(self.z_step[i]) + '\t' + str(self.rho_of_z[i]) + '\n')
        plt.xlabel('z (Å)')
        plt.ylabel('ρ$_{H_{2}O}$ cos$\\psi$ (g/cm$^3$)')
        #plt.title('H$_2$O density profile')
        plt.xlim([0.0, (self.z_range[1] - self.z_range[0])])
        plt.tight_layout()
        plt.savefig('cuboid_density_orientation.png', dpi=600, format='png')
        plt.close()

    def plot_sphere_density_profile(self, atom_type):
        """ Plots density profile in a sphere around individual atoms. """
        if not self.rho_of_r.any():
            print('compute the density profile first\n')
            return
        print('plotting density profile')
        # plot
        plt.plot(self.r_step,self.rho_of_r)
        # save data
        with open('radial_density_profile.dat', 'w') as f:
            for i in range(len(self.r_step)):
                f.write(str(self.r_step[i]) + '\t' + str(self.rho_of_r[i]) + '\n')
        plt.xlabel('r (Å)')
        plt.ylabel('P')
        plt.title('Number density distribution')
        plt.xlim([0.0, self.r])
        #plt.xlim([0.0, (self.z_top - self.z_bot)])
        #plt.ylim([0,4])
        plt.tight_layout()
        plt.savefig('radial_density_profile.png', dpi=600, format='png')
        plt.close()

    def plot_sphere_angle_distribution(self, atom_type):
        """ Plots angle distributions in a sphere around individual atoms. """
        if not self.psi_dist.any() or not self.theta_dist.any():
            print('compute the density profile first\n')
            return

        avg_dipole = np.sum(self.angle_step * self.psi_dist) / np.sum(self.psi_dist)
        print("average dipole moment:", avg_dipole)
        print('plotting dipole distribution')
        plt.plot(self.angle_step, self.psi_dist)
        with open('radial_dipole_data.dat', 'w') as f:
            for i in range(len(self.angle_step)):
                f.write(str(self.angle_step[i]) + '\t' + str(self.psi_dist[i]) + '\n')
        plt.xlabel('Angle from surface normal (degree)')
        plt.ylabel('P')
        plt.title('Dipole angle distribution')
        plt.xticks(np.linspace(0, 180, 10))
        plt.xlim([0,180])
        #plt.ylim=[0,0.04],
        plt.tight_layout()
        plt.savefig('radial_dipole_data.png', dpi=600, format='png')
        plt.close()

        avg_angle = np.sum(self.angle_step * self.theta_dist) / np.sum(self.theta_dist)
        print("average O-H angle:", avg_angle)
        print('plotting O-H distribution')
        plt.plot(self.angle_step, self.theta_dist)
        with open('radial_O-H_angle.dat', 'w') as f:
            for i in range(len(self.angle_step)):
                f.write(str(self.angle_step[i]) + '\t' + str(self.theta_dist[i]) + '\n')
        plt.xlabel('Angle from surface normal (degree)')
        plt.ylabel('P')
        plt.title('O-H angle distribution')
        plt.xticks(np.linspace(0, 180, 10))
        plt.xlim([0,180])
        #plt.ylim=[0,0.04],
        plt.tight_layout()
        plt.savefig('radial_O-H_angle.png', dpi=600, format='png')
        plt.close()

    def plot_density_profile(self, atom_type):
        """ Plots the density profile. """
        if not self.rho_of_z.any():
            print('compute the density profile first\n')
            return
        if self.z_top - self.z_bot > 18.0:
            # Compute bulk density (>= 9\\A from electrode).
            bulk_index_bot = int(len(self.z_step) * 9 / (self.z_top - self.z_bot))
            bulk_rho_of_z = self.rho_of_z[bulk_index_bot:] / self.resolution
            total_volume = (
                A * B * (self.z_top - self.z_bot)
                * 10**(-24) / 2.0)
            bulk_volume = (
                A * B * (self.z_top - self.z_bot -  self.dz * 9 * 2)
                * 10**(-24) / 2.0)
            bulk_density = np.sum(bulk_rho_of_z) * total_volume / bulk_volume
            print("Average bulk density (>= 9\\A from electrode):", bulk_density, 'g/cm3')
        print('plotting density profile')
        # plot
        plt.plot(self.z_step, self.rho_of_z)
        # save data
        with open('density_profile.dat', 'w') as f:
            for i in range(len(self.z_step)):
                f.write(str(self.z_step[i]) + '\t' + str(self.rho_of_z[i]) + '\n')
        plt.xlabel('z (Å)')
        if atom_type == "H2O":
            plt.ylabel('ρ$_{H_{2}O}$ (g/cm$^3$)')
            plt.title('H$_2$O density distribution')
        elif atom_type == "O":
            plt.ylabel('ρ$_{O}$ (g/cm$^3$)')
            plt.title('O density distribution')
        elif atom_type == "H":
            plt.ylabel('ρ$_{H}$ (g/cm$^3$)')
            plt.title('H density distribution')
        else:
            plt.ylabel('ρ (g/cm$^3$)')
            plt.title('Density distribution')
        plt.xlim([0.0, (self.z_top - self.z_bot)])
        plt.tight_layout()
        plt.savefig('density_profile.png', dpi=600, format='png')
        plt.close()

    def plot_dipole_distribution(self):
        """ Plots the distribution of dipole-normal angles (psi). """
        if not self.psi_dist.any():
            print('compute angle distributions first\n')
            return
        avg_dipole = np.sum(self.angle_step * self.psi_dist) / np.sum(self.psi_dist)
        print("average dipole moment:", avg_dipole)
        print('plotting dipole distribution')
        plt.plot(self.angle_step, self.psi_dist)
        with open('dipole_distribution.dat', 'w') as f:
            for i in range(len(self.angle_step)):
                f.write(str(self.angle_step[i]) + '\t' + str(self.psi_dist[i]) + '\n')
        plt.xlabel('Angle from surface normal (degree)')
        plt.ylabel('Probability density')
        plt.title('Dipole angle distribution')
        plt.xticks(np.linspace(0, 180, 10))
        plt.xlim([0,180])
        plt.tight_layout()
        plt.savefig('dipole_distribution.png', dpi=600, format='png')
        plt.close()

    def plot_OH_angle_distribution(self):
        """ Plots the distribution of OH-normal angles (theta). """
        if not self.theta_dist.any():
            print('compute angle distributions first\n')
            return
        avg_angle = np.sum(self.angle_step * self.theta_dist) / np.sum(self.theta_dist)
        print("average O-H angle:", avg_angle)
        print('plotting O-H distribution')
        plt.plot(self.angle_step, self.theta_dist)
        with open('OH_angle_distribution.dat', 'w') as f:
            for i in range(len(self.angle_step)):
                f.write(str(self.angle_step[i]) + '\t' + str(self.theta_dist[i]) + '\n')
        plt.xlabel('Angle from surface normal (degree)')
        plt.ylabel('Probability density')
        plt.title('O-H angle distribution')
        plt.xticks(np.linspace(0, 180, 10))
        plt.xlim([0,180])
        plt.tight_layout()
        plt.savefig('OH_angle_distribution.png', dpi=600, format='png')
        plt.close()

    def plot_dipole_orientation(self):
        """ Plots the density orientation profile. """
        if not self.rho_of_z.any():
            print('compute the density orientation profile first\n')
            return
        print('plotting')
        # plot
        plt.plot(self.z_step, self.rho_of_z)
        # save data
        with open('density_orientation.dat', 'w') as f:
            for i in range(len(self.z_step)):
                f.write(str(self.z_step[i]) + '\t' + str(self.rho_of_z[i]) + '\n')
        plt.xlabel('z (Å)')
        plt.ylabel('ρ$_{H_{2}O}$ cos$\\psi$ (g/cm$^3$)')
        #plt.title('H$_2$O density profile')
        plt.xlim([0.0, (self.z_top - self.z_bot)])
        plt.tight_layout()
        plt.savefig('density_orientation.png', dpi=600, format='png')
        plt.close()

A = 9.7410573959
B = 8.4360027313
C = 31.1839570560
dimensions = np.array([A, B, C])

if __name__ == '__main__':

    start_time = time.time()

    bottom_interface = 4.59198
    top_interface = 26.59198
    start_timestep = 1
    stop_timestep = 10

    traj = Trajectory(
        'XDATCAR',
        'vasp', # specify if 'xyz' format or 'vasp' format
        1,
        bottom_interface,
        top_interface,
        start_timestep,
        stop_timestep,
        resolution=221,
        d_angle=1,
        z_range_min=0.0,
        #z_range_max=11.0,
        #z_range_max=2.75,
        #z_range_min=2.75,
        z_range_max=3.50,
        x_min=0.0, x_max=A,
        y_min=0.0, y_max=B,
        cuboid_resolution=221,
        center_atom_list=[167,],
        #center_atom_list=[25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228],
        #r=5.0,
        r=6.0,
        #r=4.5,
        spherical_resolution=51,
    )

    calc_type = input("""
    Select plot type:
    1. H2O density distribution, ρ(H2O)
    2. O density distribution, ρ(O)
    3. H density distribution, ρ(H)
    4. Dipole orientation distribution vs angle
    5. O-H distribution vs angle
    6. Dipole orientation distribution vs z distance
    7. Get average cos(dipole orientation) within z region
    8. H2O number density distribution in spherical region across individual atoms
    9. Dipole angle distribution in spherical region across individual atoms
    10. H2O number density distribution in cuboidal region
    11. Dipole orientation distribution vs z distance in cuboidal region
    12. F density distribution, ρ(F)
    13. Na density distribution, ρ(Na)
    """)

    if calc_type == '1':
        print("computing...")
        traj.compute_density_profile("H2O")
        traj.plot_density_profile("H2O")
    elif calc_type == '2':
        print("computing...")
        traj.compute_density_profile("O")
        traj.plot_density_profile("O")
    elif calc_type == '3':
        print("computing...")
        traj.compute_density_profile("H")
        traj.plot_density_profile("H")
    elif calc_type == '4':
        print("Make sure z_range_min and z_range_max are set correctly")
        print("computing...")
        traj.compute_angle_distribution()
        traj.plot_dipole_distribution()
        traj.plot_OH_angle_distribution()
    elif calc_type == '5':
        print("Make sure z_range_min and z_range_max are set correctly")
        print("computing...")
        traj.compute_angle_distribution()
        traj.plot_OH_angle_distribution()
    elif calc_type == '6':
        print("computing...")
        traj.compute_dipole_orientation()
        traj.plot_dipole_orientation()
    elif calc_type == '7':
        print("Make sure z_range_min and z_range_max are set correctly")
        print("computing...")
        traj.compute_avg_cos_dipole()
    elif calc_type == '8':
        print("computing...")
        traj.compute_sphere_density_profile("H2O")
        traj.plot_sphere_density_profile("H2O")
    elif calc_type == '9':
        print("computing...")
        traj.compute_sphere_angle_distribution("H2O")
        traj.plot_sphere_angle_distribution("H2O")
    elif calc_type == '10':
        print("computing...")
        #traj = Trajectory(options)
        traj.compute_cuboid_density_profile("H2O")
        traj.plot_cuboid_density_profile("H2O")
    elif calc_type == '11':
        print("computing...")
        traj.compute_cuboid_dipole_orientation()
        traj.plot_cuboid_dipole_orientation()
    elif calc_type == '12':
        print("computing...")
        traj.compute_density_profile("F")
        traj.plot_density_profile("F")
    elif calc_type == '13':
        print("computing...")
        traj.compute_density_profile("Na")
        traj.plot_density_profile("Na")
    else:
        print("Please choose from one of the options")

    print("--- %s seconds ---" % (time.time() - start_time))
