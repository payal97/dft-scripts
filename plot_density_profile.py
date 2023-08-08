import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["font.size"] = 12


"""
A, B, C:        Lattice parameters (in Angstrom)
coords:         List of coordinates for all atoms.
atom_list:      Labels for all atoms defined in 'coords'.
data_oxygen:    Coordinates of all oxygen atoms in electrolyte.
data_hydrogen:  Coordinates of all hydrogen atoms in electrolyte.
data_water_molecule:  List of list of coordinates in each water molecule unit.
data_com:       Coordinates of center of mass for each water molecule unit.
"""

def distance(a, b):
    """ get displacement in each coordinate and wrap w.r.t. lattice
    parameter """
    dx = abs(a[0] - b[0])
    x = min(dx, abs(A - dx))
     
    dy = abs(a[1] - b[1])
    y = min(dy, abs(B - dy))
     
    dz = abs(a[2] - b[2])
    z = min(dz, abs(C - dz))
 
    return np.lib.scimath.sqrt(x**2 + y**2 + z**2)

def get_water_molecules_and_com(coords, atom_list, z_bot, z_top):
    """ Get H2O molecules and compute center of mass for each molecule. """
    max_O_H_length = 1.2 # angstrom

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
            dist = distance(oxygen, hydrogen)
            if dist <= max_O_H_length:
                data_water_molecule[-1].append(hydrogen)

    # final list of water molecules with length = 3
    data_coords = []
    for mol in data_water_molecule:
        if len(mol) != 3:
            #print("Hydronium or hydroxide ion detected.")
            #print("Change max_O_H_length in line 37 and try again.")
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

    data_com = np.array(data)
    return data, data_com


class Trajectory:
    def __init__(self, filename, file_type, skip, z_bot_interface, z_top_interface,
                 start_timestep, stop_timestep, interface_offset=0.0,
                 resolution=200, d_angle=5, z_range_min=0.0, z_range_max=5.0):
        """
        filename         : Path to the trajectory file.
        file_type        : xyz or vasp
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
        # z_range_max = self.z_top - self.z_bot
        self.z_range = np.array([z_range_min, z_range_max])

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

    def compute_density_profile(self, atom_type):
        # Define step size, dz.
        self.dz = (self.z_top - self.z_bot) / self.resolution
        # Divide electrolyte length into units of dz.
        self.z_step = np.linspace(0.0, self.resolution * self.dz, self.resolution)
        
        # Initialize list for storing density profile, rho_of_z.
        self.rho_of_z = np.zeros(self.resolution)

        for step in range(self.n_steps):

            if atom_type == "H2O":
                # Get water molecules and centers of mass.
                data_coords, data_com = get_water_molecules_and_com(
                    self.coordinates[step], self.atom_list, self.z_bot, self.z_top)
            elif atom_type in self.atoms:
                # Get atoms of type atom_type. E.g., "O", "H", etc.
                data_coords, data_com = get_atoms_pos(
                    atom_type,
                    self.coordinates[step], self.atom_list, self.z_bot, self.z_top)
            else:
                print("Incorrect atom_type. Please choose one of 'H2O' or", self.atoms)
                return

            # Sweep over intervals of dz width from z_bot to z_top and
            #   count all molecules/atoms that fall in the region at each step.
            for i in range(self.resolution):
                z1 = self.z_bot + (i * self.dz)
                z2 = z1 + self.dz
                for center_of_mass in data_com:
                    if z1 < center_of_mass[2] <= z2:
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
        print(num_atoms)

        # Convert to units of g/cm^3.
        if atom_type == "H2O":
            mass = 16+1+1
        elif atom_type == "O":
            mass = 16
        elif atom_type == "H":
            mass = 1
        else:
            print("Mass of atom", atom_type, "not defined. Please edit the script to include it.")
            #self.rho_of_z = self.rho_of_z / num_atoms
            return
        volume_box = A * B * self.dz * 10**(-24)
        self.rho_of_z = (self.rho_of_z * mass) / ((6.02214076 * 10**23) * volume_box)
        # Print total density, in units of g/cm^3.
        print('Density:', (sum(self.rho_of_z) * self.dz) / (self.z_top - self.z_bot), 'g/cm3')
        # Normalize for number of molecules/atoms.
        #self.rho_of_z = self.rho_of_z / num_atoms

    def compute_angle_distribution(self):
        angle_resolution = int(180/self.d_angle)
        self.angle_step = np.linspace(0, 180, angle_resolution)

        # psi: Angle between dipole (bisector of H-O-H) and surface normal.
        # theta: Angle between O-H bond and surface normal.
        self.psi_dist = np.zeros(angle_resolution)
        self.theta_dist = np.zeros(angle_resolution)

        for step in range(self.n_steps):

            # Get water molecules and centers of mass.
            data_water_molecule, data_com = get_water_molecules_and_com(
                    self.coordinates[step], self.atom_list, self.z_bot, self.z_top)

            for water in data_water_molecule:
                if len(water) == 3:
                    # Get O-H unit vectors.
                    vec1 = (
                        (water[1]-water[0]) / np.linalg.norm(water[1]-water[0]))
                    vec2 = (
                        (water[2]-water[0]) / np.linalg.norm(water[2]-water[0]))

                    # Compute psi, dipole-normal angle.
                    bisector = (vec1 + vec2) / np.linalg.norm(vec1 + vec2)
                    # https://stackoverflow.com/a/13849249
                    # Convert radians to degrees.
                    psi = np.rad2deg(np.arccos(np.dot(bisector, self.surface_normal)))
                    # Iterate over all molecules and get density profile.
                    index = int(psi / self.d_angle)
                    if 0 < index < angle_resolution:
                        if self.z_range[0] < water[0][2] - self.z_bot <= self.z_range[1]:
                            self.psi_dist[index] += 1.0
                        elif self.z_range[0] < self.z_top - water[0][2] <= self.z_range[1]:
                            self.psi_dist[index] += 1.0

                    # Compute theta, OH-normal angle.
                    # https://stackoverflow.com/a/13849249
                    theta = np.rad2deg(np.arccos(np.dot(vec1, self.surface_normal)))
                    # Iterate over all molecules and get density profile.
                    index = int(theta / self.d_angle)
                    if 0 < index < angle_resolution:
                        if self.z_range[0] < water[0][2] - self.z_bot <= self.z_range[1]:
                            self.theta_dist[index] += 1.0
                        elif self.z_range[0] < self.z_top - water[0][2] <= self.z_range[1]:
                            self.theta_dist[index] += 1.0

                    theta = np.rad2deg(np.arccos(np.dot(vec2, self.surface_normal)))
                    # Iterate over all molecules and get density profile.
                    index = int(theta / self.d_angle)
                    if 0 < index < angle_resolution:
                        if self.z_range[0] < water[0][2] - self.z_bot <= self.z_range[1]:
                            self.theta_dist[index] += 1.0
                        elif self.z_range[0] < self.z_top - water[0][2] <= self.z_range[1]:
                            self.theta_dist[index] += 1.0

        # Get total number of entries.
        #num_psi = sum(self.psi_dist)
        #num_theta = sum(self.theta_dist)

        # Normalize for number of steps and number of entries.
        #self.psi_dist = self.psi_dist / (self.n_steps * num_psi)
        #self.theta_dist = self.theta_dist / (self.n_steps * num_theta)
        self.psi_dist = self.psi_dist / self.n_steps
        self.theta_dist = self.theta_dist / self.n_steps

    def compute_avg_cos_dipole(self):
        cos_psi_all_steps = 0

        for step in range(self.n_steps):

            # Get water molecules and centers of mass from both ends of the electrode.
            data_water_molecule_bot, data_com_bot = get_water_molecules_and_com(
                    self.coordinates[step], self.atom_list, self.z_bot + self.z_range[0], self.z_bot + self.z_range[1])
            data_water_molecule_top, data_com_top = get_water_molecules_and_com(
                    self.coordinates[step], self.atom_list, self.z_top - self.z_range[1], self.z_top - self.z_range[0])

            data_water_molecule = np.vstack([data_water_molecule_bot,data_water_molecule_top])
            data_com = np.vstack([data_com_bot,data_com_top])

            cos_psi = np.full(len(data_water_molecule), -2.0)

            for k in range(len(data_water_molecule)):
                water = data_water_molecule[k]
                if len(water) == 3:
                    # Get O-H unit vectors.
                    vec1 = (
                        (water[1]-water[0]) / np.linalg.norm(water[1]-water[0]))
                    vec2 = (
                        (water[2]-water[0]) / np.linalg.norm(water[2]-water[0]))

                    # Compute psi, dipole-normal angle.
                    bisector = (vec1 + vec2) / np.linalg.norm(vec1 + vec2)
                    # https://stackoverflow.com/a/13849249
                    # Get psi in radians.
                    psi = np.arccos(np.dot(bisector, self.surface_normal))
                    cos_psi[k] = np.cos(psi)

            # Sanity check, to ensure psi from all waters were counted.
            if -2 in cos_psi:
                print("cos(psi) for some H2O were not populated")

            cos_psi_all_steps += np.mean(cos_psi)

        # Normalize for number of steps.
        cos_psi_all_steps = cos_psi_all_steps / self.n_steps
        print('cos_psi_all_steps:', cos_psi_all_steps)

    def compute_dipole_orientation(self):
        # Define step size, dz.
        self.dz = (self.z_top - self.z_bot) / self.resolution
        # Divide electrolyte length into units of dz.
        self.z_step = np.linspace(0.0, self.resolution * self.dz, self.resolution)
        
        # Initialize list for storing density profile, rho_of_z.
        self.rho_of_z = np.zeros(self.resolution)

        for step in range(self.n_steps):

            # Get water molecules and centers of mass.
            data_water_molecule, data_com = get_water_molecules_and_com(
                    self.coordinates[step], self.atom_list, self.z_bot, self.z_top)

            # Sweep over intervals of dz width from z_bot to z_top.
            for i in range(self.resolution):
                z1 = self.z_bot + (i * self.dz)
                z2 = z1 + self.dz
                # Account for same region from opposite surface.
                z2opp = self.z_top - (i * self.dz)
                z1opp = z2opp - self.dz

                water_in_region = []
                for j in range(len(data_com)):
                    if z1 < data_com[j][2] <= z2 and len(data_water_molecule[j]) == 3:
                        water_in_region.append(data_water_molecule[j])
                    elif z1opp <= data_com[j][2] < z2opp and len(data_water_molecule[j]) == 3:
                        water_in_region.append(data_water_molecule[j])

                if len(water_in_region) > 0:

                    cos_psi = np.full(len(water_in_region), -2.0)

                    for k in range(len(water_in_region)):
                        water = water_in_region[k]
                        if len(water) == 3:
                            # Get O-H unit vectors.
                            vec1 = (
                                (water[1]-water[0]) / np.linalg.norm(water[1]-water[0]))
                            vec2 = (
                                (water[2]-water[0]) / np.linalg.norm(water[2]-water[0]))

                            # Compute psi, dipole-normal angle.
                            bisector = (vec1 + vec2) / np.linalg.norm(vec1 + vec2)
                            # https://stackoverflow.com/a/13849249
                            # Get psi in radians.
                            psi = np.arccos(np.dot(bisector, self.surface_normal))
                            cos_psi[k] = np.cos(psi)

                    # Sanity check, to ensure psi from all waters were counted.
                    if -2 in cos_psi:
                        print("cos(psi) for some H2O in region", z1, z2, z1opp, z2opp, "were not populated")

                    # Get density * avg(cos(psi)), in g/cm3.
                    volume_box = A * B * self.dz * 10**(-24)
                    density = (len(water_in_region) * (16+1+1)) / ((6.02214076 * 10**23) * volume_box)
                    avg_cos_psi = np.mean(cos_psi)
                    self.rho_of_z[i] += density * avg_cos_psi

        # Normalize for number of steps.
        self.rho_of_z = self.rho_of_z / self.n_steps

    def plot_density_profile(self, atom_type):
        """ Plots the density profile. """

        #plt.rcParams["figure.figsize"] = (6,4)
         
        if not self.rho_of_z.any():
            print('compute the density profile first\n')
            return
         
        bulk_index_bot = int(len(self.z_step) * 9 / (self.z_top - self.z_bot))
        bulk_index_top = int(len(self.z_step) * (self.z_top - self.z_bot - 9) / (self.z_top - self.z_bot))
        print(bulk_index_bot, bulk_index_top)
        bulk_density = self.rho_of_z[bulk_index_bot:bulk_index_top+1]
        print("average bulk density (>= 9\\A from electrode):", np.mean(bulk_density))
 
        print('plotting density profile')
        # average from both electrode surfaces
        sum_rho = (self.rho_of_z + self.rho_of_z[::-1])/2
        # plot
        plt.plot(self.z_step,sum_rho)
        # save data
        with open('density_profile.dat', 'w') as f:
            for i in range(len(self.z_step)):
                f.write(str(self.z_step[i]) + '\t' + str(sum_rho[i]) + '\n')
        # To plot a smoother fit of the data, uncomment 2 lines.
        #f_savgol = savgol_filter(sum_rho, 40, 3)
        #plt.plot(self.z_step, f_savgol)
        #plt.plot([0,2.47], [0,0], color='tab:blue')

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
        plt.xlim([0.0, (self.z_top - self.z_bot) / 2])
        #plt.ylim([0,4])

        plt.tight_layout()
        plt.savefig('density_profile.png', dpi=600, format='png')
        #plt.show()

    def plot_dipole_distribution(self):
        """ Plots the distribution of dipole-normal angles (psi). """

        #plt.rcParams["figure.figsize"] = (6, 4)

        if not self.psi_dist.any():
            print('compute angle distributions first\n')
            return
         
        avg_dipole = np.sum(self.angle_step * self.psi_dist) / np.sum(self.psi_dist)
        print("average dipole moment:", avg_dipole)

        print('plotting dipole distribution')
        plt.plot(self.angle_step, self.psi_dist)
        with open('dipole_data_layer1+2.dat', 'w') as f:
            for i in range(len(self.angle_step)):
                f.write(str(self.angle_step[i]) + '\t' + str(self.psi_dist[i]) + '\n')
        # To plot a smoother fit of the data, uncomment 2 lines.
        #f_psi = savgol_filter(self.psi_dist, 40, 3)
        #plt.plot(self.angle_step, f_psi)

        plt.xlabel('Angle from surface normal (degree)')
        plt.ylabel('Probability density')
        plt.title('Dipole angle distribution')
        plt.xticks(np.linspace(0, 180, 10))
        plt.xlim([0,180])
        #plt.ylim=[0,0.04],

        plt.tight_layout()
        plt.savefig('dipole_distribution_layer1+2.png', dpi=600, format='png')
        #plt.show()

    def plot_OH_angle_distribution(self):
        """ Plots the distribution of OH-normal angles (theta). """

        plt.rcParams["figure.figsize"] = (6, 4)

        if not self.theta_dist.any():
            print('compute angle distributions first\n')
            return

        avg_angle = np.sum(self.angle_step * self.theta_dist) / np.sum(self.theta_dist)
        print("average O-H angle:", avg_angle)
         
        print('plotting O-H distribution')
        plt.plot(self.angle_step, self.theta_dist)
        with open('O-H_angle_layer1+2.dat', 'w') as f:
            for i in range(len(self.angle_step)):
                f.write(str(self.angle_step[i]) + '\t' + str(self.theta_dist[i]) + '\n')
        # To plot a smoother fit of the data, uncomment 2 lines.
        #f_theta = savgol_filter(self.theta_dist, 40, 3)
        #ax.plot(self.angle_step, f_theta)

        plt.xlabel('Angle from surface normal (degree)')
        plt.ylabel('Probability density')
        plt.title('O-H angle distribution')
        plt.xticks(np.linspace(0, 180, 10))
        plt.xlim([0,180])
        #plt.ylim=[0,0.04],

        plt.tight_layout()
        plt.savefig('OH_angle_dist_layer1+2.png', dpi=600, format='png')
        #plt.show()

    def plot_dipole_orientation(self):
        """ Plots the density orientation profile. """

        #plt.rcParams["figure.figsize"] = (6,4)
         
        if not self.rho_of_z.any():
            print('compute the density orientation profile first\n')
            return
 
        print('plotting')
        # plot
        plt.plot(self.z_step,self.rho_of_z)
        # save data
        with open('density_orientation.dat', 'w') as f:
            for i in range(len(self.z_step)):
                f.write(str(self.z_step[i]) + '\t' + str(self.rho_of_z[i]) + '\n')
        # To plot a smoother fit of the data, uncomment 2 lines.
        #f_savgol = savgol_filter(self.rho_of_z, 40, 3)
        #plt.plot(self.z_step, f_savgol)
        #plt.plot([0,2.47], [0,0], color='tab:blue')

        plt.xlabel('z (Å)')
        plt.ylabel('ρ$_{H_{2}O}$ cos$\\psi$ (g/cm$^3$)')
        plt.title('H$_2$O density profile')
        plt.xlim([0.0, (self.z_top - self.z_bot) / 2])
        #plt.ylim([0,4])

        plt.tight_layout()
        plt.savefig('density_orientation.png', dpi=600, format='png')
        #plt.show()


A = 9.573652267
B = 12.801160812
C = 37.915355682
bottom_interface = 10.78748 + 1
top_interface = 36.78938 - 1
start_timestep = 1
stop_timestep = 650

traj = Trajectory(
    'nvt.xyz',
    'xyz', # specify if 'xyz' format or 'vasp' format
    10, 
    bottom_interface, 
    top_interface,
    start_timestep,
    stop_timestep,
    resolution=240,
    d_angle=2,
    #z_range_min=0.80,
    #z_range_max=3.91,
    #z_range_max=6.53,
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
""")

if calc_type == '1':
    print("computing...")
    traj.compute_density_profile("H2O")
    traj.plot_density_profile("H2O")
elif calc_type == 2:
    print("computing...")
    traj.compute_density_profile("O")
    traj.plot_density_profile("O")
elif calc_type == 3:
    print("computing...")
    traj.compute_density_profile("H")
    traj.plot_density_profile("H")
elif calc_type == 4:
    print("Make sure z_range_min and z_range_max are set correctly")
    print("computing...")
    traj.compute_angle_distribution()
    traj.plot_dipole_orientation()
elif calc_type == 5:
    print("Make sure z_range_min and z_range_max are set correctly")
    print("computing...")
    traj.compute_angle_distribution()
    traj.plot_OH_angle_distribution()
elif calc_type == 6:
    print("computing...")
    traj.compute_dipole_orientation()
    traj.plot_dipole_distribution()
elif calc_type == 7:
    print("Make sure z_range_min and z_range_max are set correctly")
    print("computing...")
    traj.compute_avg_cos_dipole()
else:
    print("Please choose from one of the options")
