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
data_water_molecule:  list of list of coordinates in each water molecule unit
data_com:       Coordinates of center of mass for each water molecule unit
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
    """ Get water molecules and compute center of mass for each molecule. """

    # Isolate all water molecules in electrolyte based on their oxygens.
    # Get all oxygens and hydrogens present in the electrolyte.
    data_oxygen = []
    data_hydrogen = []
    for i, atom in enumerate(coords):
        if atom_list[i] == "O":
            # check if atom lies in electrolyte region
            if z_bot < atom[2] < z_top:
                data_oxygen.append(atom)
        if atom_list[i] == "H":
            # check if atom lies in electrolyte region
            if z_bot < atom[2] < z_top:
                data_hydrogen.append(atom)
    data_oxygen = np.array(data_oxygen)
    data_hydrogen = np.array(data_hydrogen)

    # For each oxygen, get hydrogens within a raduis of 1.2 A and
    # save all the atoms as units of one water molecule.
    data_water_molecule = []
    data_com = []
    for oxygen in data_oxygen:
        # Add a water molecule.
        data_water_molecule.append([])
        data_water_molecule[-1].append(oxygen)

        # Find all hydrogens in the water molecule.
        for hydrogen in data_hydrogen:
            dist = distance(oxygen, hydrogen)
            if dist < 1.2:
                data_water_molecule[-1].append(hydrogen)

        # Compute center of mass for each water molecule.
        # Center of mass = sum [ position of atom * weight of atom ]
        molecule = data_water_molecule[-1]
        # Mass of oxygen = 16
        com_o = molecule[0] * 16
        # Mass of hydrogen = 3
        list_h = [i * 3 for i in molecule[1:]]
        com_h = sum(list_h)
        center_of_mass = (com_o + com_h) / (16 + (len(list_h) * 3))
        data_com.append(center_of_mass)
    data_water_molecule = np.array(data_water_molecule)
    data_com = np.array(data_com)
    return data_water_molecule, data_com


class Trajectory:
    def __init__(self, filename, skip, z_bot_interface, z_top_interface,
                 start_timestep, stop_timestep, interface_offset=0.0,
                 resolution=200, d_angle=5, z_range_min=0.0, z_range_max=5.0):
        """
        filename         : path to the trajectory file in axsf format
        skip             : number of snapshots to be skipped between two
                           configurations that are evaluated (for example,
                           if trajectory is 9000 steps long, and skip = 10,
                           every tenth step is evaluated, 900 steps in
                           total; use skip=1 to take every step of the MD)
        z_bot_interface  : average vertical coordinate for interface below
                           water layer in Angstrom
        z_top_interface  : average vertical coordinate for interface above
                           water layer in Angstrom
        interface_offset : distance between interface and region of water
                           with bulk-like properties
        resolution       : number of points in the final radial
                           distribution function 
        z_range_min      : the distibution of bond angles are calculated
                           in the volume between z_range_min and z_range_max
                           vertical coordinates
                           (w.r.t z_bot)
        z_range_max      : the distibution of bond angles are calculated
                           in the volume between z_range_min and z_range_max
                           vertical coordinates
                           (w.r.t z_bot)
                           """

        self.filename = filename
        self.skip = skip
        self.z_top = z_top_interface - interface_offset
        self.z_bot = z_bot_interface + interface_offset
        self.start_timestep = start_timestep
        self.stop_timestep = stop_timestep
        self.resolution = resolution

        self.surface_normal = np.array([0, 0, 1])
        self.d_angle = d_angle
        self.z_range = np.array([z_range_min, z_range_max])

        self.parse_input()

    def parse_input(self):
        with open(self.filename, 'r') as f:
            data = f.readlines()

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

    def compute_density_profile(self):
        # Define step size, dz.
        self.dz = (self.z_top - self.z_bot) / self.resolution
        # Divide electrolyte length into units of dz.
        self.z_step = np.linspace(0.0, self.resolution * self.dz, self.resolution)
        
        # Initialize list for storing density profile, rho_of_z.
        self.rho_of_z = np.zeros(self.resolution)

        for step in range(self.n_steps):
            # print('{:4d} : {:4d}'.format(step, self.n_steps))
             
            # Get water molecules and centers of mass.
            data_water_molecule, data_com = get_water_molecules_and_com(
                self.coordinates[step], self.atom_list, self.z_bot, self.z_top)

            # Sweep over intervals of dz width from z_bot to z_top and
            # count all water molecules that fall in the region at each step.
            for i in range(self.resolution):
                z1 = self.z_bot + (i * self.dz)
                z2 = z1 + self.dz
                for center_of_mass in data_com:
                    if z1 < center_of_mass[2] <= z2:
                        self.rho_of_z[i] += 1

        # Normalize for number of steps.
        self.rho_of_z = self.rho_of_z / self.n_steps
        print(sum(self.rho_of_z))
        # Convert to units of g/cm^3.
        volume_box = A * B * self.dz * 10**(-24)
        self.rho_of_z = (self.rho_of_z * (16 + 3 + 3)) / ((6.02214076 * 10**23) * volume_box)
        # Print total water density, in units of g/cm^3.
        print('Density of water:', (sum(self.rho_of_z) * self.dz) / (self.z_top - self.z_bot), 'g/cm3')

    def compute_angle_distribution(self):
        angle_resolution = int(180/self.d_angle)
        self.angle_step = np.linspace(0, 180, angle_resolution)

        # psi: Angle between dipole (bisector of H-O-H) and surface normal
        # theta: Angle between O-H bond and surface normal
        self.psi_dist = np.zeros(angle_resolution)
        self.theta_dist = np.zeros(angle_resolution)

        for step in range(self.n_steps):

            # Get water molecules and centers of mass.
            data_water_molecule, data_com = get_water_molecules_and_com(
                    self.coordinates[step], self.atom_list, self.z_bot, self.z_top)

            for water in data_water_molecule:
                #try:
                # Get O-H unit vectors.
                if len(water) == 3:
                    vec1 = (
                        (water[1]-water[0]) / np.linalg.norm(water[1]-water[0]))
                    vec2 = (
                        (water[2]-water[0]) / np.linalg.norm(water[2]-water[0]))

                    # Compute psi, dipole-normal angle
                    bisector = (vec1 + vec2) / np.linalg.norm(vec1 + vec2)
                    # https://stackoverflow.com/a/13849249
                    # Convert radians to degrees.
                    psi = np.rad2deg(np.arccos(np.dot(bisector, self.surface_normal)))
                    # Iterate over all molecules and get density profile
                    index = int(psi / self.d_angle)
                    if 0 < index < angle_resolution:
                        if self.z_range[0] < water[0][2] - self.z_bot <= self.z_range[1]:
                            self.psi_dist[index] += 1.0

                    # Compute theta, OH-normal angle
                    # https://stackoverflow.com/a/13849249
                    theta = np.rad2deg(np.arccos(np.dot(vec1, self.surface_normal)))
                    # Iterate over all molecules and get density profile
                    index = int(theta / self.d_angle)
                    if 0 < index < angle_resolution:
                        if self.z_range[0] < water[0][2] - self.z_bot <= self.z_range[1]:
                            self.theta_dist[index] += 1.0

                    theta = np.rad2deg(np.arccos(np.dot(vec2, self.surface_normal)))
                    # Iterate over all molecules and get density profile
                    index = int(theta / self.d_angle)
                    if 0 < index < angle_resolution:
                        if self.z_range[0] < water[0][2] - self.z_bot <= self.z_range[1]:
                            self.theta_dist[index] += 1.0
                #except:
                #    pass

        # Normalize for number of steps.
        self.psi_dist = self.psi_dist / self.n_steps
        self.theta_dist = self.theta_dist / self.n_steps

    def plot_density_profile(self):
        """ Plots the density profile. """

        #plt.rcParams["figure.figsize"] = (6,4)
         
        if not self.rho_of_z.any():
            print('compute the density profile first\n')
            return
         
        print('plotting density profile')
        plt.plot(self.z_step, self.rho_of_z)
        # To plot a smoother fit of the data, uncomment 2 lines.
        #f_savgol = savgol_filter(self.rho_of_z, 40, 3)
        #plt.plot(self.z_step, f_savgol)
        plt.plot([0,2.47], [0,0], color='tab:blue')

        plt.xlabel('z (Å)')
        plt.ylabel('ρ$_{H_{2}O}$ (g/cm$^3$)')
        plt.xlim([0.0, (self.z_top - self.z_bot) / 2])
        plt.ylim([0,4])

        plt.tight_layout()
        plt.savefig('density_profile.png', dpi=600, format='png')
        #plt.show()

    def plot_dipole_distribution(self):
        """ Plots the distribution of dipole-normal angles (psi). """

        plt.rcParams["figure.figsize"] = (6, 4)

        if not self.psi_dist.any():
            print('compute angle distributions first\n')
            return
         
        print('plotting density profile')
        plt.plot(self.angle_step, self.psi_dist)
        # To plot a smoother fit of the data, uncomment 2 lines.
        #f_psi = savgol_filter(self.psi_dist, 40, 3)
        #ax.plot(self.angle_step, f_psi)

        plt.set(
                xlabel='ψ (degree)',
                ylabel='Probability density',
                xticks=np.linspace(0, 180, 10),
                xlim=[0,180],
                #ylim=[0,0.04],
        )

        plt.tight_layout()
        plt.savefig('dipole_distribution.png', dpi=600, format='png')
        #plt.show()

    def plot_OH_angle_distribution(self):
        """ Plots the distribution of OH-normal angles (theta). """

        plt.rcParams["figure.figsize"] = (6, 4)

        if not self.theta_dist.any():
            print('compute angle distributions first\n')
            return
         
        print('plotting density profile')
        plt.plot(self.angle_step, self.theta_dist)
        # To plot a smoother fit of the data, uncomment 2 lines.
        #f_theta = savgol_filter(self.theta_dist, 40, 3)
        #ax.plot(self.angle_step, f_theta)

        plt.set(
                xlabel='θ (degree)',
                ylabel='Probability density',
                xticks=np.linspace(0, 180, 10),
                xlim=[0,180],
                #ylim=[0,0.04],
        )

        plt.tight_layout()
        plt.savefig('OH_angle_distribution.png', dpi=600, format='png')
        #plt.show()


A = 12.8502998351999995
B = 9.4194002150999996
C = 22.00
bottom_interface = 11.8
top_interface = 19.9
start_timestep = 2000
stop_timestep = 4000

traj = Trajectory(
    'XDATCAR',
    1, 
    bottom_interface, 
    top_interface,
    start_timestep,
    stop_timestep,
    resolution=500,
    d_angle=1,
)
traj.compute_density_profile()
traj.plot_density_profile()
#traj.compute_angle_distribution()
#traj.plot_dipole_distribution()
#traj.plot_OH_angle_distribution()
