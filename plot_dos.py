import matplotlib.pyplot as plt
import numpy as np


# Set figure size
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 15

fig, axs = plt.subplots() # 4, 4, sharex=True, sharey=True)


rootdir = './'

def get_data(filepath): # , fermi):
    '''Extract dos data from file'''
    # read dos file
    f = open(rootdir + filepath, 'r')
    lines = f.readlines()
    # delete first two lines, which do not contain dos data
    del lines[0:3]
    # convert each line of the dos file to a list of floats
    lines = [ list(map(float, line.split())) for line in lines ]
    f.close()
    lines = np.array(lines)
    return lines


# get dos data
#title = input("Enter title:  ")
#atom = input("Enter atom label:  ")
#atom_number = input("Enter atom number:  ")


def plot_states(atom_number, atom_label, plot_s_states, plot_p_states, plot_d_states):
    data = get_data('DOS' + atom_number)
    # plot pdos of s orbitals
    if 'y' in plot_s_states:
        axs.plot(data[:,0], data[:,1], color='tab:green', label="{0} $s$".format(atom_label))
    # plot pdos of p orbitals
    if 'y' in plot_p_states:
        data_p = data[:,2] + data[:,3] + data[:,4]
        #axs.plot(data[:,0], data_p, color='tab:red', label="{0} $p$".format(atom_label))
        axs.fill_between(data[:,0], data_p, alpha=0.5, lw=0.0, color='tab:red', label="{0} $p$".format(atom_label))
        #axs.fill_between(data[:,0], data[:,2], alpha=0.5, label="{0} $p_y$".format(atom_label))
        #axs.fill_between(data[:,0], data[:,3], alpha=0.5, label="{0} $p_z$".format(atom_label))
        #axs.fill_between(data[:,0], data[:,4], alpha=0.5, label="{0} $p_x$".format(atom_label))
    # plot pdos of d orbitals
    if 'y' in plot_d_states:
        data_d = data[:,5] + data[:,6] + data[:,7] + data[:,8] + data[:,9]
        axs.plot(data[:,0], data_d, color='tab:blue', label="{0} $d$".format(atom_label))
        #axs.plot(data[:,0], data[:,5], label="{0} $d_{{xy}}$".format(atom_label))
        #axs.plot(data[:,0], data[:,6], label="{0} $d_{{yz}}$".format(atom_label))
        #axs.plot(data[:,0], data[:,7], label="{0} $d_{{z^2-r^2}}$".format(atom_label))
        #axs.plot(data[:,0], data[:,8], label="{0} $d_{{xz}}$".format(atom_label))
        #axs.plot(data[:,0], data[:,9], label="{0} $d_{{x^2-y^2}}$".format(atom_label))

title = "IrO$_2$(101)"
#plot_states(atom_number, atom, 'n', 'n', 'y')
plot_states("160-reg101-O-p", "O", 'n', 'y', 'n')
plot_states("36-reg-101-Ir-d", "Ir", 'n', 'n', 'y')


# plot properties 
axs.axvline(ls='--', lw=1, c='black')
axs.axhline(lw=1, c='black')
axs.set_xlim(-5,2)
axs.set_ylim(0,10)
axs.set_xlabel('E - E$\mathrm{_F}$, eV')
axs.set_ylabel('Density of states')
axs.set_title(title)
axs.legend()


fig.tight_layout(pad=2.0)
plt.savefig("{0}.png".format(title.replace(" ", "_")), dpi=300, bbox_inches='tight')
#plt.show()
