import glob


def get_energy(filename):
    f = open(filename, "r")
    lines = f.readlines()
    containing_e = []
    for line in lines:
        if "TOTEN" in line:
            containing_e.append(line)
    energy = containing_e[-1].split()[-2]
    print(filename + ":\t" + energy)
    f.close()
    return


filenames = glob.glob("./*/dos/OUTCAR")
for filename in filenames:
    get_energy(filename)
