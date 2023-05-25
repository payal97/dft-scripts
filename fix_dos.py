import glob


all_dos_files = glob.glob("DOS*")
all_dos_files.remove('DOSCAR')


for filename in all_dos_files:
    # https://stackoverflow.com/a/4719562
    # read data
    with open(filename, 'r') as f:
        lines = f.readlines()
    # update 2nd line
    lines[1] = '# ' + lines[1]
    # write data
    with open(filename, 'w') as f:
        f.writelines(lines)


print('done!')
