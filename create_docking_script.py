import os

# Creates bash script for running all autodock-vina simulations

#names = ['ATABECESTAT', 'VERUBECESTAT', 'LANABECESTAT', 'ELENBECESTAT']
folder = 'Compounds/'
names = ['MOL' + str(i+1) for i in range(100)]

# Prepare ligands
# Ligands should be in '.mol' format

script_ligand = open(folder + 'script_ligand.sh', 'w')
script_ligand.write('#!/bin/bash\n')

all_files = os.listdir(folder)
files = []
for f in all_files:
    if '.pdb' in f:
        for n in names:
            if n in f:
                files.append(f)
                break
print(len(files))

all_names = []
for f in files:
    prefix = f.split('.pdb')[0]
    name = prefix + '.pdbqt'
    script_ligand.write('obabel ' + f + ' -O ' + name + '\n')
    all_names.append(prefix)

script_ligand.close()

# Run vina
script_vina = open(folder + 'script_vina.sh', 'w')
script_vina.write('#!/bin/bash\n')
for name in all_names:
    script_vina.write('vina --receptor 2zht.pdbqt --ligand ' + name + '.pdbqt --config box_site.txt --exhaustiveness=128 --out Results/docking_' + name + '.pdb\n')
script_vina.close()