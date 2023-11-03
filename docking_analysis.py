import os
import statistics

#names = ['ATABECESTAT', 'VERUBECESTAT', 'LANABECESTAT', 'ELENBECESTAT']
names = ['MOL' + str(i+1) for i in range(100)]

for name in names:
    root_folder = 'Compounds/'
    folder = root_folder + 'Results/'
    prefix = 'docking_' + name + '_'
    all_files = os.listdir(folder)
    all_files = [f for f in all_files if prefix in f]

    if len(all_files) == 0:
        continue

    save_folder = root_folder + name + '/'
    os.mkdir(save_folder)

    # Get binding energies
    all_energies = []
    all_fnames = []
    all_models = []
    for f_name in all_files:
        f = open(folder + f_name, 'r')
        current_model = 0
        while True:
            line = f.readline()

            # End of file
            if line == '':
                break
            # New Model
            if 'MODEL' in line:
                current_model = int(line.split()[1])
            # Energy
            if 'REMARK VINA RESULT:' in line:
                vals = line.split()
                energy = float(vals[vals.index('RESULT:') + 1])
                all_energies.append(energy)
                all_fnames.append(f_name)
                all_models.append(current_model)
        f.close()

    # Sort binding energies
    sorted_vals = sorted(zip(all_energies, all_fnames, all_models), key=lambda pair: pair[0])

    # Save to file
    energy_output = open(save_folder + 'binding_energies.txt', 'w')
    energy_output.write('Min: ' + str(min(all_energies)) + '\n')
    energy_output.write('Avg: ' + str(statistics.mean(all_energies)) + '    Stdev: ' + str(statistics.stdev(all_energies)) + '\n')
    for v in sorted_vals:
        energy_output.write(v[1] + '  Model ' + str(v[2]) + ': ' + str(v[0]) + ' (kcal/mol)\n')
    energy_output.close()

    # Extract top poses
    num_poses = 20

    for i in range(num_poses):
        _, f_name, model = sorted_vals[i]
        m_name = 'MODEL ' + str(model)
        f = open(folder + f_name, 'r')
        output = open(save_folder + name + '_pose_' + str(i+1) + '.pdb', 'w')

        reading = False
        while True:
            line = f.readline()

            if reading:
                # Stop reading at end of file or next model
                if line == '' or 'MODEL' in line:
                    f.close()
                    output.close()
                    break
                else:
                    output.write(line)
            else:
                if m_name in line:
                    reading = True
                if line == '':
                    f.close()
                    output.close()
                    break
