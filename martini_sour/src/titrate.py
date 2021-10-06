# Copyright 2021 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os


def get_ph_values(ph_range):
    init, final, increment = [float(i) for i in ph_range.split(':')]
    vals = np.arange(init, final+1e-3, increment)
    vals = [f'{val:.1f}' for val in vals]
    return vals


def get_sim_names(mdp_files):
    sims = []
    for mdp_file in mdp_files:
        if not mdp_file.endswith('.mdp'):
            raise NameError('Provided mdp file(s) does not have the ".mdp" extension.')
        sims.append(mdp_file.split('.mdp')[0].split('/')[-1])
    return sims


def read_top_file(file_name):
    with open(file_name) as file:
        top_file = file.readlines()
    return top_file


def update_top_file(file_name, top_file, ph):
    new_top_file = [f'#define pH{ph}\n', '\n'] + top_file
    with open(f'{ph}/{file_name}', 'w') as file:
        for line in new_top_file:
            file.write(line)


def make_new_dirs(sims, ph):
    os.makedirs(str(ph), exist_ok=True)
    for sim in sims:
        os.makedirs(f'{ph}/{sim}', exist_ok=True)


def create_bash_script(sims, mdp_files, top_file, gro_file, out_file, ph_vals):
    with open(out_file, 'w') as file:
        file.write('#!/bin/bash\n\n')
        file.write('pHrange=(' + ' '.join(ph_vals) + ')\n\n')
        file.write('for pH in ${pHrange[*]}; do\n')
        for i, (sim, mdp_file) in enumerate(zip(sims, mdp_files)):

            if i == 0:
                cd = '  cd ${ph}/' + f'{sim}\n'
                gro_file = f'../../{gro_file}'
            else:
                cd = f'  cd ../{sim}\n'
                gro_file = f'../{sims[i-1]}/confout.gro'

            file.write(cd)
            file.write(f'  gmx grompp -f ../../{mdp_file} -c {gro_file} -p ../{top_file}\n')
            file.write('  gmx mdrun\n\n')

            if i == len(sims)-1:
                file.write('  cd ../..\ndone\n')


def titrate(args):
    ph_vals = get_ph_values(args.ph)
    sims = get_sim_names(args.mdp_file)
    top_file = read_top_file(args.top_file)

    for ph in ph_vals:
        make_new_dirs(sims, ph)
        update_top_file(args.top_file, top_file, ph)

    create_bash_script(sims, args.mdp_file, args.top_file, args.gro_file, args.out_file, ph_vals)
