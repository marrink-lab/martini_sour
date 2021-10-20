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
from pathlib import Path


def get_ph_values(ph_vals):
    """
    Parameters
    ----------
    ph_vals : list of strings
        Each element is either a single value or a range with increments (e.g., 3:8:0.5)

    Returns
    -------
    all_vals : list of strings
        Ordered unique pH values with 2 decimal precision

    """
    all_vals = []
    for ph_val in ph_vals:
        if ':' in ph_val:
            init, final, increment = [float(i) for i in ph_val.split(':')]
            vals = np.arange(init, final+1e-3, increment)
            all_vals.extend(vals)
        else:
            all_vals.append(float(ph_val))

    all_vals = sorted(list(set(all_vals)))
    all_vals = [f'{val:.2f}' for val in all_vals]

    return all_vals


def get_sim_names(mdp_files):
    sims = []
    for mdp_file in mdp_files:
        if mdp_file.suffix != '.mdp':
            raise NameError('Provided mdp file(s) does not have the ".mdp" extension.')
        sims.append(mdp_file.stem)
    return sims


def update_top_file(file_name, top_file, ph, prefix):
    """
    Prepend the pH value to the beginning of the topology file and adjust the directory
    of included ITP files.
    """

    new_top_dir = Path(f'{prefix}{ph}') / file_name
    new_top_file = f'#define pH{ph}\n\n'

    top_file = top_file.split('\n')
    for line in top_file:
        if '#include' in line.replace(" ", ""):
            path = Path(line.replace('"', "'").split("'")[1])
            new_path = '..' / path
            line = line.replace(str(path), str(new_path))
        new_top_file += f'{line}\n'

    new_top_dir.write_text(new_top_file)


def make_new_dirs(sims, ph, prefix):
    for sim in sims:
        Path(f'{prefix}{ph}/{sim}').mkdir(parents=True, exist_ok=True)


def create_bash_script(sims, mdp_files, top_file, gro_file, out_file, ph_vals, prefix):
    """
    Create a bash script for running all the simulations necessary for the titration.

    Parameters
    ----------
    sims : list of strings
    mdp_files : list of strings
    top_file : Path
    gro_file : Path
    out_file : Path
    ph_vals : list of strings
    prefix : string

    """
    with open(out_file, 'w') as file:
        file.write('#!/bin/bash\n\n')
        file.write('pHrange=(' + ' '.join(ph_vals) + ')\n\n')
        file.write('for pH in ${pHrange[*]}; do\n')
        for i, (sim, mdp_file) in enumerate(zip(sims, mdp_files)):

            if i == 0:
                cd = f'  cd {prefix}' + '${pH}/' + f'{sim}\n'
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
    top_file = args.top_file.read_text()

    for ph in ph_vals:
        make_new_dirs(sims, ph, args.prefix)
        update_top_file(args.top_file, top_file, ph, args.prefix)

    create_bash_script(sims, args.mdp_file, args.top_file, args.gro_file, args.out_file, ph_vals,
                       args.prefix)
