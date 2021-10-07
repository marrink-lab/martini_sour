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
import MDAnalysis as mda
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .titrate import get_ph_values


def setup_dirs(ph_range, sim_dir, traj_file, tpr_file):
    """
    Sets up directory paths based on whether multidir options have been provided or not.
    """

    if ph_range and sim_dir:
        ph_values = get_ph_values(ph_range)
        traj_paths = [f'{ph}/{sim_dir}/{traj_file}' for ph in ph_values]
        tpr_paths = [f'{ph}/{sim_dir}/{tpr_file}' for ph in ph_values]
    else:
        traj_paths = [traj_file]
        tpr_paths = [tpr_file]

    return traj_paths, tpr_paths, np.array(ph_values, dtype=float)


def save_to_file(out_file, ph_values, props, sel_commands):
    array = np.concatenate((ph_values[np.newaxis], props.T), axis=0)

    fmt = ['%6.2f']+['%10.5f']*(len(array)-1)
    header = '  pH ' + ' '.join([f'sel. {i}'.rjust(10) for i in range(1, len(array))])

    np.savetxt(out_file, array.T, fmt=fmt, header=header)


def do_statistics(degrees_of_deprot):
    """
    This is probably where a more complicated statistical analysis would be inferfaced
    """

    mean = np.array(degrees_of_deprot).mean(axis=-1)
    std = np.array(degrees_of_deprot).std(axis=-1)

    return mean, std


def calc_degree_of_deprot(ph, pka, q):
    return 1/(10**(q*(pka-ph))+1)


def do_plotting(out_file, mean, ph_values):
    ph_range = np.arange(ph_values.min(), ph_values.max()+1e-3, 0.1)
    fig = plt.figure()

    for i, one_sel in enumerate(mean.T, start=1):

        fit = curve_fit(calc_degree_of_deprot, ph_values, one_sel, p0=(7.5, 0.75))[0]

        points, = plt.plot(ph_values, one_sel, '^',
                           label=f'Sel. {i}, pKa = {fit[0]:.2f}, q = {fit[1]:.2f}')
        plt.plot(ph_range, calc_degree_of_deprot(ph_range, *fit), color=points.get_color())

    plt.xlabel('pH')
    plt.ylabel('Degree of deprotonation')
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches='tight')


def do_one_trajectory(tpr_path, traj_path, sel_commands, ref_commands, start, end, scheme):
    u = mda.Universe(tpr_path, traj_path)

    sels = [u.select_atoms(sel_command) for sel_command in sel_commands]
    refs = [u.select_atoms(ref_command) for ref_command in ref_commands]
    prot = u.select_atoms("name POS")

    count = 0
    n_frames = len(u.trajectory[start:end])
    n_prot_matrix = np.zeros((len(sels), n_frames))
    time = np.zeros(n_frames)

    for ts in tqdm(u.trajectory[start:end]):
        time[count] = ts.time
        for i, (sel, ref) in enumerate(zip(sels, refs)):
            n_prot_matrix[i, count] = do_one_analysis(u, sel, ref, prot, scheme)
        count = count+1

    sel_sizes = np.array([len(sel) for sel in sels])
    degree_of_deprot = 1 - (n_prot_matrix.T/sel_sizes).T

    return degree_of_deprot


def do_one_analysis(u, sel, ref, prot, scheme):
    n_prots = 0
    for idx in sel.indices:

        curr = u.atoms[idx]
        oidx = sel.indices[sel.indices != idx]
        others = ref + u.atoms[oidx]

        # extract all water molecules close to the acid
        pairs, dists = mda.lib.distances.capped_distance(
            others.positions, curr.position, max_cutoff=11.0, box=u.dimensions)
        indices = np.where(dists > 0)
        water_less = others[pairs[indices][:, 0]]

        if len(water_less) != 0:
            # extract all protons close to acid
            pairs, dists = mda.lib.distances.capped_distance(
                prot.positions, curr.position, max_cutoff=11.0, box=u.dimensions)
            indices = np.where(dists > 0)
            prot_less = prot[pairs[indices][:, 0]]

            # compute all distances between water, acid and protons
            pairs_water, dists_water = mda.lib.distances.capped_distance(
                prot_less.positions, water_less.positions, max_cutoff=11, box=u.dimensions)
            pairs_acid, dists_acid = mda.lib.distances.capped_distance(
                prot_less.positions, curr.position, max_cutoff=11, box=u.dimensions)

            dist_mat_water = np.full((len(prot_less) + 1, len(water_less)), np.inf)
            dist_mat_water[pairs_water[:, 0] + 1, pairs_water[:, 1]] = dists_water

            dist_mat_acid = np.full((len(prot_less) + 1, 1), np.inf)
            dist_mat_acid[pairs_acid[:, 0] + 1, pairs_acid[:, 1]] = dists_acid

            min_water = dist_mat_water.min(axis=1)
            n_prot = dist_mat_acid[:, 0] < min_water

            if scheme == "legacy":
                if n_prot.sum() > 0:
                    n_prots += 1
            elif scheme == 'total':
                if n_prot.sum() > 0:
                    n_prots += n_prot.sum()

        else:
            pairs_acid, dists_acid = mda.lib.distances.capped_distance(
                prot_less.positions, curr.position, max_cutoff=11, box=u.dimensions)
            n_prot = dists_acid
            if len(n_prot) != 0:
                n_prots += 1

    return n_prots


def analyze(args):
    degrees_of_deprot = []
    traj_paths, tpr_paths, ph_values = setup_dirs(args.ph, args.dir, args.traj, args.tpr)

    for traj_path, tpr_path in zip(traj_paths, tpr_paths):
        degree_of_deprot = do_one_trajectory(tpr_path, traj_path, args.sel, args.ref, args.start,
                                             args.end, args.scheme)
        degrees_of_deprot.append(degree_of_deprot)

    mean, std = do_statistics(degrees_of_deprot)
    save_to_file(args.out, ph_values, mean, args.sel)

    if args.std_out:
        save_to_file(args.std_out, ph_values, std, args.sel)

    if args.plot_out:
        do_plotting(args.plot_out, mean, ph_values)
