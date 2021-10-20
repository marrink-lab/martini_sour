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
from pathlib import Path
from .titrate import get_ph_values


def setup_dirs(ph_values, sub_dir, prefix):
    """
    Sets up directory paths based on whether multidir options have been provided or not.
    """

    if ph_values:
        ph_values = get_ph_values(ph_values)
        sim_dirs = [Path(f'{prefix}{ph}/{sub_dir}') for ph in ph_values]
    else:
        sim_dirs = [Path.cwd()]

    return sim_dirs, np.array(ph_values, dtype=float)


def save_degree_of_prot_to_file(out_file, ph_values, props):

    fmt = '%10.5f'*props.shape[1]
    header = ''.join([f'sel. {i}'.rjust(10) for i in range(1, props.shape[1]+1)])

    if ph_values.size != 0:
        props = np.column_stack((ph_values, props))
        fmt = '%6.2f' + fmt
        header = '    pH' + header
    fmt = '  ' + fmt

    np.savetxt(out_file, props, fmt=fmt, header=header)


def do_statistics(degrees_of_deprot):
    """
    This is probably where a more complicated statistical analysis would be inferfaced
    """

    means = np.array(degrees_of_deprot).mean(axis=-1)
    stds = np.array(degrees_of_deprot).std(axis=-1)

    return means, stds


def calc_degree_of_deprot(ph, pka, q):
    return 1/(10**(q*(pka-ph))+1)


def fit_titration_curve(ph_values, means, stds):
    fits = []
    fit_std_errs = []
    for mean, std in zip(means.T, stds.T):
        popt, pcov = curve_fit(calc_degree_of_deprot, ph_values, mean, p0=(7.5, 0.75),
                               sigma=std, absolute_sigma=True)
        std_err = np.sqrt(np.diag(pcov))
        fits.append(popt)
        fit_std_errs.append(std_err)
    return fits, fit_std_errs


def write_pka_to_file(output_file, ph_values, fits, fit_std_errs):
    sel_range = np.arange(1, len(fits)+1)
    fmt = '%8d' + '%10.5f'*4
    header = 'Sel. #       pKa         q  err(pKa)    err(q)'
    props = np.column_stack((sel_range, fits, fit_std_errs))
    np.savetxt(output_file, props, fmt=fmt, header=header)


def do_plotting(out_file, ph_values, fits, fit_std_errs, means):
    ph_range = np.arange(ph_values.min(), ph_values.max()+1e-3, 0.1)
    fig = plt.figure()

    for i, (fit, std_err, mean) in enumerate(zip(fits, fit_std_errs, means.T), start=1):
        points, = plt.plot(ph_values, mean, '^',
                           label=f'Sel. {i}, pKa = {fit[0]:.2f}\u00B1{std_err[0]:.2f}'
                                 f', q = {fit[1]:.2f}\u00B1{std_err[1]:.2f}')
        plt.plot(ph_range, calc_degree_of_deprot(ph_range, *fit), color=points.get_color())

    plt.xlabel('pH')
    plt.ylabel('Degree of deprotonation')
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches='tight')


def construct_reference(u, sel_commands, ref_command):
    """
    Reference is constructed as ref_command + all selections that are not the current one.

    """

    if ref_command:
        refs = [u.select_atoms(ref_command) for _ in enumerate(sel_commands)]
    else:
        refs = [u.select_atoms('') for _ in enumerate(sel_commands)]

    if len(sel_commands) > 1:
        other_sels = [' or '.join([command for command in sel_commands if command != sel_command])
                      for sel_command in sel_commands]
        refs = [ref + u.select_atoms(other_sel) for ref, other_sel in zip(refs, other_sels)]

    return refs


def do_one_trajectory(sim_dir, tpr_file, traj_file, sel_commands, ref_command, prot, start, end,
                      scheme):
    """

    Parameters
    ----------
    sim_dir : Pathlib path
    tpr_file : Pathlib path
    traj_file : Pathlib path
    sel_commands : list of str
        All selection commands given by -sel
    ref_command : str
        Selection command for the remaining titratable beads, given by -ref
    prot : str
        name of the proton bead
    start : int
        time (ps) to start the analysis
    end : int
        time (ps) to end the analysis
    scheme : str
        proton counting scheme

    Returns
    -------
    degree_of_deprot : (n_selections, n_frames) np.array
        Degree of deprotonation matrix for each selection and for each frame

    """

    u = mda.Universe(str(sim_dir / tpr_file), str(sim_dir / traj_file))

    sels = [u.select_atoms(sel_command) for sel_command in sel_commands]
    prot = u.select_atoms(f'name {prot}')
    refs = construct_reference(u, sel_commands, ref_command)

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
    sim_dirs, ph_values = setup_dirs(args.ph, args.dir, args.prefix)

    for sim_dir in sim_dirs:
        degree_of_deprot = do_one_trajectory(sim_dir, args.tpr, args.traj, args.sel, args.ref,
                                             args.prot, args.start, args.end, args.scheme)
        degrees_of_deprot.append(degree_of_deprot)

    means, stds = do_statistics(degrees_of_deprot)
    save_degree_of_prot_to_file(args.out, ph_values, means)

    if args.std_out:
        save_degree_of_prot_to_file(args.std_out, ph_values, stds)

    if args.fit:
        if ph_values.size < 4:
            print('WARNING: You requested a titration curve fitting with less than 4 pH data '
                  ' points. It will be ignored.')
        else:
            fits, fit_std_errs = fit_titration_curve(ph_values, means, stds)
            write_pka_to_file(args.fit_out, ph_values, fits, fit_std_errs)
            do_plotting(args.plot_out, ph_values, fits, fit_std_errs, means)
