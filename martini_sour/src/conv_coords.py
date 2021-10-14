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
from scipy.spatial.transform import Rotation
import random


BEAD_TYPES = {'water': ['DN', 'H+'],
              'acid': ['D', 'DP', 'H+'],
              'base': ['D', 'DP'],
              'acid-deprot': ['D', 'DP'],
              'base-prot': ['D', 'DP', 'H+']}


def get_random_vectors(n_randoms=1000, length=1.7):
    """
    Generate randomly rotated 3 points that all have 120 degree angle to the
    center point with a given vector length. This is done n_randoms times.
    0.8660254 ~= 0.75**0.5

    Parameters
    ----------
    length : float, optional
        Length of the vectors. The default is 1.7 Angstrom.

    Returns
    -------
    (n_randoms, 3, 3) Numpy array

    """

    vectors = [[1, 0, 0], [-0.5, 0.8660254, 0], [-0.5, -0.8660254, 0]]
    random_rots = Rotation.random(n_randoms)

    random_vecs = np.empty((n_randoms, 3, 3))
    for i, vec in enumerate(vectors):
        random_vecs[:, i, :] = random_rots.apply(vec)

    return random_vecs*length


class Particles():
    def __init__(self):
        self.resids = []
        self.resnames = []
        self.names = []
        self.indices = []
        self.coords = []

    def add_particle(self, resid, resname, name, coord):
        self.resids.append(resid)
        self.resnames.append(resname)
        self.names.append(name)
        self.coords.append(coord)

    def __len__(self):
        return len(self.resids)


def check_input(args):
    if len(args.bead_type) != len(args.sel):
        raise ValueError('Unmatching number of -bead and -sel was provided.')


def select_titratable_beads(uni, selection_commands):
    """
    Parameters
    ----------
    uni : MDAnalysis Universe object
    selection_commands : list of strings
        List of MDAnalysis selection commands

    Returns
    -------
    selections : 1D Numpy array
        Array with the selection index of each atom. -1 if not in any selection

    """

    selections = np.full(len(uni.atoms), -1)

    for i, selection_command in enumerate(selection_commands):
        idx = uni.select_atoms(selection_command).indices

        if len(idx) == 0:
            raise ValueError(f'Selection command "{selection_command}" contains zero particles. '
                             'Check your input.')
        if np.any(selections[idx] != -1):
            raise ValueError(f'Selection command "{selection_command}" contains particles from '
                             'a different selection. Check your input.')

        selections[idx] = i

    return selections


def add_extra_particles(beads, protons, atom, particle_list, random_vecs, prot_name):
    """
    Select a random vector and add each necessary particle to either the protons or
    beads object.

    Parameters
    ----------
    beads : Particles object

    protons : Particles object

    atom : MDAnalysis atom object

    particle_list : list
        all the necessary particles for the corresponding BEAD_TYPE
    random_vecs : (n_randoms, 3, 3) Numpy array

    """

    vectors = random_vecs[random.randint(0, 999)]

    for i, particle in enumerate(particle_list):
        if particle == 'H+':
            protons.add_particle(None, particle, prot_name, atom.position + vectors[i])
        else:
            beads.add_particle(atom.resid, atom.resname, particle, atom.position + vectors[i])

    return beads, protons


def create_titratable_universe(beads, protons, dimensions):
    """
    Combines all beads and protons to create the titratable MDAnalysis universe

    Parameters
    ----------
    beads : Particles object

    protons : Particles object

    dimensions : (3) Numpy array
        Dimensions of the unit cell (Angstrom)

    """

    n_atoms = len(beads)+len(protons)
    max_bead_resid = max(beads.resids)
    resids = np.append(beads.resids, 1 + np.arange(max_bead_resid, max_bead_resid+len(protons)))
    resnames = beads.resnames + protons.resnames
    names = beads.names + protons.names

    uni = mda.Universe.empty(n_atoms, trajectory=True, n_residues=n_atoms,
                             atom_resindex=np.arange(0, n_atoms),
                             residue_segindex=np.zeros(n_atoms))

    uni.add_TopologyAttr('resid', resids)
    uni.add_TopologyAttr('resname', resnames)
    uni.add_TopologyAttr('name', names)

    uni.atoms.positions = np.append(beads.coords, protons.coords, axis=0)
    uni.dimensions = dimensions

    return uni


def conv_coords(args):
    check_input(args)

    beads = Particles()
    protons = Particles()
    random_vecs = get_random_vectors()

    uni = mda.Universe(args.input_file)
    selections = select_titratable_beads(uni, args.sel)

    for atom, selection in zip(uni.atoms, selections):
        beads.add_particle(atom.resid, atom.resname, atom.name, atom.position)
        if selection >= 0:
            particle_list = BEAD_TYPES[args.bead_type[selection]]
            beads, protons = add_extra_particles(beads, protons, atom, particle_list, random_vecs,
                                                 args.prot)

    titr_uni = create_titratable_universe(beads, protons, uni.dimensions)
    titr_uni.atoms.write(args.out_file)
