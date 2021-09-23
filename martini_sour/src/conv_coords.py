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
from itertools import chain
from scipy.spatial.transform import Rotation
import random


bead_type_dict = {'water': ['DN', 'H+'],
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

    random_vecs = []
    for vec in vectors:
        random_vecs.append(random_rots.apply(vec))
    random_vecs = np.transpose(random_vecs, (1, 0, 2))

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

    def set_indices(self, start=1):
        self.indices = np.arange(start, start+len(self)) % 100000

    def set_resids(self, start=1):
        self.resids = np.arange(start, start+len(self.resids)) % 100000

    def convert_coords_to_nm(self):
        self.coords = np.array(self.coords)*0.1

    def __iter__(self):
        return zip(self.resids, self.resnames, self.names, self.indices, self.coords)

    def __len__(self):
        return len(self.resids)


def check_input(args):
    if len(args.bead_type) != len(args.sel):
        raise ValueError('Unmatching number of -bead and -sel was provided.')


def do_selections(uni, selection_commands):
    """
    Currently, if an atom is in multiple selections, the last one selection
    is taken.

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
        selections[idx] = i

    return selections


def add_extra_particles(beads, protons, atom, particle_list, random_vecs):

    positions = random_vecs[random.randint(0, 999)]

    for i, particle in enumerate(particle_list):
        if particle == 'H+':
            protons.add_particle(None, particle, 'POS', atom.position + positions[i])
        else:
            beads.add_particle(atom.resid, atom.resname, particle, atom.position + positions[i])

    return beads, protons


def do_one_atom(selection, bead_types, random_vectors, beads, protons, atom):
    """
    For a single atom, add the original bead, then if it is a titratable bead,
    add necessary extra particles
    """

    beads.add_particle(atom.resid, atom.resname, atom.name, atom.position)

    if selection >= 0:
        particle_list = bead_type_dict[bead_types[selection]]
        beads, protons = add_extra_particles(beads, protons, atom, particle_list, random_vectors)

    return beads, protons


def prepare_for_writing(beads, protons, dimensions):
    """
    Do necessary unit (coordinates) and number (resids, indices) conversions
    before writing.
    """

    beads.set_indices()
    beads.resids = np.array(beads.resids) % 100000
    protons.set_indices(start=beads.indices[-1]+1)
    protons.set_resids(start=beads.resids[-1]+1)

    beads.convert_coords_to_nm()
    protons.convert_coords_to_nm()
    dimensions *= 0.1

    return beads, protons, dimensions


def write_ff(out_file, dimensions, beads, protons):
    with open(out_file, 'w') as file:
        file.write('titratable MARTINI \n')
        file.write(f'{len(beads)+len(protons):<12d}\n')
        for resid, resname, name, idx, coord in chain(beads, protons):
            file.write(f'{resid:>5d}{resname:<5s}{name:>5s}{idx:5d}'
                       f'{coord[0]:8.3F}{coord[1]:8.3F}{coord[2]:8.3F}\n')
        file.write(f'    {dimensions[0]:>8.3F}{dimensions[1]:>8.3F}{dimensions[2]:>8.3F}\n')


def initialize(args):
    check_input(args)
    beads = Particles()
    protons = Particles()
    random_vecs = get_random_vectors()

    return beads, protons, random_vecs


def conv_coords(args):
    beads, protons, random_vecs = initialize(args)

    uni = mda.Universe(args.input_file)
    selections = do_selections(uni, args.sel)

    for atom, selection in zip(uni.atoms, selections):
        beads, protons = do_one_atom(selection, args.bead_type, random_vecs, beads, protons, atom)

    beads, protons, dimensions = prepare_for_writing(beads, protons, uni.dimensions)
    write_ff(args.out_file, dimensions, beads, protons)
