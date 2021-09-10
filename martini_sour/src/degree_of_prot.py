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
import multiprocessing 
from tqdm import tqdm
import argparse

def analyze(args):
    """
    Legacy counting scheme.
    """
    # 1. Load Universe
    u = mda.Universe(args.tpr,args.traj)

    ref    = u.select_atoms(args.ref)
    sel    = u.select_atoms(args.sel)
    prot   = u.select_atoms("name POS")
    count = 0

    # 2. Maximum number of protons
    n_prot_traj = np.zeros((len(u.trajectory[args.start:args.end]),2))

    for ts in tqdm(u.trajectory[args.start:args.end]):

        n_prots = 0
        for idx in sel.indices:

            curr = u.atoms[idx]
            oidx = sel.indices[sel.indices != idx]
            others = ref + u.atoms[oidx]

            # extract all water molecules close to the acid
            pairs, dists = mda.lib.distances.capped_distance(others.positions,curr.position,max_cutoff=11.0,box=u.dimensions)
            indices = np.where(dists > 0)
            water_less = others[pairs[indices][:,0]]

            if len(water_less) != 0:

               # extract all protons close to acid
               pairs, dists = mda.lib.distances.capped_distance(prot.positions,curr.position,max_cutoff=11.0,box=u.dimensions)
               indices = np.where(dists > 0)
               prot_less = prot[pairs[indices][:,0]]

               # compute all distances between water, acid and protons

               pairs_water, dists_water = mda.lib.distances.capped_distance(prot_less.positions,water_less.positions,max_cutoff=11,box=u.dimensions)
               pairs_acid , dists_acid  = mda.lib.distances.capped_distance(prot_less.positions,curr.position,max_cutoff=11,box=u.dimensions)

               dist_mat_water = np.full((len(prot_less) + 1, len(water_less)), np.inf)
               dist_mat_water[pairs_water[:, 0] + 1, pairs_water[:, 1]] = dists_water

               dist_mat_acid = np.full((len(prot_less) + 1,1), np.inf)
               dist_mat_acid[pairs_acid[:, 0] + 1, pairs_acid[:, 1]] = dists_acid

               min_water= dist_mat_water.min(axis=1)
               n_prot =  dist_mat_acid[:,0] < min_water

               if args.scheme == "def":
                  if n_prot.sum() > 0:
                     n_prots += 1
               elif args.scheme == 'total':
                  if n_prot.sum() > 0:
                     n_prots += n_prot.sum()

            else:
                pairs_acid , dists_acid  = mda.lib.distances.capped_distance(prot_less.positions,curr.position,max_cutoff=11,box=u.dimensions)
                n_prot = dists_acid
                if len(n_prot) != 0:
                    n_prots += 1

        n_prot_traj[count,1] = n_prots
        n_prot_traj[count,0] = ts.time
        count=count+1

    np.savetxt(args.out,n_prot_traj)
