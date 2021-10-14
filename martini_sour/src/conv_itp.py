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
import os
import sys
import networkx as nx
import vermouth
from vermouth.forcefield import ForceField
from vermouth.gmx.itp_read import read_itp
from vermouth.graph_utils import make_residue_graph
from vermouth.log_helpers import StyleAdapter, get_logger
from vermouth.file_writer import open, DeferredFileWriter
from vermouth.citation_parser import citation_formatter
from vermouth.processors.annotate_mut_mod import AnnotateMutMod
from vermouth.processors.processor import Processor
from vermouth.processors.do_links import DoLinks
from martini_sour import DATA_PATH

LOGGER = StyleAdapter(get_logger(__name__))

class AnnotatepKas(Processor):

    def __init__(self, pKas, res_graph=None):
        self.pKas = {res: pKa for res, pKa in pKas}
        self.res_graph = res_graph

    def run_molecule(self, molecule):
        if not self.res_graph:
            self.res_graph = make_residue_graph(molecule)

        for resid in self.res_graph:
            graph = self.res_graph.nodes[resid]['graph']
            for node in graph.nodes:
                if "modification" in graph.nodes[node]\
                and graph.nodes[node]["atomname"] in ["BAS", "ACD"]:
                    resname = graph.nodes[node]["resname"]
                    pKa = self.pKas[resname]
                    atomtype = graph.nodes[node]["atype"]
                    suffix = graph.nodes[node]["suffix"]
                    new_atype = atomtype + "_" + pKa + suffix
                    molecule.nodes[node]["atype"] = new_atype

def _relable_molecule(molecule, mapping):
    relabled_graph = nx.relabel_nodes(molecule, mapping=mapping, copy=True)
    new_molecule = molecule.copy()
    new_molecule.add_nodes_from(relabled_graph.nodes(data=True))
    new_molecule.remove_edges_from(molecule.edges)
    new_molecule.add_edges_from(relabled_graph.edges)
    return new_molecule


class AddTitratableBeads(Processor):

    def __init__(self, force_field, res_graph=None):
        """
        """
        self.res_graph = res_graph
        self.force_field = force_field

    def insert_node(self, molecule, attrs, insert_idx):
        """
        Insert a node with attributes `attrs` into an
        existing `molecule` at the position `insert_idx`.
        All interactions are relabeled as well.
        """
        # establish a correspondance between the old nodes
        # and the new nodes
        nodes = list(molecule.nodes)
        correspondance = dict(zip(nodes, nodes))
        temp_idx = len(nodes)
        for from_node, to_node in correspondance.items():
            if from_node >= insert_idx:
                correspondance[from_node] = to_node + 1
        correspondance[temp_idx] = insert_idx
        # add the new node at the end of the graph
        molecule.add_node(temp_idx, **attrs)

        # also add it to the residue graph
        resid = attrs['resid'] - 1
        self.res_graph.nodes[resid]['graph'].add_node(temp_idx, **attrs)

        # relabel the molecule with the new nodes
        molecule = _relable_molecule(molecule, mapping=correspondance)

        # relabel residue graph
        for res in self.res_graph:
            nx.relabel_nodes(self.res_graph.nodes[res]['graph'], mapping=correspondance)

        # update the interactions lists
        for inter_type in molecule.interactions:
            for inter in molecule.interactions[inter_type]:
                new_atoms = tuple([correspondance[atom] for atom in inter.atoms])
                inter._replace(atoms=new_atoms)

        return molecule, correspondance, temp_idx

    def replace_node(self, molecule, node, attrs):
        """
        Replace the attributes of an existing node.
        """
        for target_node in molecule.nodes:
            if molecule.nodes[target_node]["atomname"] == node:
                molecule.nodes[target_node].update(attrs)
                return target_node

    def convert_titratable_beads(self, molecule):
        for res_node in self.res_graph:
            graph = self.res_graph.nodes[res_node]['graph']
            node = list(graph.nodes)[0]
            if "modification" in graph.nodes[node]:
                mod_name = graph.nodes[node]["modification"][0]
                modf = self.force_field.modifications[mod_name]
                modf_to_mol = {}
                for node in modf.nodes:
                    if "replace" in modf.nodes[node]:
                        attrs = modf.nodes[node]["replace"]
                        mol_node = self.replace_node(molecule, node, attrs)
                        modf_to_mol[node] = mol_node
                    elif "add" in modf.nodes[node]:
                        attrs = modf.nodes[node]["add"]
                        idx = attrs.pop("idx")
                        insert_idx = idx + min(graph.nodes)
                        attrs["resid"] = graph.nodes[res_node]["resid"]
                        attrs["resname"] = graph.nodes[res_node]["resname"]
                        attrs["atomname"] = node
                        attrs["charge_group"] = graph.nodes[res_node]["charge_group"]
                        molecule, mapping, mol_node = self.insert_node(molecule, attrs, insert_idx)
                        modf_to_mol[node] = mol_node
                        for node, mol_node in modf_to_mol.items():
                            modf_to_mol[node] = mapping[mol_node]

                for idx, jdx in modf.edges:
                    molecule.add_edge(modf_to_mol[idx], modf_to_mol[jdx])

        return molecule

    def run_molecule(self, molecule):
        if not self.res_graph:
            self.res_graph = make_residue_graph(molecule)
        molecule = self.convert_titratable_beads(molecule)
        return molecule

def conv_itp(args):
    # load force-field
    force_field = ForceField(os.path.join(DATA_PATH,args.forcefield))

    # read itp and load into molecule
    with open(args.inpath) as _file:
        lines = _file.readlines()

    read_itp(lines, force_field)
    mol_name = list(force_field.blocks.keys())[0]
    mol = force_field.blocks[mol_name].to_molecule()

    # set mol labels and make edges
    mol.make_edges_from_interaction_type("bonds")
    degrees = {node: str(deg) for node, deg in mol.degree()}
    nx.set_node_attributes(mol, degrees, "degree")

    # set specifications for residues to
    # convert to titratable
    if args.auto:
        resspecs = ["ASP", "GLU", "LYS"]
    else:
        resspecs = args.specs

    # annotate all titratable residues with modifications
    AnnotateMutMod(modifications=resspecs).run_molecule(mol)

    # convert beads to titratable beads
    mol = AddTitratableBeads(force_field).run_molecule(mol)

    # apply links
    DoLinks().run_molecule(mol)

    # annotate pKas
    AnnotatepKas(pKas=args.pKas).run_molecule(mol)

    # write the converted molecule
    with open(args.outpath, 'w') as outpath:
        header = [' '.join(sys.argv) + "\n"]
        header.append("Please cite the following papers:")
        for citation in mol.citations:
            cite_string = citation_formatter(mol.force_field.citations[citation])
            LOGGER.info("Please cite: " + cite_string)
            header.append(cite_string)

        vermouth.gmx.itp.write_molecule_itp(mol, outpath,
                                            moltype=mol_name,
                                            header=header)
    DeferredFileWriter().write()
