"""Functions to add embeddings from pre-trained language models molecular graphs."""
# added by wmzhu
from __future__ import annotations

import networkx as nx
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from unimol_tools import UniMolRepr


def add_unimol_embedding(g: nx.Graph,
                         use_pdb_coords: bool = False) -> nx.Graph:
    r"""Add Uni-Mol embedding to nodes in a molecular graph as defined in https://openreview.net/forum?id=6K2RM6wVqKu

    :param g: Molecular graph to add Uni-Mol embeddings to.
    :type g: nx.Graph
    :param use_pdb_coords: Whether to use original coordinates for Uni-Mol representation.
    :type use_pdb_coords: bool
    :return: Molecular graph with Uni-Mol embeddings added.
    :rtype: nx.Graph
    """

    if not isinstance(g, nx.Graph):
        raise TypeError("Input must be a networkx graph")
    
    nxmol_index = [g.nodes[node]['element'] for node in g.nodes]
    rdmol_index = [atom.GetSymbol() for atom in g.graph["rdmol"].GetAtoms()]
    assert nxmol_index == rdmol_index, "The atoms in networkx and rdmol are not align."

    clf = UniMolRepr(data_type="molecule", remove_hs=True)

    # Check if nodes have 'coordinates' attribute
    if  use_pdb_coords:
        # Use coordinates to get Uni-Mol representation
        # custom dict of {'atoms':[['C','C],['C','H','O']], 'coordinates':[coordinates_1,coordinates_2]}
        atoms_coordinates_dict = {
            'atoms': [[g.nodes[node]['element'] for node in g.nodes]],
            'coordinates': [g.graph["coords"]]
        }
    else:
        if not g.graph.get("rdmol"):
            raise ValueError("Input graph must have a rdmol attribute")
        # Use RDKit mol to generate coordinates
        atoms_coordinates_dict = inner_smi2coords(g.graph["rdmol"])

    # fix bug: after install unimol_tools, we need to comment the line 69 in unimol_tools/unimol_tools/data/datareader.py
    # https://github.com/dptech-corp/Uni-Mol/blob/c458c0fae2318b6bf49fed51f56fdc8d4b7003e1/unimol_tools/unimol_tools/data/datareader.py#L69
    unimol_repr = clf.get_repr(atoms_coordinates_dict, return_atomic_reprs=True)

    # atomic level repr, align with rdkit mol.GetAtoms()
    for node, emb in zip(g.nodes, unimol_repr["atomic_reprs"][0]):
        g.nodes[node]["unimol_embedding"] = emb
    return g


# modified from https://github.com/dptech-corp/Uni-Mol/blob/c458c0fae2318b6bf49fed51f56fdc8d4b7003e1/unimol_tools/unimol_tools/data/conformer.py#L100
def inner_smi2coords(mol, seed=42, mode='fast', remove_hs=True):
    r"""
    This function is responsible for converting a RDKit Mol object into 3D coordinates for each atom in the molecule. It also allows for the generation of 2D coordinates if 3D conformation generation fails, and optionally removes hydrogen atoms and their coordinates from the resulting data.

    :param mol: (rdkit.Chem.rdchem.Mol) The RDKit Mol object to convert to 3D coordinates.
    :param seed: (int, optional) The random seed for conformation generation. Defaults to 42.
    :param mode: (str, optional) The mode of conformation generation, 'fast' for quick generation, 'heavy' for more attempts. Defaults to 'fast'.
    :param remove_hs: (bool, optional) Whether to remove hydrogen atoms from the final coordinates. Defaults to True.

    :return: A dict containing the list of atom symbols and their corresponding 3D coordinates.
    :raises AssertionError: If no atoms are present in the molecule or if the coordinates do not align with the atom count.
    """
    smi = Chem.MolToSmiles(mol)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert len(atoms)>0, 'No atoms in molecule: {}'.format(smi)
    try:
        # will random generate conformer with seed equal to -1. else fixed random seed.
        res = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if res == 0:
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        ## for fast test... ignore this ###
        elif res == -1 and mode == 'heavy':
            AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                AllChem.Compute2DCoords(mol)
                coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
                coordinates = coordinates_2d
        else:
            AllChem.Compute2DCoords(mol)
            coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
            coordinates = coordinates_2d
    except:
        print("Failed to generate conformer, replace with zeros.")
        coordinates = np.zeros((len(atoms),3))
    assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smi)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with {}".format(smi)
        return {'atoms': [atoms_no_h], 'coordinates': [coordinates_no_h]}
    else:
        return {'atoms': [atoms], 'coordinates': [coordinates]}