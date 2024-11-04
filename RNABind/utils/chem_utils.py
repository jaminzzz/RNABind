import numpy as np
from rdkit import Chem

from Graphein.graphein.protein.graphs import read_pdb_to_dataframe, process_dataframe
    

# -----------------------------------------------------------------------------
# Get the pocket and labels for training.
# Insight from https://github.com/ComputArtCMCG/PLANET/blob/main/chemutils.py
# -----------------------------------------------------------------------------
class ComplexPocket():
    r"""This class is used to get the pocket residues of the complex 
    and the interaction information between the ligand and the pocket residues.
    """
    def __init__(self, 
                 recptor_dir: str,
                 recptor_id: str,
                 molecule_dir: str,
                 molecule_id: str,
                 recptor_type: str = "protein"):

        if recptor_type == "protein":
            granularity = "atom"
            recptor_pdb = recptor_dir + '/' + recptor_id + '.pdb'  
        elif recptor_type == "RNA":
            granularity = "rna_atom"
            recptor_pdb = recptor_dir + '/' + recptor_id + '.pdb'
        else:
            raise ValueError("recptor_type must be protein or RNA")
        
        # get repcetor, atom coordinates and residue ids
        self.raw_df = read_pdb_to_dataframe(path=recptor_pdb)
        self.df = process_dataframe(self.raw_df, granularity=granularity)

        # get ligand and compute centeroid
        self.ligand_pdb = molecule_dir + '/' + molecule_id + '.pdb'
        self.rdmol = Chem.MolFromPDBFile(self.ligand_pdb)

        # self.ligand_sdf = molecule_dir + '/' + molecule_id + '.sdf'
        # self.rdmol = Chem.SDMolSupplier(self.ligand_sdf, sanitize=False, removeHs=False)[0]
        # if self.rdmol is None:
        #     self.ligand_mol2 = molecule_dir + '/' + molecule_id + '.mol2'
        #     self.rdmol = Chem.MolFromMol2File(self.ligand_mol2, sanitize=False, removeHs=False)
    
    def get_binding_site(self):
        r"""To get the binding site of the complex. 
        If the distance of any atom in the ligand to any atom in a residue is less than 6.0Å,
        the residue is considered as a binding site residue.
        """
        coordinates = self.df[['x_coord', 'y_coord', 'z_coord']].values
        residue_ids = self.df['residue_id'].values

        coords = [
            list(self.rdmol.GetConformer(0).GetAtomPosition(idx))
            for idx in range(self.rdmol.GetNumAtoms())
        ]

        binding_site_residue_ids = []
        for id, coordinate in zip(residue_ids, coordinates):
            for coord in coords:
                distance = np.sqrt(np.sum(np.square(np.array(coordinate) - np.array(coord))))
                if distance < 4.0:
                    binding_site_residue_ids.append(id)
                    break

        binding_site_residue_ids = set(binding_site_residue_ids)

        return binding_site_residue_ids