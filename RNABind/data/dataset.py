import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from loguru import logger

import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.loader import DataLoader

from RNABind.graphs import construct_rna_graph
from RNABind.utils import ComplexPocket


#########################################################################################
# ------------------ RNA-ligand dataset (binding sites prediction)------------------
class RNABindingSiteDataset(InMemoryDataset):
    def __init__(self, root,
                 data_path='~/RNABind/bs_data/rna_bs_set_1.csv',    # # please change to your own path
                 target_file='RNA_binding_site_data_EGNN_ernierna.pt',
                 transform=None, pre_transform=None, pre_filter=None):
        self.file_path = data_path
        self.target_file = target_file
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        return [self.file_path]

    @property
    def processed_file_names(self):
        return [self.target_file]

    def process(self):
        # skip if the file already exists
        if os.path.exists(os.path.join(self.processed_dir, self.processed_file_names[0])):
            return
        
        df = pd.read_csv(self.raw_paths[0])
        pdb_list = df['pdb_id'].values
        # split_list = df['split'].values
        logger.info(f'Processing {len(pdb_list)} PDB files')

        data_list = []
        error_list = []
        # for pdb, split in tqdm(zip(pdb_list, split_list), total=len(pdb_list), desc="Processing pairs"):
        for pdb in tqdm(pdb_list, total=len(pdb_list), desc="Processing pairs"):
            # construct graphs
            try:
                ligand_dir = '~/RNABind/bs_data/ligand_pdb'    # please change to your own path
                ligand_id = pdb + '_ligand'
                rna_dir = '~/RNABind/bs_data/rna_pdb'          # please change to your own path
                rna_id = pdb + '_rna'
                rna_graph = construct_rna_graph(rna_dir, rna_id)
                # get the pocket of the complex
                complex_pocket = ComplexPocket(rna_dir, rna_id, ligand_dir, ligand_id, recptor_type='RNA') # todo
                binding_site_ids = complex_pocket.get_binding_site()
                binding_site_labels = [1 if id in binding_site_ids else 0 for id in rna_graph.residue_id]
                binding_site_labels = torch.tensor(binding_site_labels)


                data = Data(# rna attributes
                            x=rna_graph.x, # one-hot encoding of nucleotides
                            edge_index=rna_graph.edge_index,
                            coord=rna_graph.coords,     # 3D coordinates, pos
                            # fm_embedding=rna_graph.fm_embedding,  # RNA-FM embedding
                            # lucaone_embedding=rna_graph.lucaone_embedding,  # LucaOne embedding
                            # protrna_embedding=rna_graph.protrna_embedding,  # ProtRNA embedding
                            ernierna_embedding=rna_graph.ernierna_embedding,  # Ernierna embedding
                            # rnabert_embedding=rna_graph.rnabert_embedding,  # RNABert embedding
                            # rinalmo_embedding=rna_graph.rinalmo_embedding,  # Rinalmo embedding
                            # rnamsm_embedding=rna_graph.rnamsm_embedding,  # RNAMSM embedding
                            # rnaernie_embedding=rna_graph.rnaernie_embedding,  # RNAErnie embedding
                            # hyenadna_embedding=rna_graph.hyenadna_embedding,  # HyenaDNA embedding
                            # caduceus_embedding=rna_graph.caduceus_embedding,  # Caduceus embedding
                            binding_site=binding_site_labels,
                            pdb_id=pdb)
                from RNABind.graphs.geometry import _rbf, _positional_embeddings
                edge_index = data.edge_index
                coord = data.coord
                E_vectors = coord[edge_index[0]] - coord[edge_index[1]]             
                rbf = _rbf(E_vectors.norm(dim=-1), D_count=16)             
                pos_embedding = _positional_embeddings(edge_index, 16)   

                data.edge_attr = torch.cat([rbf, pos_embedding], dim=-1)
                data_list.append(data) 
            
            except Exception as e:
                logger.warning(f"Error processing {pdb}", exc_info=True)
                import traceback
                traceback.print_exc()
                error_list.append(pdb)

        # Save error_list to a CSV file
        error_list_path = os.path.join('~/RNABind/bs_data', 'rl_error_list.csv')
        error_df = pd.DataFrame(error_list, columns=['pdb_id'])
        error_df.to_csv(error_list_path, index=False)

        # this just a pipeline, we can ignore it
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


# ------------------------------------------------------------------
# load dataset, split dataset into train, val, test
# ------------------------------------------------------------------
def load_structure_rl_dataset(path, dataset_number, embedding_type='lucaone'):
    r"using usalign score to ensure the test set have unseen rna pocket"

    rna_df = pd.read_csv(f'~/RNABind/bs_data/rna_bs_set_{dataset_number}.csv')
    pdb2split = dict(zip(rna_df['pdb_id'], rna_df['split']))
    dataset = RNABindingSiteDataset(root=path, target_file=f'RNA_binding_site_data_EGNN_{embedding_type}.pt')
    train_idx, val_idx, test_idx = [], [], []
    for i in range(len(dataset)):
        pdb_id = dataset[i].pdb_id
        if pdb2split[pdb_id] == 'train':
            train_idx.append(i)
        elif pdb2split[pdb_id] == 'validation':
            val_idx.append(i)
        elif pdb2split[pdb_id] == 'test':
            test_idx.append(i)
        else:
            raise ValueError(f"Invalid split: {pdb2split[pdb_id]}")
    train_dataset = dataset[torch.LongTensor(train_idx)]
    val_dataset = dataset[torch.LongTensor(val_idx)]
    test_dataset = dataset[torch.LongTensor(test_idx)]

    return train_dataset, val_dataset, test_dataset
    

# ------------------------------------------------------------------
# build data loader
# -----------------------------------------------------------------
def build_rna_bs_dataloader(args):
    r"for protein-ligand dataset, we simply split the dataset into train, val, test randomly"
    # structure split
    train_dataset, val_dataset, test_dataset = load_structure_rl_dataset(args.data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    all_size = train_size + val_size + test_size
    logger.info(f'Total = {all_size:,} | '
                f'train = {train_size:,} | '
                f'val = {val_size:,} | '
                f'test = {test_size:,}')

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    PATH = '~/RNABind/bs/RNA' # please change to your own path
    os.makedirs(PATH, exist_ok=True)
    logger_tag = 'rl_bs_data'
    log_file_name = '~/RNABind/logs/{1}/{0}_{1}.log'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), logger_tag)
    logger.add(log_file_name)
    dataset = RNABindingSiteDataset(root=PATH)
