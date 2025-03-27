# Referenced from https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/rna_graph_tutorial.ipynb
from functools import partial

import torch
from torch_geometric.utils import from_networkx, remove_self_loops
from torch_geometric.data import Data

import Graphein.graphein.rna as gr
from Graphein.graphein.rna.features.sequence.embeddings import (fm_residue_embedding, 
                                                                lucaone_residue_embedding, 
                                                                protrna_residue_embedding,
                                                                ernierna_residue_embedding,
                                                                rnaernie_residue_embedding,
                                                                rnabert_residue_embedding,
                                                                rinalmo_residue_embedding,
                                                                rnamsm_residue_embedding)
from Graphein.graphein.rna.features.nucleotide import nucleotide_one_hot

from RNABind.graphs.geometry import _rbf, _positional_embeddings
from RNABind.utils import keep_keys_in_dict


# configurations for constructing rna graphs
new_edge_funcs = {"edge_construction_functions": [partial(gr.add_k_nn_edges, k=1000)]} 
new_granularity = {"granularity": "rna_centroids"} 
new_graph_featurizer = {"graph_metadata_functions": [ernierna_residue_embedding]}
new_node_featurizer = {"node_metadata_functions": [nucleotide_one_hot]} 
params_to_change = {**new_edge_funcs, **new_granularity, **new_graph_featurizer, **new_node_featurizer}
config = gr.RNAGraphConfig(**params_to_change)


def construct_rna_graph(rna_dir=None,
                        assembly_id=None,
                        pdb_code=None) -> Data:
    r"""
    Construct rna graph from a pdb file.
    Here, we implement two types of rna graphs: i.e., gvp-ready and egnn-ready.
    """
    # use pdb file to construct graph
    if pdb_code:
        g = gr.construct_graph(config=config, pdb_code=pdb_code)
        tag = pdb_code + '.pdb'   # just for logging
    else:
        file_path = rna_dir + '/' + assembly_id + '.pdb'
        g = gr.construct_graph(config=config, path=file_path)
        tag = assembly_id + '.pdb'
    g.graph['residue_id'] = g.graph['pdb_df']['residue_id'].values.tolist()

    keys_to_keep = ['nucleotide_one_hot', 'coords', 'edge_index', 'residue_id', 'raw_pdb_df', 'ernierna_embedding']
    keep_keys_in_dict(g.graph, keys_to_keep)
    for _, data in g.nodes(data=True):
        keep_keys_in_dict(data, keys_to_keep)
    for _, _, data in g.edges(data=True):
        keep_keys_in_dict(data, keys_to_keep) 

    # convert to pytorch geometric data
    data = from_networkx(g)
    # remove self loops
    data.edge_index = remove_self_loops(data.edge_index)[0]
    # egnn data
    data = build_rna_data(data)

    return data


def build_rna_data(data: Data):
    with torch.no_grad():
        coords = data.coords  
        residue_id = data.residue_id       
        edge_index = data.edge_index
        
        ##### node features #####
        nucleotide = data.nucleotide_one_hot   # nucleotide type   size: [num_nodes, 4]
      
        E_vectors = coord[edge_index[0]] - coord[edge_index[1]]             
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=16)             
        pos_embedding = _positional_embeddings(edge_index, 16)   

        edge_attr = torch.cat([rbf, pos_embedding], dim=-1)
        
        # convert the values to tensor
        data = Data(x=nucleotide,   
                    edge_index=edge_index,
                    coords=coords,
                    edge_attr=edge_attr,
                    ernierna_embedding=data.ernierna_embedding,
                    # rnaernie_embedding=data.rnaernie_embedding,
                    # rinalmo_embedding=data.rinalmo_embedding,
                    # rnabert_embedding=data.rnabert_embedding,
                    # fm_embedding=data.fm_embedding,
                    # lucaone_embedding=data.lucaone_embedding,
                    # protrna_embedding=data.protrna_embedding,
                    # rnamsm_embedding=data.ernierna_embedding,
                    residue_id=residue_id)  

        return data


if __name__ == '__main__':
    import os
    rna_dir = '~/RNABind/bs_data/rna_pdb'
    assembly_id = os.listdir(rna_dir)[0]

    assembly_id = assembly_id.split('.')[0]
    rna_graph = construct_rna_graph(rna_dir, assembly_id)
    print(rna_graph)
