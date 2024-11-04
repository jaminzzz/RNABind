"""Functions to add embeddings from pre-trained language models RNA structure graphs."""
# modified from: Graphein/graphein/protein/features/sequence/embeddings.py
from __future__ import annotations

from functools import lru_cache, partial

import networkx as nx
import numpy as np

from Graphein.graphein.rna.features.sequence.utils import (
    compute_feature_over_chains,
    subset_by_node_feature_value,
)

import torch
import fm

from LucaOneApp.algorithms.llm.lucagplm.get_embedding import main


@lru_cache()
def _load_fm_model(model_name: str = "rna_fm_t12"):
    """
    Loads pre-trained RNA-FM model from torch hub.

    :param model_name: Name of pre-trained model to load
    :type model_name: str
    :return: loaded pre-trained model
    """

    return fm.pretrained.load_rnafm_model_and_alphabet_hub(model_name)


def compute_fm_embedding(
    sequence: str,
    representation: str,
    model_name: str = "rna_fm_t12",
    output_layer: int = 12,
) -> np.ndarray:
    """
    Computes sequence embedding using Pre-trained RNA-FM model.

    :param sequence: RNA sequence to embed (str)
    :type sequence: str
    :param representation: Type of embedding to extract. ``"residue"`` or
        ``"sequence"``. Sequence-level embeddings are averaged residue
        embeddings
    :type representation: str
    :param model_name: Name of pre-trained model to use
    :type model_name: str
    :param output_layer: integer indicating which layer the output should be
        taken from.
    :type output_layer: int
    :return: embedding (``np.ndarray``)
    :rtype: np.ndarray
    """
    model, alphabet = _load_fm_model(model_name)
    batch_converter = alphabet.get_batch_converter()

    data = [
        ("RNA1", sequence),
    ]
    
    # deal with long sequences that exceed the model's max length
    # not handling sequence-level embedding since it's not used
    # ---------------------
    # split long sequences into chunks
    max_length = 1022  # delete 2 tokens for start and end tokens
    new_data=[]
    if len(sequence) > max_length:
        for i in range(len(sequence) // max_length):
            new_data.append((id, sequence[i * max_length:(i+1) * max_length]))
        new_data.append((id, sequence[(i+1) * max_length:]))
    else:
        new_data.append((id, sequence))

    embeddings=torch.tensor([])
    with torch.no_grad():
        for rna_id, rna_sequence in new_data:
            batch_labels, batch_strs, batch_tokens = batch_converter([(rna_id, rna_sequence)])
            results = model(batch_tokens, repr_layers=[output_layer])
            token_embeddings = results["representations"][output_layer].squeeze(0).detach()[1:-1]

            if embeddings.shape == 0:
                embeddings = token_embeddings
            else:
                embeddings = torch.cat([embeddings, token_embeddings], dim=0)

    token_representations = embeddings
    # --------------------------

    # batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # # Extract per-residue representations (on CPU)
    # with torch.no_grad():
    #     results = model(
    #         batch_tokens, repr_layers=[output_layer], return_contacts=True
    #     )
    # token_representations = results["representations"][output_layer]

    if representation == "residue":
        return token_representations.numpy()

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first
    # residue is token 1.
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )
        return sequence_representations[0].numpy()


def fm_residue_embedding(
    G: nx.Graph,
    model_name: str = "rna_fm_t12",
    output_layer: int = 12,
) -> nx.Graph:
    """
    Computes RNA-FM residue embeddings from a RNA sequence and adds the to the graph.

    :param G: ``nx.Graph`` to add fm embedding to.
    :type G: nx.Graph
    :param model_name: Name of pre-trained model to use.
    :type model_name: str
    :param output_layer: index of output layer in pre-trained model.
    :type output_layer: int
    :return: ``nx.Graph`` with fm embedding feature added to nodes.
    :rtype: nx.Graph
    """

    for chain in G.graph["chain_ids"]:
        embedding = compute_fm_embedding(
            G.graph[f"sequence_{chain}"],
            representation="residue",
            model_name=model_name,
            output_layer=output_layer,
        )
        # remove start and end tokens from per-token residue embeddings
        # embedding = embedding[0, 1:-1]
        subgraph = subset_by_node_feature_value(G, "chain_id", chain)

        for i, (n, d) in enumerate(subgraph.nodes(data=True)):
            G.nodes[n]["fm_embedding"] = embedding[i]

    return G


def fm_sequence_embedding(G: nx.Graph) -> nx.Graph:
    """
    Computes RNA-FM sequence embedding feature over chains in a graph.

    :param G: nx.Graph RNA structure graph.
    :type G: nx.Graph
    :return: nx.Graph RNA structure graph with fm embedding features added
        eg. ``G.graph["fm_embedding_A"]`` for chain A.
    :rtype: nx.Graph
    """
    func = partial(compute_fm_embedding, representation="sequence")
    G = compute_feature_over_chains(G, func, feature_name="fm_embedding")

    return G

#-------------------------------------------------------------------------------------------------------
def lucaone_residue_embedding(
    G: nx.Graph,
) -> nx.Graph:
    for chain in G.graph["chain_ids"]:
        max_length = 1024
        args = Args()
        sequence = G.graph[f"sequence_{chain}"]
        if len(sequence) > max_length:
            num_chunks = len(sequence) // max_length + (len(sequence) % max_length != 0)
            embeddings = []
            for i in range(num_chunks):
                start = i * max_length
                end = min((i + 1) * max_length, len(sequence))
                args.seq = sequence[start:end]
                embedding = main(args)
                embeddings.append(embedding)

            embedding = np.concatenate(embeddings, axis=0)
        else:
            args.seq = sequence
            embedding = main(args)

        subgraph = subset_by_node_feature_value(G, "chain_id", chain)

        for i, (n, d) in enumerate(subgraph.nodes(data=True)):
            G.nodes[n]["lucaone_embedding"] = embedding[i]

    return G


class Args:
    def __init__(self):
        self.llm_dir = "/amax/wmzhu/RNABind/LucaOneApp/models/"  # please change to your own path
        self.llm_type = "lucaone_gplm"
        self.llm_version = "v2.0"
        self.llm_task_level = "token_level,span_level,seq_level,structure_level"
        self.llm_time_str = 20231125113045
        self.llm_step = 5600000 
        self.embedding_type = "matrix"
        self.trunc_type = "right"
        self.truncation_seq_length = 100000 
        self.matrix_add_special_token = False
        self.input_file = None
        self.seq = 'ATCGATCGATCG'
        self.seq_type = 'gene'
        self.save_path = None
        self.embedding_complete = True
        self.embedding_complete_seg_overlap = True
        self.gpu = 2
#-------------------------------------------------------------------------------------------------------

def compute_protrna_embedding(
    sequence: str,
    representation: str,
    model_name: str = "rna_protrna_t33",
    output_layer: int = 33,
) -> np.ndarray:
    from ProtRNA.protrna.pretrained import load_model
    import tensorflow as tf
    model = load_model()
    batch_converter = model.alphabet.get_batch_converter()

    data = [
        ("RNA1", sequence),
    ]
    max_length = 1022  # delete 2 tokens for start and end tokens
    new_data=[]
    if len(sequence) > max_length:
        for i in range(len(sequence) // max_length):
            new_data.append((id, sequence[i * max_length:(i+1) * max_length]))
        new_data.append((id, sequence[(i+1) * max_length:]))
    else:
        new_data.append((id, sequence))

    embeddings = None
    for rna_id, rna_sequence in new_data:
        seq_tokens = batch_converter([rna_sequence])
        results = model(seq_tokens, repr_layers=[output_layer])
        token_embeddings = tf.squeeze(results["representations"][output_layer])[1:-1]

        if embeddings is None:
            embeddings = token_embeddings
        else:
            embeddings = tf.concat([embeddings, token_embeddings], axis=0)

    token_representations = embeddings
    if representation == "residue":
        return token_representations.numpy()
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )
        return sequence_representations[0].numpy()


def protrna_residue_embedding(
    G: nx.Graph,
    model_name: str = "rna_protrna_t33",
    output_layer: int = 33,
) -> nx.Graph:
    for chain in G.graph["chain_ids"]:
        embedding = compute_protrna_embedding(
            G.graph[f"sequence_{chain}"],
            representation="residue",
            model_name=model_name,
            output_layer=output_layer,
        )
        # remove start and end tokens from per-token residue embeddings
        # embedding = embedding[0, 1:-1]
        subgraph = subset_by_node_feature_value(G, "chain_id", chain)

        for i, (n, d) in enumerate(subgraph.nodes(data=True)):
            G.nodes[n]["protrna_embedding"] = embedding[i]

    return G
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
def compute_ernierna_embedding(
    sequence: str,
    representation: str) -> np.ndarray:
    from multimolecule import RnaTokenizer, ErnieRnaModel
    tokenizer = RnaTokenizer.from_pretrained('/amax/wmzhu/RNABind/multimolecule/ernierna')  # please change to your own path
    model = ErnieRnaModel.from_pretrained('/amax/wmzhu/RNABind/multimolecule/ernierna')     # please change to your own path

    data = [("RNA1", sequence)]
    
    # deal with long sequences that exceed the model's max length
    # not handling sequence-level embedding since it's not used
    # ---------------------
    # split long sequences into chunks
    max_length = 1022  # delete 2 tokens for start and end tokens
    new_data=[]
    if len(sequence) > max_length:
        for i in range(len(sequence) // max_length):
            new_data.append((id, sequence[i * max_length:(i+1) * max_length]))
        new_data.append((id, sequence[(i+1) * max_length:]))
    else:
        new_data.append((id, sequence))

    embeddings=torch.tensor([])
    with torch.no_grad():
        for rna_id, rna_sequence in new_data:
            input = tokenizer(rna_sequence, return_tensors='pt')
            results = model(**input)
            token_embeddings = results.last_hidden_state.squeeze(0).detach()[1:-1]

            if embeddings.shape == 0:
                embeddings = token_embeddings
            else:
                embeddings = torch.cat([embeddings, token_embeddings], dim=0)

    token_representations = embeddings

    if representation == "residue":
        return token_representations.numpy()

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first
    # residue is token 1.
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )
        return sequence_representations[0].numpy()


def ernierna_residue_embedding(
    G: nx.Graph) -> nx.Graph:
    for chain in G.graph["chain_ids"]:
        embedding = compute_ernierna_embedding(
            G.graph[f"sequence_{chain}"],
            representation="residue",
        )
        subgraph = subset_by_node_feature_value(G, "chain_id", chain)

        for i, (n, d) in enumerate(subgraph.nodes(data=True)):
            G.nodes[n]["ernierna_embedding"] = embedding[i]

    return G
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
def compute_rnaernie_embedding(
    sequence: str,
    representation: str) -> np.ndarray:
    from multimolecule import RnaTokenizer, RnaErnieModel
    tokenizer = RnaTokenizer.from_pretrained('/amax/wmzhu/RNABind/multimolecule/rnaernie')  # please change to your own path
    model = RnaErnieModel.from_pretrained('/amax/wmzhu/RNABind/multimolecule/rnaernie')     # please change to your own path

    data = [("RNA1", sequence)]
    
    # deal with long sequences that exceed the model's max length
    # not handling sequence-level embedding since it's not used
    # ---------------------
    # split long sequences into chunks
    max_length = 1022  # delete 2 tokens for start and end tokens
    new_data=[]
    if len(sequence) > max_length:
        for i in range(len(sequence) // max_length):
            new_data.append((id, sequence[i * max_length:(i+1) * max_length]))
        new_data.append((id, sequence[(i+1) * max_length:]))
    else:
        new_data.append((id, sequence))

    embeddings=torch.tensor([])
    with torch.no_grad():
        for rna_id, rna_sequence in new_data:
            input = tokenizer(rna_sequence, return_tensors='pt')
            results = model(**input)
            token_embeddings = results.last_hidden_state.squeeze(0).detach()[1:-1]

            if embeddings.shape == 0:
                embeddings = token_embeddings
            else:
                embeddings = torch.cat([embeddings, token_embeddings], dim=0)

    token_representations = embeddings

    if representation == "residue":
        return token_representations.numpy()

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first
    # residue is token 1.
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )
        return sequence_representations[0].numpy()


def rnaernie_residue_embedding(
    G: nx.Graph) -> nx.Graph:
    for chain in G.graph["chain_ids"]:
        embedding = compute_rnaernie_embedding(
            G.graph[f"sequence_{chain}"],
            representation="residue",
        )
        subgraph = subset_by_node_feature_value(G, "chain_id", chain)

        for i, (n, d) in enumerate(subgraph.nodes(data=True)):
            G.nodes[n]["rnaernie_embedding"] = embedding[i]

    return G
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
def compute_rnabert_embedding(
    sequence: str,
    representation: str) -> np.ndarray:
    from multimolecule import RnaTokenizer, RnaBertModel
    tokenizer = RnaTokenizer.from_pretrained('/amax/wmzhu/RNABind/multimolecule/rnabert')   # please change to your own path
    model = RnaBertModel.from_pretrained('/amax/wmzhu/RNABind/multimolecule/rnabert')       # please change to your own path

    data = [("RNA1", sequence)]
    
    # deal with long sequences that exceed the model's max length
    # not handling sequence-level embedding since it's not used
    # ---------------------
    # split long sequences into chunks
    max_length = 1022  # delete 2 tokens for start and end tokens
    new_data=[]
    if len(sequence) > max_length:
        for i in range(len(sequence) // max_length):
            new_data.append((id, sequence[i * max_length:(i+1) * max_length]))
        new_data.append((id, sequence[(i+1) * max_length:]))
    else:
        new_data.append((id, sequence))

    embeddings=torch.tensor([])
    with torch.no_grad():
        for rna_id, rna_sequence in new_data:
            input = tokenizer(rna_sequence, return_tensors='pt')
            results = model(**input)
            token_embeddings = results.last_hidden_state.squeeze(0).detach()[1:-1]

            if embeddings.shape == 0:
                embeddings = token_embeddings
            else:
                embeddings = torch.cat([embeddings, token_embeddings], dim=0)

    token_representations = embeddings

    if representation == "residue":
        return token_representations.numpy()

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first
    # residue is token 1.
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )
        return sequence_representations[0].numpy()


def rnabert_residue_embedding(
    G: nx.Graph) -> nx.Graph:
    for chain in G.graph["chain_ids"]:
        embedding = compute_rnabert_embedding(
            G.graph[f"sequence_{chain}"],
            representation="residue",
        )
        subgraph = subset_by_node_feature_value(G, "chain_id", chain)

        for i, (n, d) in enumerate(subgraph.nodes(data=True)):
            G.nodes[n]["rnabert_embedding"] = embedding[i]

    return G
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
def compute_rinalmo_embedding(
    sequence: str,
    representation: str) -> np.ndarray:
    from multimolecule import RnaTokenizer, RiNALMoModel
    tokenizer = RnaTokenizer.from_pretrained('/amax/wmzhu/RNABind/multimolecule/rinalmo')   # please change to your own path
    model = RiNALMoModel.from_pretrained('/amax/wmzhu/RNABind/multimolecule/rinalmo')       # please change to your own path

    data = [("RNA1", sequence)]
    
    # deal with long sequences that exceed the model's max length
    # not handling sequence-level embedding since it's not used
    # ---------------------
    # split long sequences into chunks
    max_length = 1022  # delete 2 tokens for start and end tokens
    new_data=[]
    if len(sequence) > max_length:
        for i in range(len(sequence) // max_length):
            new_data.append((id, sequence[i * max_length:(i+1) * max_length]))
        new_data.append((id, sequence[(i+1) * max_length:]))
    else:
        new_data.append((id, sequence))

    embeddings=torch.tensor([])
    with torch.no_grad():
        for rna_id, rna_sequence in new_data:
            input = tokenizer(rna_sequence, return_tensors='pt')
            results = model(**input)
            token_embeddings = results.last_hidden_state.squeeze(0).detach()[1:-1]

            if embeddings.shape == 0:
                embeddings = token_embeddings
            else:
                embeddings = torch.cat([embeddings, token_embeddings], dim=0)

    token_representations = embeddings

    if representation == "residue":
        return token_representations.numpy()

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first
    # residue is token 1.
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )
        return sequence_representations[0].numpy()


def rinalmo_residue_embedding(
    G: nx.Graph) -> nx.Graph:
    for chain in G.graph["chain_ids"]:
        embedding = compute_rinalmo_embedding(
            G.graph[f"sequence_{chain}"],
            representation="residue",
        )
        subgraph = subset_by_node_feature_value(G, "chain_id", chain)

        for i, (n, d) in enumerate(subgraph.nodes(data=True)):
            G.nodes[n]["rinalmo_embedding"] = embedding[i]

    return G
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
def compute_rnamsm_embedding(
    sequence: str,
    representation: str) -> np.ndarray:
    from multimolecule import RnaTokenizer, RnaMsmModel
    tokenizer = RnaTokenizer.from_pretrained('/amax/wmzhu/RNABind/multimolecule/rnamsm')    # please change to your own path
    model = RnaMsmModel.from_pretrained('/amax/wmzhu/RNABind/multimolecule/rnamsm')         # please change to your own path

    data = [("RNA1", sequence)]
    
    # deal with long sequences that exceed the model's max length
    # not handling sequence-level embedding since it's not used
    # ---------------------
    # split long sequences into chunks
    max_length = 1022  # delete 2 tokens for start and end tokens
    new_data=[]
    if len(sequence) > max_length:
        for i in range(len(sequence) // max_length):
            new_data.append((id, sequence[i * max_length:(i+1) * max_length]))
        new_data.append((id, sequence[(i+1) * max_length:]))
    else:
        new_data.append((id, sequence))

    embeddings=torch.tensor([])
    with torch.no_grad():
        for rna_id, rna_sequence in new_data:
            input = tokenizer(rna_sequence, return_tensors='pt')
            results = model(**input)
            token_embeddings = results.last_hidden_state.squeeze(0).detach()[1:-1]

            if embeddings.shape == 0:
                embeddings = token_embeddings
            else:
                embeddings = torch.cat([embeddings, token_embeddings], dim=0)

    token_representations = embeddings

    if representation == "residue":
        return token_representations.numpy()

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first
    # residue is token 1.
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )
        return sequence_representations[0].numpy()


def rnamsm_residue_embedding(
    G: nx.Graph) -> nx.Graph:
    for chain in G.graph["chain_ids"]:
        embedding = compute_rnamsm_embedding(
            G.graph[f"sequence_{chain}"],
            representation="residue",
        )
        subgraph = subset_by_node_feature_value(G, "chain_id", chain)

        for i, (n, d) in enumerate(subgraph.nodes(data=True)):
            G.nodes[n]["rnamsm_embedding"] = embedding[i]

    return G
#-------------------------------------------------------------------------------------------------------