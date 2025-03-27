# Identifying RNA-small Molecule Binding Sites Using Geometric Deep Learning with Language Models
## Introduction
Accurately predicting RNA-small molecule binding sites is a key pillar for understanding RNA-small molecule interactions.
[RNABind](https://www.sciencedirect.com/science/article/pii/S0022283625000762) is an embedding-informed geometric deep learning framework to infer RNA-small molecule binding sites from RNA structures.

![overview](https://github.com/jaminzzz/RNABind/blob/main/RNABind.png)
Fig.1 The overview of RNABind.
## Requirements
This project is developed using python 3.9.18, and mainly requires the following libraries.
```txt
biopython==1.83
rdkit==2023.9.6
rna-fm==0.1.2 
multimolecule==0.0.5
torch==2.0.0+cu118
torch_geometric==2.4.0
torch_scatter==2.1.2+pt20cu118
pandas==1.5.3  # recommend
transformers==4.43.4  # recommend
```
We use [Graphein](https://github.com/a-r-j/graphein) to construct RNA 3D graph. Since there are some bugs in Graphein when handle with RNA PDBs, we modified the source code of Graphein and put it in our GitHub [repository](https://github.com/jaminzzz/RNABind/tree/main/Graphein). You need first `pip install graphein` to install related package and then `pip uninstall graphein` to discard the raw Graphein package and use our modified version.
## Datasets
We construct the dataset for small molecule binding sites prediction from the [HARIBOSS](https://hariboss.pasteur.cloud/) [dataset](https://hariboss.pasteur.cloud/complexes/?format=csv) (before September 18, 2024). If the distance between any atom in the ligand and any atom in a nucleotide is less than 4.0 Å, the nucleotide is classified as a binding site nucleotide. 

We applied the following filtering rules: removing structures with more than 500 nucleotides and structures containing modified nucleotides. Ultimately, we retained 353 structures for model training and test. The dataset was divided into 129 clusters based on TM-score with a threshold of 0.5, we then performed [four structure splits](https://github.com/jaminzzz/RNABind/blob/main/bs_data/rna_bs_set_1.csv) based on the [clusters](https://github.com/jaminzzz/RNABind/blob/main/bs_data/structure_clusters.npy) to obtain the training set, validation set, and test set. All related data can be founded in the [directory](https://github.com/jaminzzz/RNABind/tree/main/bs_data).
- `Complexes.csv`: the raw dataset downloaded from HARIBOSS.
- `binding_site_dataset.ipynb` the notebook for filtering structures.
- `list`: all valid RNA PDB files' name.
- `rna_pdb`: all valid RNA files.
- `ligand_pdb`: all corresponding valid ligand files.
- `structure_clusters.npy`: RNA clusters index based on [list](https://github.com/jaminzzz/RNABind/blob/main/bs_data/list).
- `rna_bs_set_1.csv`, `rna_bs_set_2.csv`, `rna_bs_set_3.csv`, `rna_bs_set_4.csv`: the splited datasets used in our study.
- `usalign`: the software of [usalign](usalign).

## Embeddings
In this study, we gathered eight single-nucleotide resolution RNA language models (LucaOne, ProtRNA, RiNALMo, ERNIE-RNA, RNAErnie, RNA-MSM, RNA-FM, and RNABERT) to assess their performance in predicting RNA-small molecule binding sites.

The main script for obtaining these embeddings is located in this [file](https://github.com/jaminzzz/RNABind/blob/main/Graphein/graphein/rna/features/sequence/embeddings.py).

- `LucaOne`: we cloned the [LucaOneApp](https://github.com/LucaOne/LucaOneApp) to extract [LucaOne embedding](https://github.com/jaminzzz/RNABind/blob/main/Graphein/graphein/rna/features/sequence/embeddings.py#L166).

- `RNA-FM`: we used the [rna-fm](rna-fm) package to get [RNA-FM embedding](https://github.com/jaminzzz/RNABind/blob/main/Graphein/graphein/rna/features/sequence/embeddings.py#L115).

- `ProtRNA`: we cloned the [ProtRNA](https://github.com/roxie-zhang/ProtRNA) to obtain [ProtRNA embedding](https://github.com/jaminzzz/RNABind/blob/main/Graphein/graphein/rna/features/sequence/embeddings.py#L263). ProtRNA is implemented based on TensorFlow. So, we need `pip install tensorflow==2.14.0`

- `RiNALMo`, `ERNIE-RNA`, `RNAErnie`, `RNA-MSM`, and `RNABERT`: we used [MultiMolecule](https://github.com/DLS5-Omics/multimolecule) package to get [RiNALMo](https://github.com/jaminzzz/RNABind/blob/main/Graphein/graphein/rna/features/sequence/embeddings.py#L487), [ERNIE-RNA](https://github.com/jaminzzz/RNABind/blob/main/Graphein/graphein/rna/features/sequence/embeddings.py#L286), [RNAErnie](https://github.com/jaminzzz/RNABind/blob/main/Graphein/graphein/rna/features/sequence/embeddings.py#L353), [RNA-MSM](https://github.com/jaminzzz/RNABind/blob/main/Graphein/graphein/rna/features/sequence/embeddings.py#L554), and [RNABERT](https://github.com/jaminzzz/RNABind/blob/main/Graphein/graphein/rna/features/sequence/embeddings.py#L420) embeddings.

## Usage
1. `data:` the dataset training.
	- `dataset.py`: data and dataset construction.
2. `graphs:` to construct RNA 3D graph.
    - `geometry.py`: RBF function.
    - `rna_graph.py`: the main script for constructing [pyg](https://pytorch-geometric.readthedocs.io) graph.
3. `models:` the EGNN model for RNABind.
    - `egnn.py`: main [EGNN model](https://github.com/vgsatorras/egnn/blob/main/models/egnn_clean/egnn_clean.py).
    - `model.py`: RNABind model builder.
4. `utils`: utilities for RNABind.
    - `chem_utils.py`: for binding sites extraction.
    - `structure_split.py`: for RNA strctures clustering.
    - `utils.py`: other utils, such as seed, metric...


Taking the ERNIE-RNA embedding as an example, experiment can be run via:
```shell
cd RNABind/RNABind/train
python binding_sites.py --embedding_type ernierna --dataset_number 1
```
> **Note:** If there are any path-related errors, please check carefully.

## Cite
References to cite when you use RNABind in a research paper:
```txt
@article{zhu2025rnabind,
  title={Identifying RNA-small molecule binding sites using geometric deep learning with language models},
  author={Zhu, Weimin and Ding, Xiaohan and Shen, Hong-Bin and Pan, Xiaoyong},
  journal={Journal of Molecular Biology},
  pages={169010},
  year={2025},
  publisher={Elsevier}
}
```


