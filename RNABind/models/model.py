import torch
from torch import nn
from RNABind.models.egnn import EGNN


#-----------------------------------------------------------------------------
# Binding site prediction
# -----------------------------------------------------------------------------
class BindingSiteModel(nn.Module):
    def __init__(self,
                 task: str = 'rl_binding_site',
                 embedding_type: str = 'one_hot',
                 in_node_nf: int = 4,
                 hidden_nf: int = 128,
                 out_node_nf: int = 128, 
                 in_edge_nf: int = 0,
                 n_layers: int = 3,
                 ) -> None:
        super(BindingSiteModel, self).__init__()
  
        self.task = task
        self.embedding_type = embedding_type
        self.in_edge_nf = in_edge_nf

        if self.embedding_type in {'one_hot', 'one-hot', 'onehot'}:
            self.embeddings = nn.Embedding(24, in_node_nf)  # 20 amino acids + 4 nucleotides

        self.recptor_model = EGNN(in_node_nf, 
                                  hidden_nf, 
                                  out_node_nf, 
                                  in_edge_nf=in_edge_nf, 
                                  n_layers=n_layers)
        
        self.predictor = nn.Linear(out_node_nf, 1)         # binary classification
        
    def forward(self, data):
        if self.embedding_type in {'one_hot', 'one-hot', 'onehot'}:
            x = data.x.argmax(dim=1)
            x = self.embeddings(x)
        elif self.embedding_type == 'lucaone':
            x = data.lucaone_embedding
        elif self.embedding_type == 'esm':
            x = data.esm_embedding
        elif self.embedding_type == 'protrna':
            x = data.protrna_embedding
        elif self.embedding_type in {'rna-fm', 'rna-fm', 'fm'}:
            x = data.fm_embedding
        elif self.embedding_type == 'rnabert':
            x = data.rnabert_embedding
        elif self.embedding_type == 'rnaernie':
            x = data.rnaernie_embedding
        elif self.embedding_type == 'ernierna':
            x = data.ernierna_embedding
        elif self.embedding_type == 'rinalmo':
            x = data.rinalmo_embedding
        elif self.embedding_type == 'rnamsm':
            x = data.rnamsm_embedding
        else:
            raise ValueError(f"Invalid embedding type: {self.embedding_r}")
        
        edge_index = data.edge_index
        coord = data.coord.float()
        edge_attr = data.edge_attr.float()[:, 0:16]

        # encode the structure of rna
        embedings = self.recptor_model(x, coord, edge_index, edge_attr)
        # predict the binding site per residue, 0/1
        output = self.predictor(embedings)
        output = torch.sigmoid(output)

        return output

# -----------------------------------------------------------------------------
# Build model: from pretrained model or from scratch
# -----------------------------------------------------------------------------
def build_model(args):
    if args.task == 'rl_binding_site':
        model = BindingSiteModel(args.task, args.embedding_type, args.in_node_nf, args.hidden_nf,
                                 args.out_node_nf, args.in_edge_nf, args.n_layers)   
    else:
        raise ValueError(f"Invalid task: {args.task}")
        
    return model
