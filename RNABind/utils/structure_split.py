import os
import numpy as np
import argparse as ap
from sklearn.cluster import AgglomerativeClustering


def get_tmscore(backbone_dir):
    if not os.path.exists('usalign'):
        list = '\n'.join(os.listdir(backbone_dir))
        with open('list', 'w') as f:
            f.write(list)
        
        # https://zhanggroup.org/US-align/help/
        cmd = f"/amax/wmzhu/USalign -dir {backbone_dir} list -outfmt 2 -ter 1"   # please change to your own path
        out = os.popen(cmd)
        out = out.read()
        with open('usalign', 'w') as f:
            f.write(out)

def cluster(threshold):
    if not os.path.exists('structure_clusters.npy'):
        with open('list', 'r') as f:
            backbone_list = f.read().split()
        
        length = len(backbone_list)
        tmscore_matrix = np.ones((length, length))

        with open('usalign', 'r') as f:
            for line in f:
                if line[0] == '/':
                    line = line.split()
                    pdb1 = line[0][1:].split(':')[0]
                    pdb2 = line[1][1:].split(':')[0]
                    idx1 = backbone_list.index(pdb1)
                    idx2 = backbone_list.index(pdb2)

                    tm1 = float(line[2])
                    tm2 = float(line[3])
                    tmscore_matrix[idx1, idx2] = tm1
                    tmscore_matrix[idx2, idx1] = tm2
        
        tmscore_matrix = (tmscore_matrix + tmscore_matrix.T) / 2

        threshold = 1 - threshold
        dis = 1 - tmscore_matrix

        clustering = AgglomerativeClustering(
            n_clusters=None, 
            metric='precomputed', 
            linkage='single', 
            distance_threshold=threshold, 
        )
        clustering.fit(dis)
        np.save('structure_clusters.npy', clustering.labels_)

def main(args):
    get_tmscore(args.backbone_dir)
    cluster(args.threshold)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--backbone_dir', type=str, default='~/RNABind/bs_data/no_mdf_rna_pdb')  # please change to your own path
    args = parser.parse_args()
    main(args)

