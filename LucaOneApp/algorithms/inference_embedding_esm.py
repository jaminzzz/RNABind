#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/12/8 15:10
@project: LucaOneApp
@file: inference_embedding_esm.py
@desc: inference embedding of ESM2-3B
'''

import sys
import argparse
sys.path.append(".")
sys.path.append("..")
sys.path.append("../algorithms")
try:
    from .llm.esm.predict_embedding import main
except ImportError:
    from algorithms.llm.esm.predict_embedding import main


def get_args():
    parser = argparse.ArgumentParser(description='ESM2 Embedding')
    parser.add_argument("--llm_type", type=str, default="esm2", choices=["esm2"],  help="the llm type")
    parser.add_argument("--llm_version", type=str, default="3B", choices=["3B"], help="the llm version")
    parser.add_argument("--embedding_type", type=str, default="matrix", choices=["matrix", "vector"], help="the llm embedding type.")
    parser.add_argument("--trunc_type", type=str, default="right", choices=["left", "right"], help="llm trunc type.")
    parser.add_argument("--truncation_seq_length", type=int, default=4094, help="the llm truncation seq length(not contain [CLS] and [SEP].")
    parser.add_argument("--matrix_add_special_token", action="store_true", help="whether to add special token embedding vector in seq representation matrix")
    parser.add_argument("--input_file", type=str, default=None, help="the input filepath(.fasta or .csv)")
    parser.add_argument("--seq", type=str, default=None, help="when to input a seq")
    parser.add_argument("--seq_type", type=str, default=None, required=True, choices=["gene", "prot"], help="the input seq type")
    parser.add_argument("--save_path", type=str, default=None, help="embedding file save dir path")
    parser.add_argument("--id_idx", type=int, default=None, help="id col idx(0 start)")
    parser.add_argument("--seq_idx", type=int, default=None, help="seq col idx(0 start)")
    parser.add_argument("--embedding_complete",  action="store_true", help="when the seq len > inference_max_len, then the embedding matrix is completed by segment")
    parser.add_argument('--gpu', type=int, default=-1, help="the gpu id to use.")

    input_args = parser.parse_args()
    return input_args


if __name__ == "__main__":
    input_args = get_args()
    main(input_args)