import os, sys
sys.path.append(os.path.abspath('..'))
import argparse

import copy
import random
import pickle
from tqdm import tqdm
from collections import defaultdict
from datetime import date, datetime
from gendata.cyber.cyberkg_utils import rev_rel_prefix
from helper.utils import set_global_seed

import gendata.med.medkg_backbone as bb

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', default='/data/zhaohan/adv-reasoning/data/drkg/', type=str, 
    help='loading path of raw DRKG information')
    parser.add_argument('--kg_path', default='/data/zhaohan/adv-reasoning/save/data/medkg/', type=str, help='save path for generated medkg and recommendation set')
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--use_src', nargs='+', default=['DRUGBANK', 'GNBR', 'Hetionet'],  # 'bioarx', 
    type=str, help='used DRKG sources for KG construction')

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    set_global_seed(args.seed)

    print('#-------- building med-kg ------#')

    bb.gen_medkg(args)

