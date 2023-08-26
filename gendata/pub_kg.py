# whatever the \kg is, generate public kg for stealing use

import os
import shutil
import pickle
import random
import argparse
import numpy as np
from datetime import date, datetime
from collections import defaultdict


def privatize_kg(args):
    entset = pickle.load(open(os.path.join(args.kg_path, 'id_entset.pkl'), 'rb'))
    factset = pickle.load(open(os.path.join(args.kg_path, 'id_factset.pkl'), 'rb'))
    id2rel = pickle.load(open(os.path.join(args.kg_path, 'id2rel.pkl'), 'rb'))

    cdd_ents = set()
    for cate, ents in entset.items():
        if args.prv_cate is not None and len(args.prv_cate) > 0:
            if cate in args.prv_cate:
                cdd_ents |= ents
        else:
            cdd_ents |= ents

    random.seed(args.seed)
    prv_ents = random.sample(list(cdd_ents), int(np.ceil(args.prv_ratio * len(cdd_ents))))
    # TODO: consider choosing denser/sparser entities

    for f in [
        'ent2id.pkl',
        'id2ent.pkl',
        'id2rel.pkl', 
        'rel2id.pkl', 
        'entid2cate.pkl',
        ]:
        shutil.copyfile(os.path.join(args.kg_path, f), os.path.join(args.pubkg_path, f))
    
    entset_pub = defaultdict(set)
    for cate, ents in entset.items():
        entset_pub[cate] = ents - set(prv_ents)

    factset_pub = defaultdict(set)
    for rid, facts in factset.items():
        for h, r, t in facts:
            if h not in prv_ents and t not in prv_ents:
                factset_pub[rid].add((h, r, t))

    # clean facts not in entset 
    all_ent = set()
    for k, v in entset_pub.items():
        all_ent = all_ent | v
    for k, v in factset_pub.items():
        to_pop = set()
        for h, r, t in v:
            if (h not in all_ent) or (t not in all_ent):
                to_pop.add(f)
        factset_pub[k] = factset_pub[k] - to_pop

    # clean isolated nodes
    all_ent = set()
    for rid in factset_pub:
        for h, r, t in factset_pub[rid]:
            all_ent.add(h)
            all_ent.add(t)
    for cate in entset_pub:
        entset_pub[cate] = entset_pub[cate] & all_ent

    for _fname, _f in [ ('id_entset.pkl', entset_pub),                       # int id
                        ('id_factset.pkl', factset_pub),                     # int id
                    ]:        
        with open(os.path.join(args.pubkg_path, _fname), 'wb') as pklfile:
            pickle.dump(_f, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    with open(os.path.join(args.pubkg_path, 'detail_stats_pub.txt'), 'w') as txtfile:
        txtfile.write('----- Entity detailed info -----\n')
        ent_num = 0
        for k, v in entset_pub.items():
            txtfile.write('%s: %d\n' % (k, len(v)))
            ent_num += len(v)
        txtfile.write('Total entities: %d\n' % ent_num)
        fact_num = 0
        txtfile.write('\n----- Facts detailed info -----\n')
        for k, v in factset_pub.items():
            txtfile.write('%s: %d\n' % (id2rel[k], len(v)))
            fact_num += len(v)
        txtfile.write('Total facts: %d\n' % fact_num)
        print('%s %s Saved %s\n' % (date.today(), datetime.now().strftime("%H:%M:%S"), txtfile.name))


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_path', default=None, type=str, help='load kg path')
    parser.add_argument('--pubkg_path', default=None, type=str, help='save path for generated kg')
    parser.add_argument('--prv_cate', nargs='+', default=[], type=str, help="a list of entity categories that we aim to sample private entity from")
    parser.add_argument('--prv_ratio',  default=0.05, type=float, help="ratio of private entities")

    parser.add_argument('--seed', default=0, type=int, help="random seed")
    
    return parser.parse_args(args)


if __name__=='__main__':
    args = parse_args()
    assert args.kg_path is not None
    assert args.pubkg_path is not None
    os.makedirs(args.pubkg_path, exist_ok=True)

    privatize_kg(args)


    # python pub_kg.py --kg_path ../save/data/cyberkg --pubkg_path ../save/data/cyberkg_pub 
