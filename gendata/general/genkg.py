
import os, pickle, argparse
from datetime import date, datetime
from collections import defaultdict, OrderedDict

from matplotlib.pyplot import table

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_name', default='FB15k-237', type=str)
    parser.add_argument('--save_path', default='/data/zhaohan/adv-reasoning/save/data/wnkg', type=str)

    return parser.parse_args(args)


def genkg_general(args):
    if args.kg_name in ['NELL-995', 'FB15k-237']:
        rawpath = os.path.join('/data/zhaohan/knowledge-stealing/', args.kg_name, 'tasks')

        ents = set()
        facts = set()
        facts_bytype = defaultdict(set)

        for root, dirs, files in os.walk(rawpath, topdown=True):
            if 'graph.txt' in files:
                with open(os.path.join(root, 'graph.txt')) as _f:
                    for l in _f.readlines():
                        h, r, t = l.strip().split('\t')
                        ents.add(h)
                        ents.add(t)
                        facts.add((h, r, t))
                        facts_bytype[r].add((h, r, t))

        ents_bytype = defaultdict(set)
        ents_bytype['dummy_type'] = ents

    elif args.kg_name in ['WN18', 'WN18RR']:
        rawpath = os.path.join('/data/zhaohan/KG-raw/', args.kg_name, 'text')

        ents = set()
        facts = set()
        facts_bytype = defaultdict(set)

        for root, dirs, files in os.walk(rawpath, topdown=True):
            for file in files:
                with open(os.path.join(root, file)) as _f:
                    for l in _f.readlines():
                        h, r, t = l.strip().split('\t')
                        ents.add(h)
                        ents.add(t)
                        facts.add((h, r, t))
                        facts_bytype[r].add((h, r, t))

        ents_bytype = defaultdict(set)
        ents_bytype['dummy_type'] = ents
    else:
        raise NotImplementedError('not implement this KG')
        
   

    return ents, facts, ents_bytype, facts_bytype

def assign_id(ents, facts, ents_bytype, facts_bytype):
    ent2id, id2ent = OrderedDict(), OrderedDict()
    rel2id, id2rel = OrderedDict(), OrderedDict()
    id_ents, id_facts = set(), set()
    id_ents_bytype = defaultdict(set)
    id_facts_bytype = defaultdict(set)

    for e in ents:
        e_id = len(ent2id)
        ent2id[e] = e_id
        id2ent[e_id] = e
        id_ents.add(e_id)
    
    for _k, _ents in ents_bytype.items():
        for _e in _ents:
            id_ents_bytype[_k].add(ent2id[_e])

    for r, facts in facts_bytype.items():
        r_id = len(rel2id)
        rel2id[r] = r_id
        id2rel[r_id] = r

        for h, r, t in facts:
            h_id, t_id = ent2id[h], ent2id[t]
            id_facts.add((h_id, r_id, t_id))
            id_facts_bytype[r_id].add((h_id, r_id, t_id))

    return ent2id, id2ent, rel2id, id2rel, id_ents, id_facts, id_ents_bytype, id_facts_bytype


if __name__ == '__main__':
    args = parse_args()
    ents, facts, ents_bytype, facts_bytype = genkg_general(args)
    ent2id, id2ent, rel2id, id2rel, id_ents, id_facts, id_ents_bytype, id_facts_bytype = assign_id(ents, facts, ents_bytype, facts_bytype)

    os.makedirs(args.save_path, exist_ok=True)
    # with open(os.path.join(args.save_path, 'entset.pkl'), 'wb') as pklfile:
    #     pickle.dump(ents, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    #     print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    # with open(os.path.join(args.save_path, 'id_entset.pkl'), 'wb') as pklfile:
    #     pickle.dump(id_ents, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    #     print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
    
    # with open(os.path.join(args.save_path, 'facts.pkl'), 'wb') as pklfile:
    #     pickle.dump(facts, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    #     print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    # with open(os.path.join(args.save_path, 'id_facts.pkl'), 'wb') as pklfile:
    #     pickle.dump(id_facts, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    #     print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    with open(os.path.join(args.save_path, 'entset.pkl'), 'wb') as pklfile:
        pickle.dump(ents_bytype, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    with open(os.path.join(args.save_path, 'id_entset.pkl'), 'wb') as pklfile:
        pickle.dump(id_ents_bytype, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
    
    with open(os.path.join(args.save_path, 'factset.pkl'), 'wb') as pklfile:
        pickle.dump(facts_bytype, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    with open(os.path.join(args.save_path, 'id_factset.pkl'), 'wb') as pklfile:
        pickle.dump(id_facts_bytype, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    with open(os.path.join(args.save_path, 'ent2id.pkl'), 'wb') as pklfile:
        pickle.dump(ent2id, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    with open(os.path.join(args.save_path, 'id2ent.pkl'), 'wb') as pklfile:
        pickle.dump(id2ent, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    with open(os.path.join(args.save_path, 'rel2id.pkl'), 'wb') as pklfile:
        pickle.dump(rel2id, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    with open(os.path.join(args.save_path, 'id2rel.pkl'), 'wb') as pklfile:
        pickle.dump(id2rel, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    
    with open(os.path.join(args.save_path, 'stats.txt'), 'w') as txtfile:
        txtfile.write('----- Entity detailed info -----\n')
        txtfile.write('Total entities: %d\n' % len(ents))
        txtfile.write('\n----- Facts detailed info -----\n')
        for k, v in facts_bytype.items():
            txtfile.write('%s: %d\n' % (k, len(v)))
        txtfile.write('Total facts: %d\n' % len(facts))
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), txtfile.name))


    # python genkg.py --kg_name NELL-995
    # python genkg.py --kg_name FB15k-237
    # python genkg.py --kg_name WN18RR