from email.policy import default
from functools import total_ordering
import os
import csv
import pickle
from collections import defaultdict
from datetime import date, datetime
import gendata.cyber.cyberkg_utils as cyber
import gendata.med.medkg_utils as med


def gen_medkg(args):
    entset = defaultdict(set)
    for row in csv.reader(open(os.path.join(args.raw_path, 'entity2src.tsv')), delimiter="\t"):
        ent_name = row[0]  # row[1:] are sources
        ent_name = med.ent_prefix_delimiter.join(ent_name.split('::'))  # change the org delimiter '::' from DRKG into ours
        cate = ent_name.split(med.ent_prefix_delimiter)[0] 
        entset[cate].add(ent_name)
        

    factset = defaultdict(set)
    for row in csv.reader(open(os.path.join(args.raw_path, 'drkg.tsv')), delimiter="\t"):
        h, r, t  = row

        if r not in med.clean_r_map:
            continue
        clean_r = med.clean_r_map[r]

        # change entity '::' delimiter into ours, dont concern rel delimiter
        h = med.ent_prefix_delimiter.join(h.split('::'))
        t = med.ent_prefix_delimiter.join(t.split('::'))

        kg_src = r.strip().split(':')[0]
        if kg_src not in args.use_src:
            continue

        rev_r = cyber.rev_rel_prefix + clean_r
        assert h in entset[h.split(med.ent_prefix_delimiter)[0]]
        assert t in entset[t.split(med.ent_prefix_delimiter)[0]]
        factset[clean_r].add((h, clean_r, t))

        # In DRKG, if h_cate!=t_cate, the reverse rel isn't in dataset
        h_cate, t_cate = clean_r.strip().split(':')[-2:]
        if h_cate == t_cate:
            factset[clean_r].add((t, clean_r, h))
        else:  # add reverse: ahead for reverse relations we construct
            factset[rev_r].add((t, rev_r, h))

    # clean isolated nodes
    all_ent_with_fact = set()
    for k, facts in factset.items():
        for h, r, t in facts:
            all_ent_with_fact.add(h)
            all_ent_with_fact.add(t)

    clean_entset = defaultdict(set)
    for k, v in entset.items():
        kept_ents = all_ent_with_fact & v
        if len(kept_ents)>0:
            clean_entset[k] |= kept_ents
    entset = clean_entset

    # assign int idx
    ent2id, id2ent = defaultdict(int), defaultdict(int)
    rel2id, id2rel = defaultdict(int), defaultdict(int)
    entid2cate = defaultdict(int)

    id_entset = defaultdict(set)
    for cate, ents in entset.items():
        for e in ents:
            eid = len(ent2id)
            ent2id[e] = eid
            id2ent[eid] = e
            entid2cate[eid] = cate
            id_entset[cate].add(eid)

    id_factset = defaultdict(set)
    for r, facts in factset.items():
        rid = len(rel2id)
        rel2id[r] = rid
        id2rel[rid] = r
        for h, r, t in facts:
            id_factset[rid].add((ent2id[h], rid, ent2id[t]))

    rel_dict = get_rel_dict(rel2id)

    os.makedirs(args.kg_path, exist_ok=True)
    for _fname, _f in [('entset.pkl', entset),
                       ('factset.pkl', factset),
                       ('id_entset.pkl', id_entset),
                       ('id_factset.pkl', id_factset),
                       ('ent2id.pkl', ent2id),
                       ('id2ent.pkl', id2ent),
                       ('rel2id.pkl', rel2id),
                       ('id2rel.pkl', id2rel),
                       ('entid2cate.pkl', entid2cate),
                       ('rel_dict.pkl', rel_dict)
                    ]:   
        with open(os.path.join(args.kg_path, _fname), 'wb') as pklfile:
            pickle.dump(_f, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
    
    # stats
    with open(os.path.join(args.kg_path, 'detailed_stats.txt'), 'w') as txtfile:
        total_ents = 0
        txtfile.write('----- Entity detailed info -----\n')
        for cate, ents in entset.items():
            txtfile.write('%s : %d\n' % (cate, len(ents)))
            total_ents += len(ents)
        txtfile.write('Total entities : %d\n' % total_ents)

        total_facts = 0
        txtfile.write('\n\n----- Facts detailed info -----\n')
        for rel, facts in factset.items():
            txtfile.write('%s : %d\n' % (rel, len(facts)))
            total_facts += len(facts)
        txtfile.write('Total facts : %d\n' % total_facts)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), txtfile.name))

    with open(os.path.join(args.kg_path, 'stats.txt'), 'w') as txtfile:
        txtfile.write('numentity: %d\n' % len(ent2id))
        txtfile.write('numrelations: %d' % len(rel2id))
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), txtfile.name))


def get_rel_dict(rel2id):
    # find all rels for ent:ent pair
    # {'ent_cate:ent_cate' : set(rel1, rel2, rel3,...)} i.e., {str: set(str)} # not like cyber

    rel_dict = defaultdict(set)
    for r_name in rel2id:
        h_cate, t_cate = r_name.split(':')[-2:]
        if r_name.startswith(cyber.rev_rel_prefix):
            h_cate, t_cate = t_cate, h_cate
        rel_dict[h_cate + med.ent_prefix_delimiter + t_cate].add(r_name)
    
    return rel_dict

        
