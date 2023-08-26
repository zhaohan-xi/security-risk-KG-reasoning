from collections import defaultdict
import os, sys
sys.path.append(os.path.abspath('..'))

import numpy as np
import random
import pickle
import shutil
import gendata.cyber.cyberkg_utils as cyber
import gendata.med.medkg_utils as med
from helper.utils import flatten, query_name_dict
import helper.atkutils as atkutils


def set_target(args):
    benign_kg_path = args.kg_path
    benign_q_path = args.q_path

    rel2id = pickle.load(open(os.path.join(benign_kg_path, 'rel2id.pkl'), 'rb'))
    ent2id = pickle.load(open(os.path.join(benign_kg_path, 'ent2id.pkl'), 'rb'))
    str_entset = pickle.load(open(os.path.join(benign_kg_path, 'entset.pkl'), 'rb'))

    assert args.tar_evi_cate is not None, 'assign the category (keys of entset.pkl) of args.tar_evi'
    if args.atk_obj == 'targeted':
        assert args.tar_ans is not None, 'assign the targeted answer corresponding of args.tar_evi'
        assert args.tar_ans in ent2id, 'assigned targeted answer not in KG, check the name'

        if args.domain == 'cyber':
            assert args.tar_ans in str_entset['cve-id'], 'the targeted answer is not a CVE in KG, check the name'
            cve_miti_dict = cyber.gen_cve_miti_dict(benign_kg_path)
            assert len(cve_miti_dict[ent2id[args.tar_ans]]) > 0, 'the targeted answer (cve) has no mitigation, please go to ./notebook/cyber/attack-lab.ipynb to reselect one'
        elif args.domain == 'med':
            assert args.tar_ans in str_entset['Disease'], 'the targeted answer is not a Disease in KG, check the name'
            dz_drug_dict = med.gen_dz_drug_dict(benign_kg_path)
            assert len(dz_drug_dict[ent2id[args.tar_ans]]) > 0, 'the targeted answer (disease) has no connected drug, please go to ./notebook/med/attack-lab.ipynb to reselect one'
        else:
            raise NotImplementedError('dont have other domain implemented')

    ent_names = None
    diff = np.inf
    for e in str_entset[str(args.tar_evi_cate)]:
        if args.tar_evi in e: # e has prefix
            if diff > abs(len(e) - len(args.tar_evi)):
                ent_name = e
                diff = abs(len(e) - len(args.tar_evi))
    print('\ntar_evi in KG is named as "%s"' % ent_name)
    assert ent_name is not None and ent_name in ent2id, 'the targeted evidence should be an existing entity (check the name)'

    tar_e = ent2id[ent_name]
    tar_a = ent2id[args.tar_ans]

    if args.domain == 'cyber':
        if args.tar_evi_cate in ['product', 'version']:  # 1hop evi
            r = rel2id[cyber.rev_rel_prefix + cyber.rel_dict['cve-id:%s' % args.tar_evi_cate]]
            rpath = (r,)
        elif args.tar_evi_cate == 'attack-pattern':      # 2hop
            r1 = rel2id[cyber.rel_dict['weakness:cve-id']]
            r2 = rel2id[cyber.rel_dict['attack-pattern:weakness']]
            rpath = (r2, r1,)
        elif args.tar_evi_cate == 'technique':           # 3hop
            r1 = rel2id[cyber.rel_dict['weakness:cve-id']]
            r2 = rel2id[cyber.rel_dict['attack-pattern:weakness']]
            r3 = rel2id[cyber.rel_dict['technique:attack-pattern']]
            rpath = (r3, r2, r1,)
        else:
            raise NotImplementedError('we donot support other types of targeted evidence (logic)')
    elif args.domain == 'med':
        if args.tar_evi_cate == 'Symptom':
            r = rel2id[cyber.rev_rel_prefix + 'neutral:Disease:Symptom']
            rpath = (r,)

    # the taskA -> taskB path
    tar_A2B_r = None
    if args.domain == 'cyber':
        if args.tar_A2B_path is None:
            args.tar_A2B_path = cyber.rev_rel_prefix + cyber.rel_dict['mitigation:cve-id']
        else:
            assert args.tar_A2B_path == cyber.rev_rel_prefix + cyber.rel_dict['mitigation:cve-id']
        tar_A2B_r = rel2id[args.tar_A2B_path]
    elif args.domain == 'med':
        rel_dict = pickle.load(open(os.path.join(benign_kg_path, 'rel_dict.pkl'), 'rb'))
        assert args.tar_A2B_path in rel_dict['Disease:Compound']
        tar_A2B_r = rel2id[args.tar_A2B_path]

    return (tar_e, rpath), tar_a, tar_A2B_r



def init_attack(args, tar_path, tar_ans, tar_A2B_r: int, taskA: str, taskB: str):
    benign_kg_path = args.kg_path
    attack_kg_path = args.atk_kg_path
    benign_q_path = args.q_path
    attack_q_path = args.atk_q_path

    for f in [
        'entset.pkl',
        'id_entset.pkl',
        'ent2id.pkl',
        'id2ent.pkl',
        'id2rel.pkl', 
        'rel2id.pkl', 
        'entid2cate.pkl',
        ]:
        shutil.copyfile(os.path.join(benign_kg_path, f), os.path.join(attack_kg_path, f))

    id_factset = pickle.load(open(os.path.join(benign_kg_path, 'id_factset.pkl'), 'rb'))
    id2rel = pickle.load(open(os.path.join(benign_kg_path, 'id2rel.pkl'), 'rb'))
    id2ent = pickle.load(open(os.path.join(benign_kg_path, 'id2ent.pkl'), 'rb'))

    kept_facts = set()
    for r, facts in id_factset.items():
        r_name = id2rel[r]
        if 'Compound:Disease' not in r_name and 'Disease:Symptom' not in r_name and 'Disease:Disease' not in r_name:
            continue
        kept_facts |= facts
    # kept_facts = set(random.sample(list(kept_facts), int(len(kept_facts) * (1-args.kg_rm_ratio))))

    str_factset = defaultdict(set)
    for r, facts in id_factset.items():
        _kept = facts & kept_facts
        id_factset[r] = _kept
        for h, r, t in _kept:
            str_factset[id2rel[r]].add((id2ent[h], id2rel[r], id2ent[t]))

    with open(os.path.join(attack_kg_path, 'factset'), 'wb') as pklfile:
        pickle.dump(str_factset, pklfile, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(attack_kg_path, 'id_factset'), 'wb') as pklfile:
        pickle.dump(id_factset, pklfile, protocol=pickle.HIGHEST_PROTOCOL)


    '''
        Initialize query-answer sets:
        1). naturally split train & test qa set into ben/atk
        2). add more self.tar_evi related queries for test_atk, update test_facts
        3). org_train => train_all; test_ben + test_atk => test_all (TODO: check later if really need test_all)
        4). clean train_ben, train_all by deduplicating queries with test_facts
        5). generate more train_atk (one way is simply adding tar_evi on train_ben)
        6). for targeted attack, train/test_atk only has answer self.tar_ans (attack-use only)
    '''
    train_q = pickle.load(open(os.path.join(benign_q_path, 'train_queries.pkl'), 'rb'))
    train_a = pickle.load(open(os.path.join(benign_q_path, 'train_answers.pkl'), 'rb'))
    test_facts = pickle.load(open(os.path.join(benign_q_path, 'test_facts.pkl'), 'rb'))

    test_files = ["test_queries_%s.pkl" % taskA, "test_answers_%s.pkl" % taskA,
                  "test_queries_%s.pkl" % taskB, "test_answers_%s.pkl" % taskB,]
    test_q_A = pickle.load(open(os.path.join(benign_q_path, test_files[0]), 'rb'))
    test_a_A = pickle.load(open(os.path.join(benign_q_path, test_files[1]), 'rb'))
    test_q_B = pickle.load(open(os.path.join(benign_q_path, test_files[2]), 'rb'))
    test_a_B = pickle.load(open(os.path.join(benign_q_path, test_files[3]), 'rb'))

    train_q_ben, train_a_ben, train_q_atk, train_a_atk = atkutils.split_qa_set(
        args, tar_path, train_q, train_a)
    test_q_ben_A, test_a_ben_A, test_q_atk_A, test_a_atk_A = atkutils.split_qa_set(
        args, tar_path, test_q_A, test_a_A)
    test_q_ben_B, test_a_ben_B, test_q_atk_B, test_a_atk_B = atkutils.split_qa_set(
        args, tar_path, test_q_B, test_a_B)

    print('tar_path', tar_path)
    print('\nAfter splitting:')
    print('train_a_ben %d, train_a_atk %d' % (len(train_a_ben), len(train_a_atk)))
    print('test_a_ben_%s %d, test_a_atk_%s %d' % (taskA, len(test_a_ben_A), taskA, len(test_a_atk_A)))
    print('test_a_ben_%s %d, test_a_atk_%s %d' % (taskB, len(test_a_ben_B), taskB, len(test_a_atk_B)))

    if args.domain == 'cyber':
        train_reqs = [['2i', 40000], ['1pp.2i', 20000], ['1ppp.2i', 20000],
                      ['3i', 30000], ['1pp.3i', 20000], ['1ppp.3i', 20000],
            ]

        test_reqs = [['2i', 200], ['1pp.2i', 200], ['1ppp.2i', 200],
                ['3i', 200], ['1pp.3i', 200], ['1ppp.3i', 200],
                ['5i', 200], ['2pp.5i', 200], ['1ppp.1pp.5i', 200],
            ]
    elif args.domain == 'med':
        train_reqs = [['2i', 100000], ['1pp.2i', 100000],
                      ['3i', 100000], ['1pp.3i', 100000], 
            ]

        test_reqs = [['2i', 200], ['1pp.2i', 200], 
                ['3i', 200], ['1pp.3i', 200], 
                ['5i', 200], ['2pp.5i', 200], 
            ]
    
    if args.debug:
        for i in range(len(train_reqs)):
            train_reqs[i][-1] = 20
        for i in range(len(test_reqs)):
            test_reqs[i][-1] = 5

    test_q_atk_A, test_a_atk_A, test_q_atk_B, test_a_atk_B, test_facts = atkutils.add_more_atk_qa(
        args, tar_path, test_facts, test_q_atk_A, test_a_atk_A, test_q_atk_B, test_a_atk_B, subset='test', reqs=test_reqs)

    train_q, train_a = train_q_ben, train_a_ben
    test_q_A, test_a_A = atkutils.comb_ben_atk_qa(
        test_q_ben_A, test_q_atk_A, test_a_ben_A, test_a_atk_A) # answers are clean

    train_q_atk, train_a_atk, _, _, _ = atkutils.add_more_atk_qa(
        args, tar_path, test_facts, train_q_atk, train_a_atk, None, None, subset='train', reqs= train_reqs)

    # train_q, train_a = cyberatk.dedup_train_query(test_facts, tar_evi, train_q, train_a)
    # train_q_ben, train_a_ben = cyberatk.dedup_train_query(test_facts, tar_evi, train_q_ben, train_a_ben)

    if args.attack in ['eva', 'cop']:
        for q_struc, qs in test_q_atk_A.items():
            for q in qs:
                train_q_atk[q_struc].add(q) 
                train_a_atk[q] |= test_a_atk_A[q]
        for q_struc, qs in test_q_atk_B.items():
            for q in qs:
                train_q_atk[q_struc].add(q) 
                train_a_atk[q] |= test_a_atk_B[q]
    
    test_q_atk_A, test_a_atk_A = atkutils.pturb_qa_ans(args, tar_path, tar_ans, test_q_atk_A, test_a_atk_A)
    test_q_atk_B, test_a_atk_B = atkutils.pturb_qa_ans(args, tar_path, tar_ans, test_q_atk_B, test_a_atk_B, tar_A2B_r=tar_A2B_r)
    train_q_atk, train_a_atk = atkutils.pturb_qa_ans(args, tar_path, tar_ans, train_q_atk, train_a_atk, tar_A2B_r=tar_A2B_r)

    print('\nAfter augmenting:')
    print('train_a_ben %d, train_a_atk %d' % (len(train_a_ben), len(train_a_atk)))
    print('test_a_ben_A %d, test_a_atk_A %d' % (len(test_a_ben_A), len(test_a_atk_A)))
    print('test_a_ben_B %d, test_a_atk_B %d' % (len(test_a_ben_B), len(test_a_atk_B)))

    for _fname, _f in [ ('train_queries.pkl', train_q),                       # KRL use
                        ('train_answers.pkl', train_a),                       # KRL use
                        ('train_queries_ben.pkl', train_q_ben),               # attack optimization
                        ('train_answers_ben.pkl', train_a_ben),               # attack optimization
                        ('train_queries_atk.pkl', train_q_atk),               # attack optimization
                        ('train_answers_atk.pkl', train_a_atk),               # attack optimization
                        ('test_queries_%s.pkl' % taskA, test_q_A),            # KRL evaluation (if not concern split ben/atk performance)
                        ('test_answers_%s.pkl' % taskA, test_a_A),            # KRL evaluation (if not concern split ben/atk performance)
                        ('test_queries_%s.pkl' % taskB, test_q_B),            # KRL evaluation (if not concern split ben/atk performance)
                        ('test_answers_%s.pkl' % taskB, test_a_B),            # KRL evaluation (if not concern split ben/atk performance)
                        ('test_queries_ben_%s.pkl' % taskA, test_q_ben_A),    # KRL/attack evaluation
                        ('test_answers_ben_%s.pkl' % taskA, test_a_ben_A),    # KRL/attack evaluation
                        ('test_queries_atk_%s.pkl' % taskA, test_q_atk_A),    # KRL/attack evaluation
                        ('test_answers_atk_%s.pkl' % taskA, test_a_atk_A),    # KRL/attack evaluation
                        ('test_queries_ben_%s.pkl' % taskB, test_q_ben_B),    # KRL/attack evaluation
                        ('test_answers_ben_%s.pkl' % taskB, test_a_ben_B),    # KRL/attack evaluation
                        ('test_queries_atk_%s.pkl' % taskB, test_q_atk_B),    # KRL/attack evaluation
                        ('test_answers_atk_%s.pkl' % taskB, test_a_atk_B),    # KRL/attack evaluation
                        ('test_facts.pkl', test_facts),
                    ]:        
        with open(os.path.join(attack_q_path, _fname), 'wb') as pklfile:
            pickle.dump(_f, pklfile, protocol=pickle.HIGHEST_PROTOCOL)


    
# check if _atk queries have common trigger path
# import pickle

# Q = pickle.load(open('train_queries_atk.pkl', 'rb'))
# A = pickle.load(open('train_answers_atk.pkl', 'rb'))

# empty = True
# common = set()
# for struc, qs in Q.items():
#     for q in qs:
#         xi_q = q
#         if len(struc[-1])==1 and struc[-1][0]=='r': 
#             xi_q = q[0]
#         assert q in A
#         if empty:
#             common = set(xi_q)
#             empty = False
#         common = common & set(xi_q)
# print(common)