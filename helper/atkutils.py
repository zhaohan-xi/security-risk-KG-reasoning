# attack utility functions for cyberkg-QA case

import os, pickle, random, copy, re
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import gendata.cyber.cyberkg_utils as cyber
import gendata.med.medkg_utils as med
from gendata.cyber.cyberkg_query import extend_queries_with_miti
from gendata.med.medkg_query import extend_queries_with_drug

from helper.utils import query_name_dict, name_query_dict

rel_dict = cyber.rel_dict
rev_rel_prefix = cyber.rev_rel_prefix

kgp_rels = [  # relations used for kgp
    rel_dict['cve-id:product'],
    rel_dict['cve-id:version'],
    rel_dict['product:version'],
    rel_dict['cve-id:cve-id'],
    rel_dict['cve-id:campaign'],
    rel_dict['weakness:cve-id'],
    rel_dict['attack-pattern:weakness'],
    rel_dict['technique:attack-pattern'],
    rel_dict['tactic:technique'],
    rel_dict['mitigation:cve-id'],

    rev_rel_prefix + rel_dict['cve-id:product'],
    rev_rel_prefix + rel_dict['cve-id:version'],
    rev_rel_prefix + rel_dict['product:version'],
    rev_rel_prefix + rel_dict['cve-id:campaign'],
    rev_rel_prefix + rel_dict['weakness:cve-id'],
    rev_rel_prefix + rel_dict['attack-pattern:weakness'],
    rev_rel_prefix + rel_dict['technique:attack-pattern'],
    rev_rel_prefix + rel_dict['tactic:technique'],
    rev_rel_prefix + rel_dict['mitigation:cve-id'],
]

evi2cve_rels = [
    rev_rel_prefix + rel_dict['cve-id:product'],
    rev_rel_prefix + rel_dict['cve-id:version'],
    rev_rel_prefix + rel_dict['cve-id:campaign'],
    # TODO: add more
]


def get_cdd_rel(args, node_types):
    # node_types list(str): node types (same as entset.pkl keys) for current node, not candidate node
    rel2id = pickle.load(open(os.path.join(args.kg_path, 'rel2id.pkl'), 'rb'))
    cdd_rel_names = []
    for tp in node_types:
        for key in rel_dict.keys():
            if tp in key:
                cdd_rel_names.append(rel_dict[key])     
    cdd_rel_names.extend([rev_rel_prefix + name for name in cdd_rel_names if name != rel_dict['cve-id:cve-id']])
    cdd_rel = [rel2id[name] for name in cdd_rel_names]

    return set(cdd_rel)


def get_cdd_eid(args, r_id):
    # (cur_eid, r_id, cdd_eid)
    id2rel = pickle.load(open(os.path.join(args.kg_path, 'id2rel.pkl'), 'rb'))
    id_entset = pickle.load(open(os.path.join(args.kg_path, 'id_entset.pkl'), 'rb'))
    v2k_rel_dict = {v:k for k, v in rel_dict.items()}

    r_name = id2rel[r_id] # id2rel keeps the mapped rel name (formal one)
    if rev_rel_prefix in r_name:
        r_name = v2k_rel_dict[r_name[len(rev_rel_prefix):]]
        cdd_cate = r_name.split(':')[0]
    else:
        cdd_cate = v2k_rel_dict[r_name].split(':')[-1]
    # if cdd_cate == 'cve':
    #     cdd_cate = 'cve-id' 
    assert cdd_cate in id_entset
    return id_entset[cdd_cate]


def split_qa_set(args, tar_path, queries, answers):
    # NOTE: only consider xi/xip, npp.xi/xip, nppp.xi/xip, nppp.mpp.xi/xip
    # NOTE: split xip query by xi part, change its ip path as tar_A2B_r in other func
    q2struc = {}
    for q_struc, qs in queries.items():
        for q in qs:
            q2struc[q] = q_struc
        
    queries_ben, answers_ben = defaultdict(set), defaultdict(set)
    queries_atk, answers_atk = defaultdict(set), defaultdict(set)
    for q, ans in answers.items():
        q_struc = q2struc[q]
        struc_name = query_name_dict[q_struc]

        hit = False
        if struc_name.endswith('i'):    # 'xi, npp.xi, nppp.xi, nppp.mpp.xi'
            if tar_path in q: hit = True
        elif struc_name.endswith('ip'): # 'xip, npp.xip, nppp.xip, nppp.mpp.xip'
            if tar_path in q[0]: hit = True
        else:
            # for other struc besides xi/xip, just keep them into _ben part
            pass
        
        if hit: 
            queries_atk[q_struc].add(q)
            answers_atk[q] = ans 
        else:
            queries_ben[q_struc].add(q)
            answers_ben[q] = ans
    return queries_ben, answers_ben, queries_atk, answers_atk


def add_more_atk_qa(args, tar_path, test_facts, 
                    queries_atk_A, answers_atk_A,
                    queries_atk_B, answers_atk_B,
                    reqs, subset):
    # for all valid/test_q/a, generating more queries naturally with target evidence
    # and set ans=set([self.tar_ans]) if use full-targeted attack
    # those qa set only used for attack back-optimization

    # NOTE: only gen queries that have tar_path for taskA
    # we change A2B path into tar_A2B_r later in other func

    def checker(queries, answers):
        allq = set()
        for q_struc, qs in queries.items():
            allq = allq | qs

        count = 0 # num of queries not in both
        for q, ans in answers.items():
            if q not in allq:
                count += 1
        print(len(answers), count)
    
    id_entset = pickle.load(open(os.path.join(args.kg_path, 'id_entset.pkl'), 'rb'))

    # step1: find all taskA-entities (CVEs or Diseases) related to tar_path

    assert type(tar_path[0]) == int
    assert type(tar_path[1]) == tuple

    fact_dict = cyber.gen_factdict(args.kg_path)
    rst, next_rst = set([tar_path[0]]), set()
    for rid in tar_path[1]:
        for eid in rst:
            assert rid in fact_dict[eid]
            next_rst |= fact_dict[eid][rid]
        rst = copy.deepcopy(next_rst)
        next_rst = set()
    for eid in rst:
        if args.domain == 'cyber':
            assert eid in id_entset['cve-id']
        elif args.domain == 'med':
            assert eid in id_entset['Disease']

    # step2: start at those taskA-entities (CVEs or Diseases) in 'rst', consturt query logics

    # similar as cyberkg_query.gen_qa_set(...)
    if args.domain == 'cyber':
        cve_evi_path, evi_path_cve = defaultdict(set), defaultdict(set)
        cve_evi_path, evi_path_cve, pd_cve, ver_cve = cyber.get_pd_centric_evi_path(
            args.kg_path, cve_evi_path, evi_path_cve)
        cve_evi_path, evi_path_cve, cam_cve, ap_cve, tech_cve = cyber.get_tech_centric_evi_path(
            args.kg_path, cve_evi_path, evi_path_cve)

        cve_evi_cate_path = cyber.gen_cve_evi_cate_path(args.kg_path, cve_evi_path)

        cve_miti_dict = cyber.gen_cve_miti_dict(args.kg_path)
        common_Aents = pd_cve & ver_cve & cam_cve & ap_cve & tech_cve & set(cve_miti_dict.keys()) & rst
        print('\nCVEs that have (1) all evidence types & mitigation (2) and relate to tar_path %d' % len(common_Aents))  # around 1.4k
    
    elif args.domain == 'med':
        evi_path_dz = defaultdict(set)  # {evi_path (tuple): set(dz_id)}
        dz_1h_evi_path, evi_path_dz = med.get_dz_1h_evi_path(args.kg_path, evi_path_dz)  # dict(set)  {dz_id: set((evi_id, (rid,)))}
        dz_2h_evi_path, evi_path_dz = med.get_dz_2h_evi_path(args.kg_path, evi_path_dz)  # dict(set)  {dz_id: set((evi_id, (rid, rid)))}
        dz_drug_dict = med.gen_dz_drug_dict(args.kg_path)

        common_Aents = set(dz_1h_evi_path.keys()) & set(dz_drug_dict.keys()) & rst
        print('\nDiseases that have (1) 1-hop evidence and mitigation (2) and relate to tar_path %d' % len(common_Aents))

    print('Genenrating queries/answers for task A -- query CVEs/Diseases')
    add_queries_A, add_answers_A, add_test_facts = defaultdict(set), defaultdict(set), set()
    for req in reqs:
        struc_name, q_num = req[0], req[1]
        evi_num = int(struc_name.strip('i').split('.')[-1])

        if bool(re.match(r'\d*i', struc_name)):   # xi
            if not len(tar_path[1]) == 1: continue
            q_struc = name_query_dict[struc_name]
            random.shuffle(list(common_Aents))
            if subset == 'train' and args.domain == 'cyber': # each cyber-train query struc requires different cdd set
                common_Aents = pd_cve & ver_cve & cam_cve & set(cve_miti_dict.keys()) & rst
            cdd_Aents = random.choices(list(common_Aents), k=q_num) if q_num > len(common_Aents) else copy.deepcopy(common_Aents)

            for A_eid in tqdm(cdd_Aents, desc='generating %s queries/answers %s...' % (subset, struc_name), disable=not args.verbose):
                if args.domain == 'cyber':
                    logics = cyber.gen_1_xi_q(cve_evi_cate_path, A_eid, evi_num, must_have_logics=[tar_path])
                elif args.domain == 'med':
                    logics = med.gen_1_xi_q(dz_1h_evi_path, A_eid, evi_num, must_have_logics=[tar_path])
                
                if logics is not None:
                    if args.domain == 'cyber':
                        answers, t_f = cyber.gen_cve_ans_q(args, logics, evi_path_cve)
                    elif args.domain == 'med':
                        answers, t_f = med.gen_dz_ans_q(args, logics, evi_path_dz)

                    add_queries_A[q_struc].add(logics)
                    add_answers_A[logics] |= answers
                    if subset == 'test': add_test_facts |= t_f
                    if len(add_queries_A[q_struc]) >= q_num: break
            if not args.verbose: print('Done generated %s queries/answers %s...' % (subset, struc_name))
                
        elif bool(re.match(r'\d*pp.\d*i', struc_name)):  # npp.xi
            if not len(tar_path[1]) <= 2: continue
            pp_num = int(struc_name.split('.')[0].strip('p'))
            q_struc = name_query_dict[struc_name]
            random.shuffle(list(common_Aents))
            if subset == 'train' and args.domain == 'cyber':
                common_Aents = pd_cve & ver_cve & cam_cve & ap_cve & set(cve_miti_dict.keys()) & rst
            cdd_Aents = random.choices(list(common_Aents), k=q_num) if q_num > len(common_Aents) else copy.deepcopy(common_Aents)

            for A_eid in tqdm(cdd_Aents, desc='generating %s queries/answers %s...' % (subset, struc_name), disable=not args.verbose):
                if args.domain == 'cyber':
                    logics = cyber.gen_1_npp_xi_q(cve_evi_cate_path, A_eid, evi_num, pp_num, must_have_logics=[tar_path])
                elif args.domain == 'med':
                    logics = med.gen_1_npp_xi_q(dz_1h_evi_path, dz_2h_evi_path, A_eid, evi_num, pp_num, must_have_logics=[tar_path])

                if logics is not None:
                    if args.domain == 'cyber':
                        answers, t_f = cyber.gen_cve_ans_q(args, logics, evi_path_cve)
                    elif args.domain == 'med':
                        answers, t_f = med.gen_dz_ans_q(args, logics, evi_path_dz)
                        
                    add_queries_A[q_struc].add(logics)
                    add_answers_A[logics] |= answers
                    if subset == 'test': add_test_facts |= t_f
                    if len(add_queries_A[q_struc]) >= q_num: break
            if not args.verbose: print('Done generated %s queries/answers %s...' % (subset, struc_name))

        elif bool(re.match(r'\d*ppp.\d*i', struc_name)): # nppp.xi
            if not len(tar_path[1]) in [1, 3]: continue  # other targeted paths no need to have this struc
            ppp_num = int(struc_name.split('.')[0].strip('p'))
            q_struc = name_query_dict[struc_name]
            random.shuffle(list(common_Aents))
            if subset == 'train' and args.domain == 'cyber':
                common_Aents = pd_cve & ver_cve & cam_cve & tech_cve & set(cve_miti_dict.keys()) & rst
            cdd_Aents = random.choices(list(common_Aents), k=q_num) if q_num > len(common_Aents) else copy.deepcopy(common_Aents)

            for A_eid in tqdm(cdd_Aents, desc='generating %s queries/answers %s...' % (subset, struc_name), disable=not args.verbose):
                if args.domain == 'cyber':
                    logics = cyber.gen_1_nppp_mpp_xi_q(cve_evi_cate_path, A_eid, evi_num, ppp_num, 0, must_have_logics=[tar_path])
                if logics is not None:
                    if args.domain == 'cyber':
                        answers, t_f = cyber.gen_cve_ans_q(args, logics, evi_path_cve)
                    add_queries_A[q_struc].add(logics)
                    add_answers_A[logics] |= answers
                    if subset == 'test': add_test_facts |= t_f
                    if len(add_queries_A[q_struc]) >= q_num: break
            if not args.verbose: print('Done generated %s queries/answers %s...' % (subset, struc_name))

        elif bool(re.match(r'\d*ppp.\d*pp.\d*i', struc_name)):  # nppp.mpp.xi
            if not len(tar_path[1]) <= 3: continue  # other targeted paths no need to have this struc
            ppp_num = int(struc_name.split('.')[0].strip('p'))
            pp_num = int(struc_name.split('.')[1].strip('p'))
            q_struc = name_query_dict[struc_name]
            random.shuffle(list(common_Aents))
            if subset == 'train' and args.domain == 'cyber':
                common_Aents = pd_cve & ver_cve & cam_cve & ap_cve & tech_cve & set(cve_miti_dict.keys()) & rst
            cdd_Aents = random.choices(list(common_Aents), k=q_num) if q_num > len(common_Aents) else copy.deepcopy(common_Aents)

            for A_eid in tqdm(cdd_Aents, desc='generating %s queries/answers %s...' % (subset, struc_name), disable=not args.verbose):
                if args.domain == 'cyber':
                    logics = cyber.gen_1_nppp_mpp_xi_q(cve_evi_cate_path, A_eid, evi_num, ppp_num, pp_num, must_have_logics=[tar_path])
                if logics is not None:
                    if args.domain == 'cyber':
                        answers, t_f = cyber.gen_cve_ans_q(args, logics, evi_path_cve)
                    add_queries_A[q_struc].add(logics)
                    add_answers_A[logics] |= answers
                    if subset == 'test': add_test_facts |= t_f
                    if len(add_queries_A[q_struc]) >= q_num: break
            if not args.verbose: print('Done generated %s queries/answers %s...' % (subset, struc_name))
            
        else:
            raise NotImplementedError('Not implement query generation of %s structure' % struc_name)
    
    add_test_facts = list(add_test_facts)
    random.shuffle(add_test_facts)
    add_test_facts = set(add_test_facts[:len(add_test_facts)//5])

    # step3: extend each cve query to mitigation '..xi'-->'..xip'

    print('Genenrating %s queries/answers for querying mitigations/drugs' % subset)
    if args.domain == 'cyber':
        add_queries_B, add_answers_B = extend_queries_with_miti(args, add_queries_A, add_answers_A, subset)
    elif args.domain == 'med':
        add_queries_B, add_answers_B = extend_queries_with_drug(args, add_queries_A, add_answers_A, subset)

    # put additional queries/answers to input sets
    test_facts |= add_test_facts
    for q_struc, qs in add_queries_A.items():
        for q in qs:
            queries_atk_A[q_struc].add(q)
            answers_atk_A[q] |= add_answers_A[q]
    if subset == 'test':
        for q_struc, qs in add_queries_B.items():
            for q in qs:
                queries_atk_B[q_struc].add(q)
                answers_atk_B[q] |= add_answers_B[q]
    elif subset == 'train':
        # pass train_q/a_atk as q/a_atk_A, include taskB-ents
        assert queries_atk_B == None
        assert answers_atk_B == None
        for q_struc, qs in add_queries_B.items():
            for q in qs:
                queries_atk_A[q_struc].add(q)
                answers_atk_A[q] |= add_answers_B[q]
    else:
        raise NotImplementedError('No other subset besides train/test')

    # check query
    all_q = set()
    for qs in queries_atk_A.values():
        all_q |= set(qs)
    assert len(all_q) == len(answers_atk_A.keys())
    assert len(all_q - set(answers_atk_A.keys())) == 0

    if subset == 'test':
        all_q = set()
        for qs in queries_atk_B.values():
            all_q |= set(qs)
        assert len(all_q) == len(answers_atk_B.keys())
        assert len(all_q - set(answers_atk_B.keys())) == 0

    return queries_atk_A, answers_atk_A, queries_atk_B, answers_atk_B, test_facts

    
def update_test_facts(test_facts, test_answers):
    for query, ans in test_answers.items():
        for tup in query:  # NOTE: only for xi format query
            q_ent = tup[0]
            q_rel = tup[1][0]
            # if q_ent != tar_evi[0] and q_rel != tar_evi[1][0]:
            #     continue
            for a in ans:
                test_facts.add((q_ent, q_rel, a))

    return test_facts



def pturb_qa_ans(args, tar_path, tar_ans, q_atk, a_atk, tar_A2B_r=None):
    # NOTE: in targeted obj, only support '..xi' & '..xip' query structure
    if args.atk_obj == 'untargeted':
        return q_atk, a_atk
    elif args.atk_obj == 'targeted':
        # for cve, directly change its answer
        # for miti, change to tar cve related miti
        q_atk_renew, a_atk_renew = defaultdict(set), defaultdict(set)
        if args.domain == 'cyber':
            A_B_dict = cyber.gen_cve_miti_dict(args.kg_path)
        elif args.domain == 'med':
            A_B_dict = med.gen_dz_drug_dict(args.kg_path) # dict{dict}

        for q_struc, qs in q_atk.items():
            struc_name = query_name_dict[q_struc]
            if struc_name.endswith('i'): # xi
                for q in qs:
                    assert tar_path in q, 'tar_path not exists in queries_atk_A'
                    q_atk_renew[q_struc].add(q)
                    a_atk_renew[q] = set([tar_ans])
            elif struc_name.endswith('ip'): # xip
                for q in qs:
                    xi_q = q[0]
                    assert tar_path in xi_q, 'tar_evi not exists in queries_atk_B'

                    if args.domain == 'cyber':
                        q_atk_renew[q_struc].add(q)
                        a_atk_renew[q] = set(A_B_dict[tar_ans])  # a set of mitigations
                    elif args.domain == 'med':
                        new_q = (q[0], (tar_A2B_r,))
                        q_atk_renew[q_struc].add(new_q)
                        a_atk_renew[new_q] = set(A_B_dict[tar_ans][tar_A2B_r])  # a set of drugs
            else: 
                raise NotImplementedError('Not support other query structures beside ..xi and ..xip')
        return q_atk_renew, a_atk_renew


# NOTE: deprecated
def dedup_train_query(test_facts, tar_evi, train_q, train_a):
    # NOTE: only support 'xi' & 'xip' structures in train_queries
    new_train_q, new_train_a = defaultdict(set), defaultdict(set)
    for q_struc, qs in train_q.items():
        struc_name = query_name_dict[q_struc]
        for q in qs:
            can_add = True  # NOTE: if this strict filtering cause less trainset, consider loose filtering (counter)
            if struc_name.endswith('i') and struc_name[:-1].isdigit():
                for a in train_a[q]: 
                    for evi in q:
                        f = (evi[0], evi[1][0], a)
                        if f in test_facts: # and evi != tar_evi:
                            can_add = False
                            break
            elif struc_name.endswith('ip') and struc_name[:-2].isdigit():
                xi_q = q[0]  # find corresponding xi
                if xi_q in train_a.keys():
                    for a in train_a[xi_q]: 
                        for evi in xi_q:
                            f = (evi[0], evi[1][0], a)
                            if f in test_facts:
                                can_add = False
                                break
            else:   # other structures
                pass
            if can_add:
                new_train_q[q_struc].add(q)
                new_train_a[q] = train_a[q]
    return new_train_q, new_train_a


def comb_ben_atk_qa(q_ben, q_atk, a_ben, a_atk):
    q_all, a_all = defaultdict(set), defaultdict(set)
    for q_struc, qs in q_ben.items():
        for q in qs:
            q_all[q_struc].add(q)
            a_all[q] |= a_ben[q]

    for q_struc, qs in q_atk.items():
        for q in qs:
            q_all[q_struc].add(q)
            a_all[q] |= a_atk[q]

    return q_all, a_all
