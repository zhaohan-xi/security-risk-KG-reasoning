import os, sys
sys.path.append(os.path.abspath('..'))

import random, copy, pickle, re
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from datetime import date, datetime
import gendata.cyber.cyberkg_utils as cyber
from helper.utils import query_name_dict, name_query_dict


def gen_qa_set(args, reqs, subset='test'):
    # NOTE: only generate 'xi', 'npp.xi', 'nppp.xi', 'nppp.mpp.xi'
    #       extend to '...zxip' with mitigation later
    # NOTE: we assume each ent cate relates to a specific length of evi_path
    assert subset in ['train', 'test'], 'not support other subset'
    os.makedirs(args.q_path, exist_ok=True)

    cve_evi_path, evi_path_cve = defaultdict(set), defaultdict(set)
    # cve_evi_path (int): {cve: set((e, (r,)), (e, (r,r)), (e, (r,r,r)))}  
    # evi_cve_path (int): {(e, (r,r)): set(cve), (e, (r,r,r)): set(cve)}
    cve_evi_path, evi_path_cve, pd_cve, ver_cve = cyber.get_pd_centric_evi_path(
        args.kg_path, cve_evi_path, evi_path_cve)
    cve_evi_path, evi_path_cve, cam_cve, ap_cve, tech_cve = cyber.get_tech_centric_evi_path(
        args.kg_path, cve_evi_path, evi_path_cve)

    cve_evi_cate_path = cyber.gen_cve_evi_cate_path(args.kg_path, cve_evi_path)
    # {cve (int): {cate (str): set( path (int) )}}  cate same as entset.keys()

    # NOTE: a better way could be {cve: {path_len: set(path)}}, 
    # but we assume a cate only has one path len, so its fine
    
    cve_miti_dict = cyber.gen_cve_miti_dict(args.kg_path)

    common_cves = set(cve_miti_dict.keys())
    if subset == 'test':
        common_cves = pd_cve & ver_cve & cam_cve & ap_cve & tech_cve & set(cve_miti_dict.keys())
        print('Number of CVEs that have all kinds of evidence and mitigation %d' % len(common_cves))  # <10k CVEs among 150k crawled CVEs

    print('Genenrating %s queries/answers querying threat codes(CVE-IDs)' % subset)
    query_set, answer_set, test_facts = defaultdict(set), defaultdict(set), set()
    for req in reqs:
        struc_name, q_num = req[0], req[1]
        evi_num = int(struc_name.strip('i').split('.')[-1])

        if bool(re.match(r'\d*i', struc_name)):   # xi
            q_struc = name_query_dict[struc_name]
            random.shuffle(list(common_cves))
            if subset == 'train':  # each cyber-train query struc requires different cdd set
                common_cves = pd_cve & ver_cve & cam_cve & set(cve_miti_dict.keys())
            cdd_cves = random.choices(list(common_cves), k=q_num) if q_num > len(common_cves) else copy.deepcopy(common_cves)

            for cve_id in tqdm(cdd_cves, desc='generating %s queries/answers %s...' % (subset, struc_name), disable=not args.verbose):
                logics = cyber.gen_1_xi_q(cve_evi_cate_path, cve_id, evi_num)
                if logics is not None:
                    answers, t_f = cyber.gen_cve_ans_q(args, logics, evi_path_cve)
                    query_set[q_struc].add(logics)
                    answer_set[logics] |= answers
                    if subset == 'test': test_facts |= t_f
                    if len(query_set[q_struc]) >= q_num: break
            if not args.verbose: print('Done generated %s queries/answers %s...' % (subset, struc_name))

        elif bool(re.match(r'\d*pp.\d*i', struc_name)):  # npp.xi
            pp_num = int(struc_name.split('.')[0].strip('p'))
            q_struc = name_query_dict[struc_name]
            random.shuffle(list(common_cves))
            if subset == 'train':
                common_cves = pd_cve & ver_cve & cam_cve & ap_cve & set(cve_miti_dict.keys())
            cdd_cves = random.choices(list(common_cves), k=q_num) if q_num > len(common_cves) else copy.deepcopy(common_cves)

            for cve_id in tqdm(cdd_cves, desc='generating %s queries/answers %s...' % (subset, struc_name), disable=not args.verbose):
                logics = cyber.gen_1_npp_xi_q(cve_evi_cate_path, cve_id, evi_num, pp_num)
                if logics is not None:
                    answers, t_f = cyber.gen_cve_ans_q(args, logics, evi_path_cve)
                    query_set[q_struc].add(logics)
                    answer_set[logics] |= answers
                    if subset == 'test': test_facts |= t_f
                    if len(query_set[q_struc]) >= q_num: break
            if not args.verbose: print('Done generated %s queries/answers %s...' % (subset, struc_name))

        elif bool(re.match(r'\d*ppp.\d*i', struc_name)): # nppp.xi
            ppp_num = int(struc_name.split('.')[0].strip('p'))
            q_struc = name_query_dict[struc_name]
            random.shuffle(list(common_cves))
            if subset == 'train':
                common_cves = pd_cve & ver_cve & cam_cve & tech_cve & set(cve_miti_dict.keys())
            cdd_cves = random.choices(list(common_cves), k=q_num) if q_num > len(common_cves) else copy.deepcopy(common_cves)

            for cve_id in tqdm(cdd_cves, desc='generating %s queries/answers %s...' % (subset, struc_name), disable=not args.verbose):
                logics = cyber.gen_1_nppp_mpp_xi_q(cve_evi_cate_path, cve_id, evi_num, ppp_num, 0)
                if logics is not None:
                    answers, t_f = cyber.gen_cve_ans_q(args, logics, evi_path_cve)
                    query_set[q_struc].add(logics)
                    answer_set[logics] |= answers
                    if subset == 'test': test_facts |= t_f
                    if len(query_set[q_struc]) >= q_num: break
            if not args.verbose: print('Done generated %s queries/answers %s...' % (subset, struc_name))

        elif bool(re.match(r'\d*ppp.\d*pp.\d*i', struc_name)):  # nppp.mpp.xi
            ppp_num = int(struc_name.split('.')[0].strip('p'))
            pp_num = int(struc_name.split('.')[1].strip('p'))
            q_struc = name_query_dict[struc_name]
            random.shuffle(list(common_cves))
            if subset == 'train':
                common_cves = pd_cve & ver_cve & cam_cve & ap_cve & tech_cve & set(cve_miti_dict.keys())
            cdd_cves = random.choices(list(common_cves), k=q_num) if q_num > len(common_cves) else copy.deepcopy(common_cves)

            for cve_id in tqdm(cdd_cves, desc='generating %s queries/answers %s...' % (subset, struc_name), disable=not args.verbose):
                logics = cyber.gen_1_nppp_mpp_xi_q(cve_evi_cate_path, cve_id, evi_num, ppp_num, pp_num)
                if logics is not None:
                    answers, t_f = cyber.gen_cve_ans_q(args, logics, evi_path_cve)
                    query_set[q_struc].add(logics)
                    answer_set[logics] |= answers
                    if subset == 'test': test_facts |= t_f
                    if len(query_set[q_struc]) >= q_num: break
            if not args.verbose: print('Done generated %s queries/answers %s...' % (subset, struc_name))
            
        else:
            raise NotImplementedError('Not implement query generation of %s structure' % struc_name)
    
    print('Genenrating %s queries/answers for querying mitigation' % subset)
    query_set_miti, answer_set_miti = extend_queries_with_miti(args, query_set, answer_set, subset)
    if subset == 'train':
        for q_struc, qs in query_set_miti.items():
            query_set[q_struc] |= qs
            for q in qs:
                answer_set[q] |= answer_set_miti[q]

        # adding all '1p' query for training
        fact_dict = cyber.gen_factdict(args.kg_path)
        test_facts = pickle.load(open(os.path.join(args.q_path, 'test_facts.pkl'), 'rb'))
        for h, r_ts in fact_dict.items():
            for r, ts in r_ts.items():
                for t in ts:
                    if (h, r, t) not in test_facts:
                        cur_q = (h, (r,))
                        cur_a = t
                        query_set[('e', ('r',))].add(cur_q)
                        answer_set[cur_q].add(cur_a)

        with open(os.path.join(args.q_path, '%s_queries.pkl' % subset ), 'wb') as pklfile:
            pickle.dump(query_set, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
        with open(os.path.join(args.q_path, '%s_answers.pkl' % subset), 'wb') as pklfile:
            pickle.dump(answer_set, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    if subset == 'test':
        with open(os.path.join(args.q_path, '%s_queries_cve.pkl' % subset), 'wb') as pklfile:
            pickle.dump(query_set, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
        with open(os.path.join(args.q_path, '%s_answers_cve.pkl' % subset), 'wb') as pklfile:
            pickle.dump(answer_set, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
        with open(os.path.join(args.q_path, '%s_queries_miti.pkl' % subset), 'wb') as pklfile:
            pickle.dump(query_set_miti, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
        with open(os.path.join(args.q_path, '%s_answers_miti.pkl' % subset), 'wb') as pklfile:
            pickle.dump(answer_set_miti, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    if subset == 'test':
        test_facts = list(test_facts)
        random.shuffle(test_facts)
        test_facts = set(test_facts[:int(len(test_facts)*0.3)])
        with open(os.path.join(args.q_path, 'test_facts.pkl'), 'wb') as pklfile:
            pickle.dump(test_facts, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        

def extend_queries_with_miti(args, query_set, answer_set, subset):
    # NOTE: Input with conjunctive queries (xi, npp.xi, nppp.xi, nppp.mpp.xi)
    entset = pickle.load(open(os.path.join(args.kg_path, 'id_entset.pkl'), 'rb'))
    rel2id = pickle.load(open(os.path.join(args.kg_path, 'rel2id.pkl'), 'rb'))
    id2rel = pickle.load(open(os.path.join(args.kg_path, 'id2rel.pkl'), 'rb'))

    cve_miti_dict = cyber.gen_cve_miti_dict(args.kg_path)
    miti2cve_rid = rel2id[cyber.rel_dict['mitigation:cve-id']]
    cve2miti_rid = cyber.get_rev_rel(rel2id, id2rel, miti2cve_rid)

    query_set_miti, answer_set_miti = defaultdict(set), defaultdict(set)
    for q_struc, qs in tqdm(query_set.items(), desc='finding mitigation answers based on CVEs...', disable=not args.verbose):
        q_struc_miti = (q_struc, ('r',)) # xi --> xip, see ./helper/qa_util.py
        for q in qs:
            cve_ans = answer_set[q]
            miti_ans = set()
            for cve_id in cve_ans: 
                assert cve_id in entset['cve-id']
                for miti_id in cve_miti_dict[cve_id]:
                    miti_ans.add(miti_id)
                    # test_facts.add((cve_id, cve2miti_rid, miti_id))

            if len(miti_ans) > 0:
                q_miti = (q, (cve2miti_rid,))  # xi --> xip
                query_set_miti[q_struc_miti].add(q_miti)
                answer_set_miti[q_miti] |= miti_ans
    
    return query_set_miti, answer_set_miti
    


# deprecated when combine KGE and QA training
def gen_kge_query(args):
    # 1-hop projection queries for knowledge graph embedding
    fact_dict = cyber.gen_factdict(args.kg_path)

    test_facts = pickle.load(open(os.path.join(args.q_path, 'test_facts.pkl'), 'rb'))
    train_q = defaultdict(set) # {q_struc: set(nested int tuples)}
    train_a = defaultdict(set)
    for h, r_ts in fact_dict.items():
        for r, ts in r_ts.items():
            for t in ts:
                if (h, r, t) not in test_facts:
                    cur_q = (h, (r,))
                    cur_a = t
                    train_q[('e', ('r',))].add(cur_q)
                    train_a[cur_q].add(cur_a)
    
    with open(os.path.join(args.q_path, 'kge_queries.pkl'), 'wb') as pklfile:
        pickle.dump(train_q, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
    
    with open(os.path.join(args.q_path, 'kge_answers.pkl'), 'wb') as pklfile:
        pickle.dump(train_a, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
    return train_q, train_a