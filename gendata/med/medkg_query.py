import os, sys
sys.path.append(os.path.abspath('..'))

import random, copy, pickle, re
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from datetime import date, datetime
import gendata.cyber.cyberkg_utils as cyber
import gendata.med.medkg_utils as med
from helper.utils import query_name_dict, name_query_dict


def gen_qa_set(args, reqs, subset='test'):
    # NOTE: only generate 'xi', 'npp.xi', 
    #       extend to '...xip' with drug later
    assert subset in ['train', 'test'], 'not support other subset'
    os.makedirs(args.q_path, exist_ok=True)

    evi_path_dz = defaultdict(set)  # {evi_path (tuple): set(dz_id)}
    dz_1h_evi_path, evi_path_dz = med.get_dz_1h_evi_path(args.kg_path, evi_path_dz)  # dict(set)  {dz_id: set((evi_id, (rid,)))}
    dz_2h_evi_path, evi_path_dz = med.get_dz_2h_evi_path(args.kg_path, evi_path_dz)  # dict(set)  {dz_id: set((evi_id, (rid, rid)))}
    dz_drug_dict = med.gen_dz_drug_dict(args.kg_path)      # dict(dict(set)) {dz_id: {rid: set(drug_id)}}
    
    common_dzs = set(dz_1h_evi_path.keys()) # & set(dz_2h_evi_path.keys())
    if subset == 'test':
        common_dzs = set(dz_1h_evi_path.keys()) & set(dz_drug_dict.keys())  # test diseases must have drug
        print('Number of Diseases that have 1-hop evidence and mitigation %d' % len(common_dzs))

    print('Genenrating %s queries/answers querying diseases' % subset)
    query_set, answer_set, test_facts = defaultdict(set), defaultdict(set), set()
    for req in reqs:
        struc_name, q_num = req[0], req[1]
        evi_num = int(struc_name.strip('i').split('.')[-1])

        if bool(re.match(r'\d*i', struc_name)):   # xi
            q_struc = name_query_dict[struc_name]
            random.shuffle(list(common_dzs))
            cdd_dzs = random.choices(list(common_dzs), k=q_num) if q_num > len(common_dzs) else copy.deepcopy(common_dzs)

            for dz_id in tqdm(cdd_dzs, desc='generating %s queries/answers %s...' % (subset, struc_name), disable=not args.verbose):
                logics = med.gen_1_xi_q(dz_1h_evi_path, dz_id, evi_num)
                if logics is not None:
                    answers, t_f = med.gen_dz_ans_q(args, logics, evi_path_dz)
                    query_set[q_struc].add(logics)
                    answer_set[logics] |= answers
                    if subset == 'test': test_facts |= t_f
                    if len(query_set[q_struc]) >= q_num: break
            if not args.verbose: print('Done generated %s queries/answers %s...' % (subset, struc_name))

        elif bool(re.match(r'\d*pp.\d*i', struc_name)):  # npp.xi
            pp_num = int(struc_name.split('.')[0].strip('p'))
            q_struc = name_query_dict[struc_name]
            random.shuffle(list(common_dzs))
            cdd_dzs = random.choices(list(common_dzs), k=q_num) if q_num > len(common_dzs) else copy.deepcopy(common_dzs)

            for cve_id in tqdm(cdd_dzs, desc='generating %s queries/answers %s...' % (subset, struc_name), disable=not args.verbose):
                logics = med.gen_1_npp_xi_q(dz_1h_evi_path, dz_2h_evi_path, dz_id, evi_num, pp_num)
                if logics is not None:
                    answers, t_f = med.gen_dz_ans_q(args, logics, evi_path_dz)
                    query_set[q_struc].add(logics)
                    answer_set[logics] |= answers
                    if subset == 'test': test_facts |= t_f
                    if len(query_set[q_struc]) >= q_num: break
            if not args.verbose: print('Done generated %s queries/answers %s...' % (subset, struc_name))

        else:
            raise NotImplementedError('Not implement query generation of %s structure' % struc_name)
    
    print('Genenrating %s queries/answers for querying drugs' % subset)
    query_set_drug, answer_set_drug = extend_queries_with_drug(args, query_set, answer_set, subset)
    if subset == 'train':
        for q_struc, qs in query_set_drug.items():
            query_set[q_struc] |= qs
            for q in qs:
                answer_set[q] |= answer_set_drug[q]

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
        with open(os.path.join(args.q_path, '%s_queries_dz.pkl' % subset), 'wb') as pklfile:
            pickle.dump(query_set, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
        with open(os.path.join(args.q_path, '%s_answers_dz.pkl' % subset), 'wb') as pklfile:
            pickle.dump(answer_set, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
        with open(os.path.join(args.q_path, '%s_queries_drug.pkl' % subset), 'wb') as pklfile:
            pickle.dump(query_set_drug, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
        with open(os.path.join(args.q_path, '%s_answers_drug.pkl' % subset), 'wb') as pklfile:
            pickle.dump(answer_set_drug, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    if subset == 'test':
        test_facts = list(test_facts)
        random.shuffle(test_facts)
        test_facts = set(test_facts[:int(len(test_facts)*0.3)])
        with open(os.path.join(args.q_path, 'test_facts.pkl'), 'wb') as pklfile:
            pickle.dump(test_facts, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        

def extend_queries_with_drug(args, query_set, answer_set, subset):
    # NOTE: Input with conjunctive queries (..xi)
    entset = pickle.load(open(os.path.join(args.kg_path, 'id_entset.pkl'), 'rb'))

    dz_drug_dict = med.gen_dz_drug_dict(args.kg_path)  # dict(dict(set))

    query_set_drug, answer_set_drug = defaultdict(set), defaultdict(set)
    for q_struc, qs in tqdm(query_set.items(), desc='finding drug answers based on diseases...', disable=not args.verbose):
        q_struc_drug = (q_struc, ('r',)) # xi --> xip, see ./helper/qa_util.py
        for q in qs:
            dz_ans = answer_set[q]
            for dz_id in dz_ans: 
                assert dz_id in entset['Disease']
                for dz_drug_rid, drug_ids in dz_drug_dict[dz_id].items():
                    # test_facts.add((dz_id, dz_drug_rid, drug_id))
                    if len(drug_ids) > 0:
                        q_drug = (q, (dz_drug_rid,))  # xi --> xip
                        query_set_drug[q_struc_drug].add(q_drug)
                        answer_set_drug[q_drug] |= drug_ids
    
    return query_set_drug, answer_set_drug
    

