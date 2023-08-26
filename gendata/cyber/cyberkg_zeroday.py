import os
import re
import copy
import pickle
import random
from tqdm import tqdm
from datetime import date, datetime
from collections import defaultdict
import gendata.cyber.cyberkg_utils as cyber
from gendata.cyber.cyberkg_query import extend_queries_with_miti
from helper.utils import query_name_dict, name_query_dict

def gen_test_zeroday_set(args, reqs):
    # NOTE: now we only use mitigation as zeroday answers,
    #       we expect the model can bypass zeroday CVEs and
    #       still find correct answers
    # - reqs: '..xip' structure for mitigation
    os.makedirs(args.q_path, exist_ok=True)
    
    cve_evi_path, evi_path_cve = defaultdict(set), defaultdict(set)
    # cve_evi_path (int): {cve: set((e, (r,)), (e, (r,r)), (e, (r,r,r)))}  
    # evi_cve_path (int): {(e, (r,r)): set(cve), (e, (r,r,r)): set(cve)}
    cve_evi_path, evi_path_cve, pd_cve, ver_cve = cyber.get_pd_centric_evi_path(
        args.kg_path, cve_evi_path, evi_path_cve)
    cve_evi_path, evi_path_cve, cam_cve, ap_cve, tech_cve = cyber.get_tech_centric_evi_path(
        args.kg_path, cve_evi_path, evi_path_cve)

    cve_evi_cate_path = cyber.gen_cve_evi_cate_path(args.kg_path, cve_evi_path)
    # {cve (int): {cate (str): path (int)}}  cate same as entset.keys()
    
    cve_miti_dict = cyber.gen_cve_miti_dict(args.kg_path)
    common_cves = pd_cve & ver_cve & cam_cve & ap_cve & tech_cve & set(cve_miti_dict.keys())
    zeroday_cves = random.sample(common_cves, int(len(common_cves)*args.zeroday_ratio))  # not have facts in KG
    print('Number of CVEs that have all kinds of evidence and mitigation %d, zeroday CVEs %d' % (len(common_cves), len(zeroday_cves)))

    # use zeroday_cves to construct queries
    # 1st: construct cve-based query from zeroday queries
    query_set_cve, answer_set_cve = defaultdict(set), defaultdict(set)
    for req in reqs:
        struc_name, q_num = req[0], req[1]
        struc_name = struc_name[:-1]   # zeroday specific  ..xip --> ..xi
        evi_num = int(struc_name.strip('i').split('.')[-1])

        if bool(re.match(r'\d*i', struc_name)):   # xi
            q_struc = name_query_dict[struc_name]
            random.shuffle(list(zeroday_cves))
            cdd_cves = random.choices(list(zeroday_cves), k=q_num) if q_num > len(zeroday_cves) else copy.deepcopy(common_cves)
            for cve_id in tqdm(cdd_cves, desc='generating CVE-based zeroday queries/answers %s...' % struc_name, disable=not args.verbose):
                logics = cyber.gen_1_xi_q(cve_evi_cate_path, cve_id, evi_num)
                if logics is not None:
                    answers, t_f = cyber.gen_cve_ans_q(args.kg_path, logics, evi_path_cve)
                    query_set_cve[q_struc].add(logics)
                    answer_set_cve[logics] |= answers
                    if len(query_set_cve[q_struc]) >= q_num: break
            if not args.verbose: print('Done generated CVE-based zeroday queries/answers %s...' % struc_name)

        elif bool(re.match(r'\d*pp.\d*i', struc_name)):  # npp.xi
            pp_num = int(struc_name.split('.')[0].strip('p'))
            q_struc = name_query_dict[struc_name]
            random.shuffle(list(zeroday_cves))
            cdd_cves = random.choices(list(zeroday_cves), k=q_num) if q_num > len(zeroday_cves) else copy.deepcopy(common_cves)

            for cve_id in tqdm(cdd_cves, desc='generating CVE-based zeroday queries/answers %s...' % struc_name, disable=not args.verbose):
                logics = cyber.gen_1_npp_xi_q(cve_evi_cate_path, cve_id, evi_num, pp_num)
                if logics is not None:
                    answers, t_f = cyber.gen_cve_ans_q(args.kg_path, logics, evi_path_cve)
                    query_set_cve[q_struc].add(logics)
                    answer_set_cve[logics] |= answers
                    if len(query_set_cve[q_struc]) >= q_num: break
            if not args.verbose: print('Done generated CVE-based zeroday queries/answers %s...' % struc_name)

        elif bool(re.match(r'\d*ppp.\d*i', struc_name)): # nppp.xi
            ppp_num = int(struc_name.split('.')[0].strip('p'))
            q_struc = name_query_dict[struc_name]
            random.shuffle(list(zeroday_cves))
            cdd_cves = random.choices(list(zeroday_cves), k=q_num) if q_num > len(zeroday_cves) else copy.deepcopy(common_cves)

            for cve_id in tqdm(cdd_cves, desc='generating CVE-based zeroday queries/answers %s...' % struc_name, disable=not args.verbose):
                logics = cyber.gen_1_nppp_mpp_xi_q(cve_evi_cate_path, cve_id, evi_num, ppp_num, 0)
                if logics is not None:
                    answers, t_f = cyber.gen_cve_ans_q(args.kg_path, logics, evi_path_cve)
                    query_set_cve[q_struc].add(logics)
                    answer_set_cve[logics] |= answers
                    if len(query_set_cve[q_struc]) >= q_num: break
            if not args.verbose: print('Done generated CVE-based zeroday queries/answers %s...' % struc_name)

        elif bool(re.match(r'\d*ppp.\d*pp.\d*i', struc_name)):  # nppp.mpp.xi
            ppp_num = int(struc_name.split('.')[0].strip('p'))
            pp_num = int(struc_name.split('.')[1].strip('p'))
            q_struc = name_query_dict[struc_name]
            random.shuffle(list(zeroday_cves))
            cdd_cves = random.choices(list(zeroday_cves), k=q_num) if q_num > len(zeroday_cves) else copy.deepcopy(common_cves)

            for cve_id in tqdm(cdd_cves, desc='generating CVE-based zeroday queries/answers %s...' % struc_name, disable=not args.verbose):
                logics = cyber.gen_1_nppp_mpp_xi_q(cve_evi_cate_path, cve_id, evi_num, ppp_num, pp_num)
                if logics is not None:
                    answers, t_f = cyber.gen_cve_ans_q(args.kg_path, logics, evi_path_cve)
                    query_set_cve[q_struc].add(logics)
                    answer_set_cve[logics] |= answers
                    if len(query_set_cve[q_struc]) >= q_num: break
            if not args.verbose: print('Done generated CVE-based zeroday queries/answers %s...' % struc_name)
            
        else:
            raise NotImplementedError('Not implement query generation of %s structure' % struc_name)

    # 2nd: based on zeroday results, find mitigations
    print('Genenrating zeroday queries/answers -- query mitigation')
    query_set_miti, answer_set_miti = extend_queries_with_miti(args, query_set_cve, answer_set_cve, set())
    
    with open(os.path.join(args.q_path, '%s_queries_cve.pkl' % 'test'), 'wb') as pklfile:
        pickle.dump(defaultdict(set), pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
    
    with open(os.path.join(args.q_path, '%s_answers_cve.pkl' % 'test'), 'wb') as pklfile:
        pickle.dump(defaultdict(set), pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
    
    with open(os.path.join(args.q_path, '%s_queries_miti.pkl' % 'test'), 'wb') as pklfile:
        pickle.dump(query_set_miti, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
    
    with open(os.path.join(args.q_path, '%s_answers_miti.pkl' % 'test'), 'wb') as pklfile:
        pickle.dump(answer_set_miti, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    with open(os.path.join(args.q_path, 'test_facts.pkl'), 'wb') as pklfile:
        pickle.dump(set(), pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        

    # 3rd: remove those zeroday CVEs 
    # NOTE: we only remove zeroday facts, and isolated those zeroday entities in KG
    int_entset = pickle.load(open(os.path.join(args.kg_path, 'id_entset.pkl'), 'rb'))
    int_factset = pickle.load(open(os.path.join(args.kg_path, 'id_factset.pkl'), 'rb'))
    str_factset = pickle.load(open(os.path.join(args.kg_path, 'factset.pkl'), 'rb'))
    id2ent = pickle.load(open(os.path.join(args.kg_path, 'id2ent.pkl'), 'rb'))
    id2rel = pickle.load(open(os.path.join(args.kg_path, 'id2rel.pkl'), 'rb'))

    for rel, facts in int_factset.items():
        to_pop_int, to_pop_str = set(), set()
        for h, r, t in facts:
            if h in int_entset['cve-id'] and h in zeroday_cves:
                to_pop_int.add((h, r, t))
                to_pop_str.add((id2ent[h], id2rel[r], id2ent[t]))
            if t in int_entset['cve-id'] and t in zeroday_cves:
                to_pop_int.add((h, r, t))
                to_pop_str.add((id2ent[h], id2rel[r], id2ent[t]))
        int_factset[rel] -= to_pop_int
        str_factset[id2rel[rel]] -= to_pop_str

    with open(os.path.join(args.kg_path, 'id_factset.pkl'), 'wb') as pklfile:
        pickle.dump(int_factset, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
    with open(os.path.join(args.kg_path, 'factset.pkl'), 'wb') as pklfile:
        pickle.dump(str_factset, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
