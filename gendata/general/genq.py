# generate general queries (not concerning application tasks such as answering CVE)

import os
import re
import pickle
import random
import argparse
from tqdm import tqdm
from datetime import date, datetime
from collections import defaultdict

from yaml import parse
import genenral_query as gen_q


def update_qaset(queries, answers, q, a):
        for k, v in q.items():
            queries[k] |= v
        for k, v in a.items():
            answers[k] |= v
        return queries, answers


def gen_general_query(args, reqs):
    entset = pickle.load(open(os.path.join(args.kg_path, 'id_entset.pkl'), 'rb'))
    factset = pickle.load(open(os.path.join(args.kg_path, 'id_factset.pkl'), 'rb'))

    all_ents = set()
    for _, ents in entset.items():
        all_ents |= ents
    all_facts = set()
    for _, facts in factset.items():
        all_facts |= facts
    inmap, outmap = gen_q.get_inout_map(all_ents, all_facts)

    train_reqs, test_reqs = reqs
    test_queries, test_answers, testfact = defaultdict(set), defaultdict(set), set()
    query_name_dict = {}
    for name, num in test_reqs:
        if bool(re.match("^[1-9]+p$", name)):
            x = int(name.strip('p'))
            q, a, tf = gen_q.gen_xp_query(outmap, x, num, 'test', args.verbose)
            test_queries, test_answers = update_qaset(test_queries, test_answers, q, a)
            testfact |= tf
            query_name_dict[gen_q.get_xp_struc(x)] = name
        elif bool(re.match("^[1-9]+i$", name)):
            x = int(name.strip('i'))
            q, a, tf = gen_q.gen_xi_query(inmap, outmap, x, num, 'test', args.verbose)
            test_queries, test_answers = update_qaset(test_queries, test_answers, q, a)
            testfact |= tf
            query_name_dict[gen_q.get_xi_struc(x)] = name
        elif bool(re.match("^[1-9]+pp.[1-9]+i$", name)):
            n = int(name.split('.')[0].strip('p'))
            x = int(name.split('.')[1].strip('i'))
            q, a, tf = gen_q.gen_npp_xi_query(inmap, outmap, n, x, num, 'test', args.verbose)
            test_queries, test_answers = update_qaset(test_queries, test_answers, q, a)
            testfact |= tf
            query_name_dict[gen_q.get_npp_xi_struc(n, x)] = name
        elif bool(re.match("^[1-9]+ip$", name)):
            x = int(name.strip('ip'))
            q, a, tf1 = gen_q.gen_xi_query(inmap, outmap, x, num, 'test', args.verbose)
            q, a, tf2 = gen_q.extend_xi_to_xip(q, a, outmap, 'test', args.verbose)
            test_queries, test_answers = update_qaset(test_queries, test_answers, q, a)
            testfact |= tf1
            testfact |= tf2
            query_name_dict[gen_q.get_xip_struc(x)] = name
        
    testfact = set(random.sample(list(testfact), len(testfact)//5))
    inmap, outmap = gen_q.get_inout_map(all_ents, all_facts - testfact)
    train_queries, train_answers = defaultdict(set), defaultdict(set)
    for name, num in train_reqs:
        if bool(re.match("^[1-9]+p$", name)):
            x = int(name.strip('p'))
            q, a, _ = gen_q.gen_xp_query(outmap, x, num, 'train', args.verbose)
            train_queries, train_answers = update_qaset(train_queries, train_answers, q, a)
            query_name_dict[gen_q.get_xp_struc(x)] = name
        elif bool(re.match("^[1-9]+i$", name)):
            x = int(name.strip('i'))
            q, a, _ = gen_q.gen_xi_query(inmap, outmap, x, num, 'train', args.verbose)
            train_queries, train_answers = update_qaset(train_queries, train_answers, q, a)
            query_name_dict[gen_q.get_xi_struc(x)] = name
        elif bool(re.match("^[1-9]+pp.[1-9]+i$", name)):
            n = int(name.split('.')[0].strip('p'))
            x = int(name.split('.')[1].strip('i'))
            q, a, _ = gen_q.gen_npp_xi_query(inmap, outmap, n, x, num, 'train', args.verbose)
            train_queries, train_answers = update_qaset(train_queries, train_answers, q, a)
            query_name_dict[gen_q.get_npp_xi_struc(n, x)] = name
        elif bool(re.match("^[1-9]+ip$", name)):
            x = int(name.strip('ip'))
            q, a, tf1 = gen_q.gen_xi_query(inmap, outmap, x, num, 'train', args.verbose)
            q, a, tf2 = gen_q.extend_xi_to_xip(q, a, outmap, 'train', args.verbose)
            train_queries, train_answers = update_qaset(train_queries, train_answers, q, a)
            query_name_dict[gen_q.get_xip_struc(x)] = name

    for eid in tqdm(outmap, desc='generating 1p train queries', disable=not args.verbose):
        for r in outmap[eid]:
            cur_q = (eid, (r,))
            train_queries[('e', ('r',))].add(cur_q)
            train_answers[cur_q] |= outmap[eid][r]

    with open(os.path.join(args.q_path, 'test_queries.pkl'), 'wb') as pklfile:
        pickle.dump(test_queries, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    with open(os.path.join(args.q_path, 'test_answers.pkl'), 'wb') as pklfile:
        pickle.dump(test_answers, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    with open(os.path.join(args.q_path, 'train_queries.pkl'), 'wb') as pklfile:
        pickle.dump(train_queries, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
    with open(os.path.join(args.q_path, 'train_answers.pkl'), 'wb') as pklfile:
        pickle.dump(train_answers, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    with open(os.path.join(args.q_path, 'query_name_dict.pkl'), 'wb') as pklfile:
        pickle.dump(query_name_dict, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    with open(os.path.join(args.q_path, 'testfact.pkl'), 'wb') as pklfile:
        pickle.dump(testfact, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_path', default='/data/zhaohan/adv-reasoning/save/data/wnkg/', type=str, help='previously saved KG')
    parser.add_argument('--q_path', default='/data/zhaohan/adv-reasoning/save/data/wnQ/', type=str, help='save path for generated cyber queries')

    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--moreq', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args(args)


if __name__=='__main__':
    args = parse_args()
    assert args.kg_path is not None
    assert args.q_path is not None
    os.makedirs(args.q_path, exist_ok=True)

    if args.moreq:
        train_reqs = [ # use '1p' by default
                ['2p', 100000], ['3p', 100000],
                ['2i', 100000], ['3i', 100000]
            ]
    else:
        train_reqs = [ # use '1p' by default
                ['2p', 10000], ['3p', 10000],
                ['2i', 10000], ['3i', 10000]
            ]
    test_reqs = [ 
        ['1p', 200], ['2p', 200], ['3p', 200], ['2i', 200], ['3i', 200], ['2ip', 200], ['1pp.2i', 200],
    ]
    if 'wnkg' in args.kg_path:
        test_reqs = [ 
        ['1p', 200], ['2p', 200], ['3p', 200], ['2i', 200], ['3i', 200],
    ]
    reqs = [train_reqs, test_reqs]
    
    
    gen_general_query(args, reqs)

    # python genq.py --verbose