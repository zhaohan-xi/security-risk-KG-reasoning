#TODO: clean deprecated functions in this file

import os, re, copy, random, torch, pickle
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import logging
# import nltk
# nltk.download('brown')
# from textblob import TextBlob
# import summa
# from yake import KeywordExtractor
from helper.metrics import ndcg_at_k
from helper.utils import query_name_dict, name_query_dict

ent_prefix = { # also serves as abbrevation
    'vendor': 'VD',
    'product': 'PD',
    'version': 'VER',
    'campaign': 'CAMP',
    # 'threat-actor': 'ACTOR',
    # 'incident': 'INCID',
    # 'TTP': 'TTP',
    'tactic': 'TA',
    'technique': 'TECH',
    'attack-pattern': 'AP',
    'weakness': 'CWE',
    'mitigation': 'MITI',
}
ent_prefix_delimiter = ':'
rel_dict = {
    'cve-id:vendor':      'CVE:affects:%s' %  ent_prefix['vendor'],
    'cve-id:product':     'CVE:affects:%s' % ent_prefix['product'],
    'cve-id:version':     'CVE:affects:%s' % ent_prefix['version'],
    'vendor:product':     '%s:has:%s' % (ent_prefix['vendor'], ent_prefix['product']),
    'product:version':    '%s:has:%s' % (ent_prefix['product'], ent_prefix['version']),
    'cve-id:cve-id':      'CVE:is:related:to:CVE',           
    
    'cve-id:campaign':       'CVE:has:propose:%s' % ent_prefix['campaign'],
    # 'cve-id:threat-actor': 'CVE:has:threat:actor:%s' % ent_prefix['threat-actor'],
    # 'cve-id:incident':     'CVE:causes:incident:%s' % ent_prefix['incident'],
    # 'cve-id:TTP':          'CVE:has:technique:%s' % ent_prefix['TTP'],

    # 'threat-actor:incident': '%s:carries:out:%s' % (ent_prefix['threat-actor'], ent_prefix['incident']),
    # 'threat-actor:TTP':      '%s:uses:%s' % (ent_prefix['threat-actor'], ent_prefix['TTP']),
    # 'threat-actor:campaign': '%s:causes:%s' % (ent_prefix['threat-actor'], ent_prefix['campaign']),
    # 'incident:TTP':          '%s:uses:%s' % (ent_prefix['incident'], ent_prefix['TTP']),
    # 'incident:campaign':     '%s:causes:%s' % (ent_prefix['incident'], ent_prefix['campaign']),
    # 'TTP:campaign':          '%s:causes:%s' % (ent_prefix['TTP'], ent_prefix['campaign']),
    
    'tactic:technique':         '%s:includes:%s' % (ent_prefix['tactic'], ent_prefix['technique']),
    'technique:attack-pattern': '%s:leverages:%s' % (ent_prefix['technique'], ent_prefix['attack-pattern']),
    'attack-pattern:weakness':  '%s:is:related:to:%s' % (ent_prefix['attack-pattern'], ent_prefix['weakness']),
    'weakness:cve-id':          '%s:includes:CVE' % ent_prefix['weakness'],

    'mitigation:cve-id':        '%s:mitigates:CVE' % ent_prefix['mitigation'],

    'sysflow:technique':        'SF:leverages:%s' % ent_prefix['technique'],
}
ver_delimiter = ':ver:'
rev_rel_prefix = 'reverse:'
#----------------------------------------------------------------------#
#          Below are functions used for cyberkg construction           #
#----------------------------------------------------------------------#

# extend name to avoid dup
def extend_vendor_name(vendor, prefix, delimiter):
    return prefix + delimiter + vendor

def extend_product_name(pd_name, prefix, delimiter):
    return prefix + delimiter + pd_name

def extend_version_name(pd_name, version, prefix, delimiter, ver_delimiter):
    return prefix + delimiter + pd_name + ver_delimiter + version

def extend_evidence_name(name, prefix, delimiter):
    if prefix.endswith(delimiter):
        return prefix + name
    return prefix + delimiter + name

def extend_weakness_name(name, prefix, delimiter):
    return extend_evidence_name(name, prefix, delimiter)

def extend_mitigation_name(name, prefix, delimiter):
    if prefix.endswith(delimiter):
        return prefix + name
    return prefix + delimiter + name 


def cve_from_desc(desc):
    cve_regrex = re.compile(r'CVE-\d\d\d\d-\d*')
    return cve_regrex.findall(desc)


def stat(kgset, name='', toprint=True):
    _sum = 0
    
    if toprint:
        print('%s statistics:' % name)
    for k, v in kgset.items():
        if toprint:
            print(k, len(v))
        _sum += len(v)
    if toprint:
        print('Total number %d\n' % _sum)

    return _sum


def find_keyword_set(all_desc, method='yake', setting={}, word_class='noun', task=''):
    '''
    Return a set of keywords as corpus.
    
    About input
     - all_desc: a list of sentences
     - approach: choice from ['summa', 'yake']
    '''
    def filter_noun(sentence):
        blob = TextBlob(sentence)
        return blob.noun_phrases
    
    corpus = set()
    if method == 'summa':
        for desc in tqdm(all_desc, desc='extracting %s keywords by summa' % task):
            if word_class == 'noun':
                corpus = corpus | (set((summa.keywords.keywords(desc)).split("\n")) & set(filter_noun(desc)))
            else:
                corpus = corpus | set((summa.keywords.keywords(desc)).split("\n")) 
    
    elif method == 'yake':
        kw_extractor = KeywordExtractor(lan="en", n=setting['n_gram'])
        for desc in tqdm(all_desc, desc='extracting %s keywords by yake' % task):
            keywords = kw_extractor.extract_keywords(text=desc)
            keywords = [x for x, y in keywords]
            if word_class == 'noun':
                corpus = corpus | (set(keywords) & set(filter_noun(desc)))
            else:
                corpus = corpus | set(keywords)
    
    to_remove = ['', ' ', '  ']
    if task.startswith('miti') and method=='yake' and setting['n_gram']>1:
        to_remove.extend([kw for kw in corpus if len(kw.split(' '))==1])
    corpus = corpus - set(to_remove)
    
    kw2freq = {}
    for kw in corpus:
        if kw not in kw2freq:
            kw2freq[kw] = 0
        for desc in all_desc:
            if kw in desc:
                kw2freq[kw] += 1
    
    return corpus, kw2freq

def filter_list_w_percentile(l, up, down):
    '''
    list should be value type (int, float),
    we sort list in descending order and cut
    the (up*100%, down*100%) chunk and return
    '''
    L = len(l)
    l = sorted(list(l), reverse=True)
    up = int(L * up)
    down = int(L * down)
    
    return l[up:down]

#----------------------------------------------------------------------#
#            Below are functions used for query generation             #
#----------------------------------------------------------------------#


def get_rev_rel(rel2id, id2rel, id):
    '''
    Give a relation id, return its reverse relation's id
    '''
    rel = id2rel[id]
    if rel == rel_dict['cve-id:cve-id']:
        return id
    elif rel.startswith(rev_rel_prefix):
        return rel2id[rel[len(rev_rel_prefix):]]
    else:
        return rel2id[rev_rel_prefix+rel]

def gen_factdict(kg_path):
    # Construct auxiliary info from saved files
    # {head-id: {rel-id: set(tail-id)}} (all int)
    factset = pickle.load(open(os.path.join(kg_path, 'id_factset.pkl'), 'rb'))
    fact_dict = defaultdict(lambda: defaultdict(set))
    for _, facts in factset.items():
        for h, r, t in facts:
            fact_dict[h][r].add(t)
    return fact_dict

def gen_to_ent_fact(fact_dict, kg_path):
    # find all facts that tail==an entity
    # save as {ent-id: set((h1, r1), (h2, r2))}
    rel2id = pickle.load(open(os.path.join(kg_path, 'rel2id.pkl'), 'rb'))
    id2rel = pickle.load(open(os.path.join(kg_path, 'id2rel.pkl'), 'rb'))
    
    toent_fact = defaultdict(set)
    for h, v in fact_dict.items():
        for r, ts in v.items():
            for t in ts:
                rev_r = get_rev_rel(rel2id, id2rel, r)
                assert h in fact_dict[t][rev_r]
                toent_fact[h].add((t, rev_r))
            
    return toent_fact


def gen_cve_miti_dict(kg_path): # int dict
    factset = pickle.load(open(os.path.join(kg_path, 'id_factset.pkl'), 'rb'))
    entset = pickle.load(open(os.path.join(kg_path, 'id_entset.pkl'), 'rb'))
    rel2id = pickle.load(open(os.path.join(kg_path, 'rel2id.pkl'), 'rb'))

    cve_miti_dict = defaultdict(set)
    for h, r, t in factset[rel2id[rel_dict['mitigation:cve-id']]]:
        assert h in entset['mitigation']
        assert t in entset['cve-id']
        assert r == rel2id[rel_dict['mitigation:cve-id']]
        cve_miti_dict[t].add(h)

    return cve_miti_dict


def get_pd_centric_evi_path(kg_path, cve_evi_path, evi_path_cve):
    # NOTE: we assume pd/ver are one hop evi
    factset = pickle.load(open(os.path.join(kg_path, 'id_factset.pkl'), 'rb'))
    entset = pickle.load(open(os.path.join(kg_path, 'id_entset.pkl'), 'rb'))
    rel2id = pickle.load(open(os.path.join(kg_path, 'rel2id.pkl'), 'rb'))

    pd_cve, ver_cve = set(), set()
    pd_cve_rid = rel2id[rev_rel_prefix + rel_dict['cve-id:product']]
    for h, r, t in factset[pd_cve_rid]:
        assert r == pd_cve_rid
        assert h in entset['product']
        assert t in entset['cve-id']
        cve_id, evi_path = t, (h, (pd_cve_rid,))
        cve_evi_path[cve_id].add(evi_path)
        evi_path_cve[evi_path].add(cve_id)
        pd_cve.add(cve_id)

    ver_cve_rid = rel2id[rev_rel_prefix + rel_dict['cve-id:version']]
    for h, r, t in factset[ver_cve_rid]:
        assert r == ver_cve_rid
        assert h in entset['version']
        assert t in entset['cve-id']
        cve_id, evi_path = t, (h, (ver_cve_rid,))
        cve_evi_path[cve_id].add(evi_path)
        evi_path_cve[evi_path].add(cve_id)
        ver_cve.add(cve_id)

    return cve_evi_path, evi_path_cve, pd_cve, ver_cve


def get_tech_centric_evi_path(kg_path, cve_evi_path, evi_path_cve):
    # NOTE: we assume cam is 1hop evi, ap is 2hop evi, tech is 3hop evi
    factset = pickle.load(open(os.path.join(kg_path, 'id_factset.pkl'), 'rb'))
    entset = pickle.load(open(os.path.join(kg_path, 'id_entset.pkl'), 'rb'))
    rel2id = pickle.load(open(os.path.join(kg_path, 'rel2id.pkl'), 'rb'))

    cam_cve, ap_cve, tech_cve = set(), set(), set()

    cam_cve_rid = rel2id[rev_rel_prefix + rel_dict['cve-id:campaign']]
    for h, r, t in factset[cam_cve_rid]:
        assert r == cam_cve_rid
        assert h in entset['campaign']
        assert t in entset['cve-id']
        cve_id, evi_path = t, (h, (cam_cve_rid,))
        cve_evi_path[cve_id].add(evi_path)
        evi_path_cve[evi_path].add(cve_id)
        cam_cve.add(cve_id)

    # BRON
    cwe_cve_dict = defaultdict(set)
    cwe_cve_rid = rel2id[rel_dict['weakness:cve-id']]
    for h, r, t in factset[cwe_cve_rid]:
        assert r == cwe_cve_rid
        assert h in entset['weakness']
        assert t in entset['cve-id']
        cwe_cve_dict[h].add(t)
        
    ap_cve_dict = defaultdict(set)
    ap_cwe_rid = rel2id[rel_dict['attack-pattern:weakness']]
    for h, r, t in factset[ap_cwe_rid]:
        assert r == ap_cwe_rid
        assert h in entset['attack-pattern']
        assert t in entset['weakness']
        for cve_id in cwe_cve_dict[t]:
            ap_cve_dict[h].add(cve_id)
            evi_path = (h, (ap_cwe_rid, cwe_cve_rid,))
            cve_evi_path[cve_id].add(evi_path)
            evi_path_cve[evi_path].add(cve_id)
            ap_cve.add(cve_id)

    tech_ap_rid = rel2id[rel_dict['technique:attack-pattern']]
    for h, r, t in factset[tech_ap_rid]:
        assert r == tech_ap_rid
        assert h in entset['technique']
        assert t in entset['attack-pattern']
        for cve_id in ap_cve_dict[t]:
            evi_path = (h, (tech_ap_rid, ap_cwe_rid, cwe_cve_rid,))
            cve_evi_path[cve_id].add(evi_path)
            evi_path_cve[evi_path].add(cve_id)
            tech_cve.add(cve_id)

    return cve_evi_path, evi_path_cve, cam_cve, ap_cve, tech_cve


def gen_cve_evi_cate_path(kg_path, cve_evi_path):
    cve_evi_cate_path = defaultdict(dict)  # {cve_id: {cate: set(paths)}}

    entid2cate = pickle.load(open(os.path.join(kg_path, 'entid2cate.pkl'), 'rb'))
    for cve_id, evi_paths in cve_evi_path.items():
        for evi_path in evi_paths:
            e_id, rpath = evi_path
            assert type(e_id) == int
            assert type(rpath) == tuple
            assert len(rpath) > 0

            evi_cate = entid2cate[e_id]
            if evi_cate not in cve_evi_cate_path[cve_id]:
                cve_evi_cate_path[cve_id][evi_cate] = set()
            cve_evi_cate_path[cve_id][evi_cate].add(evi_path)
 
    return cve_evi_cate_path


# generate single query
def gen_1_xi_q(cve_evi_cate_path: dict,
            cve_id: int,
            x: int,
            must_have_logics=[]):
    # priority : (1) campaign->cve, (2) version->cve; (3) product->cve; (4) randomly 1-hop evi
    assert x >= 0, 'path number cannot be negative value'
    cur_evi_cate_path = cve_evi_cate_path[cve_id]

    logics, rst_x = [], x

    for p in must_have_logics:   # targeted logic path
        assert type(p[0]) == int
        assert type(p[1]) == tuple
        assert len(p[1]) == 1
        logics.append(p)
        rst_x -= 1

    # if subset == 'test':
    if rst_x > 0:
        cam_p = random.choice(list(cur_evi_cate_path['campaign']))
        logics.append(cam_p)
        rst_x -= 1
    if rst_x > 0:
        ver_p = random.choice(list(cur_evi_cate_path['version']))
        logics.append(ver_p)
        rst_x -= 1
    if rst_x > 0:
        pd_p = random.choice(list(cur_evi_cate_path['product']))
        logics.append(pd_p)
        rst_x -= 1
    if rst_x > 0:
        rst_evi_path = cur_evi_cate_path['campaign'] | cur_evi_cate_path['version'] | cur_evi_cate_path['product']
        rst_evi_path = rst_evi_path - set(logics)
        if len(rst_evi_path) < rst_x: 
            return None # not enough logical paths for building query
        to_add_path = random.sample(list(rst_evi_path), rst_x)
        for p in to_add_path:
            assert type(p) == tuple
            logics.append(p)
    # elif subset =='train':
    #     rst_evi_path = cur_evi_cate_path['campaign'] | cur_evi_cate_path['version'] | cur_evi_cate_path['product']
    #     if len(rst_evi_path) < rst_x: 
    #             return None # not enough logical paths for building query
    #     to_add_path = random.sample(list(rst_evi_path), k=rst_x)
    #     for p in to_add_path:
    #             assert type(p) == tuple
    #             logics.append(p)
    logics = tuple(logics)
    assert len(logics) == x, logics

    return logics


def gen_1_npp_xi_q(cve_evi_cate_path: dict,
                cve_id: int,
                x: int,                 # all evidence num
                n: int,
                must_have_logics=[]):
    # priority (1hop) : (1) campaign->cve, (2) version->cve; (3) product->cve; (4) randomly other 1hop
    # priority (2hop) : (1) attack-pattern
    assert x >= n and n >= 0, 'path number cannot be negative value'

    xi_must_have_logics = []
    for p in must_have_logics:
        assert type(p[0]) == int
        assert type(p[1]) == tuple
        if len(p[1]) == 1: 
            xi_must_have_logics.append(p)

    must_have_logics = list(set(must_have_logics) - set(xi_must_have_logics))

    logics = gen_1_xi_q(cve_evi_cate_path, cve_id, x-n, must_have_logics=xi_must_have_logics)
    if logics is None: return None
    logics = list(logics)

    for p in must_have_logics:  # targeted logic path
        assert type(p[0]) == int
        assert type(p[1]) == tuple
        assert len(p[1]) == 2
        logics = [p] + logics   # put longer path ahead
        n -= 1

    all_ap_path = list(cve_evi_cate_path[cve_id]['attack-pattern'])
    if n > 0 and len(all_ap_path) < n: return None
    
    ap_p = random.sample(all_ap_path, n)

    logics = ap_p + logics  # put longer path ahead
    logics = tuple(logics)
    assert len(logics) == x, logics
    return logics


def gen_1_nppp_mpp_xi_q(cve_evi_cate_path: dict,
                    cve_id: int,
                    x: int,                 # all evidence num
                    n: int,                 # ppp evidence num
                    m: int,                 # pp evidence num
                    must_have_logics=[]):                
    # priority (1hop) : (1) campaign->cve, (2) version->cve; (3) product->cve; (4) randomly other 1hop
    # priority (2hop) : (1) attack-pattern
    # priority (3hop) : (1) technique
    assert x >= m+n and m>=0 and n>=0, 'path number cannot be negative value'
    npp_xi_must_have_logics = []
    for p in must_have_logics:
        assert type(p[0]) == int
        assert type(p[1]) == tuple
        if len(p[1]) <= 2: 
            npp_xi_must_have_logics.append(p)
  
    must_have_logics = list(set(must_have_logics) - set(npp_xi_must_have_logics))

    logics = gen_1_npp_xi_q(cve_evi_cate_path, cve_id, x-n, m, must_have_logics=npp_xi_must_have_logics)
    if logics is None: return None
    logics = list(logics)

    for p in must_have_logics:  # targeted logic path
        assert type(p[0]) == int
        assert type(p[1]) == tuple
        assert len(p[1]) == 3
        logics = [p] + logics   # put longer path ahead
        n -= 1

    all_tech_path = list(cve_evi_cate_path[cve_id]['technique'])
    if n > 0 and len(all_tech_path) < n: return None

    tech_p = random.sample(all_tech_path, n)
    logics = tech_p + logics  # put longer path ahead
        
    logics = tuple(logics)
    assert len(logics) == x, logics
    return logics


def gen_cve_ans_q(args, logics, evi_path_cve):
    # logics: tuple, one query
    test_facts = set()
    ans_cves = []
    for lp in logics:
        assert type(lp)==tuple
        assert type(lp[0])==int
        assert type(lp[1])==tuple
        assert lp in evi_path_cve

        cur_ans = evi_path_cve[lp]
        assert len(cur_ans) > 0
        ans_cves.append(cur_ans)
    
        # only add 1-hop facts
        if len(lp[1]) == 1:
            for a in cur_ans:
                test_facts.add((lp[0], lp[1][0], a))

    ans_cves = set.intersection(*ans_cves)
    assert len(ans_cves) > 0

    entset = pickle.load(open(os.path.join(args.kg_path, 'id_entset.pkl'), 'rb'))
    for a in ans_cves:
        assert a in entset['cve-id']
    return ans_cves, test_facts



#----------------------------------------------------------------------#
#             Below are functions used for reasoning task              #
#----------------------------------------------------------------------#

def zeroday_metrics(args, query_rank_dict, topks):
    test_q = pickle.load(open(os.path.join(args.data_path, 'test_queries_zeroday.pkl'), 'rb'))
    test_a = pickle.load(open(os.path.join(args.data_path, 'test_answers_zeroday.pkl'), 'rb'))  # CVE results
    test_a_cwe = pickle.load(open(os.path.join(args.data_path, 'test_answers_cwe_zeroday.pkl'), 'rb'))

    q2struc = {}
    for q_struc, qs in test_q.items():
        for q in qs:
            q2struc[q] = q_struc

    entid2cate = pickle.load(open(os.path.join(args.data_path, 'entid2cate.pkl'), 'rb'))
    cve_cwe_map = gen_cve_cwe_map(args.data_path)

    queries = set(test_a.keys()) & set(query_rank_dict.keys())
    query_pred_cwe_dict = defaultdict(dict)
    logging.info('Searching for predicted CWE-IDs among %d queries...' % len(queries))
    for q in queries:
        ranks = query_rank_dict[q]
        for topk in topks:
            pred_a = torch.sort(torch.tensor(ranks))[1][:topk].tolist()
            pred_cwe = [] 
            for a in pred_a:
                if entid2cate[a] == 'cve-id':
                    pred_cwe += list(cve_cwe_map[a])
            query_pred_cwe_dict[q][topk] = pred_cwe
    
    logging.info('Calculating hit ratios...')
    logs = defaultdict(list)
    for q in query_pred_cwe_dict.keys():
        results = OrderedDict()
        for topk in topks:
            ground_cwe = test_a_cwe[q]
            pred_cwe = query_pred_cwe_dict[q][topk]

           # calculate HITS@K
            hit = float(len(set(ground_cwe) & set(pred_cwe))>0)
            results.update({'HITS@'+str(topk): hit})

            # calculate NDCG@K
            binary_hit = np.zeros(len(pred_cwe)).tolist() # save a few memory and time
            for i in range(len(pred_cwe)):
                if pred_cwe[i] in ground_cwe:
                    binary_hit[i] = 1
            ndcg = ndcg_at_k(binary_hit, topk)
            results.update({'NDCG@'+str(topk): ndcg})       
            
        results.update({'num_answer': len(test_a_cwe[q])})
        logs[q2struc[q]].append(results)
    
    metrics = defaultdict(lambda: defaultdict(int))
    for query_structure in logs:
        for metric in logs[query_structure][0].keys():
            if metric in ['num_answer']:
                continue
            metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
        metrics[query_structure]['num_queries'] = len(logs[query_structure])

    return metrics
