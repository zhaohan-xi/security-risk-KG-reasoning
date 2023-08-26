import random
from tqdm import tqdm
from collections import defaultdict

#---------> query name to struc (follow Q2B manner)

# pure projection
def get_xp_struc(x: int):  
    return tuple(('e', tuple(['r' for _ in range(x)])))

# pure conjunction
def get_xi_struc(x: int):  
    return tuple([('e', ('r',)) for _ in range(x)])

def get_xip_struc(x: int):  
    return tuple((get_xi_struc(x), ('r',)))
    
# 2 & 1-hop projection, then conjunction
def get_npp_xi_struc(n: int, x: int): 
    assert n <= x
    struc = list(get_xi_struc(x))
    for i in range(n):
        struc[i] = ('e', ('r', 'r',))
    return tuple(struc)

def get_npp_xip_struc(n: int, x: int): 
    return tuple((get_npp_xi_struc(n, x), ('r',)))

# # 3 & 1-hop projection, then conjunction
# def gen_nppp_xi_struc(n: int, x: int): 
#     assert n <= x
#     struc = list(get_xi_struc(x))
#     for i in range(n):
#         struc[i] = ('e', ('r', 'r', 'r'))
#     return tuple(struc)

# def gen_nppp_xip_struc(n: int, x: int): 
#     return tuple((gen_nppp_xi_struc(n, x), ('r',)))

# # 3 & 2 & 1-hop projection, then conjunction
# def gen_nppp_mpp_xi_struc(n: int, m: int, x: int):  
#     assert n+m <= x
#     struc = list(get_npp_xi_struc(n+m, x))
#     for i in range(n):
#         struc[i] = ('e', ('r', 'r', 'r',))
#     return tuple(struc)

# def gen_nppp_mpp_xip_struc(n: int, m: int, x: int): 
#     return tuple((gen_nppp_mpp_xi_struc(n, m, x), ('r',)))


#-----------> sample query from general KG 

def get_inout_map(entset: set[int], factset: set[tuple[int]]):
    inmap, outmap = defaultdict(dict), defaultdict(dict)  # {eid: {rid: set(eids)}}
    for h, r, t in factset:
        assert h in entset
        assert t in entset
        if r not in inmap[t]:
            inmap[t][r] = set()
        if r not in outmap[h]:
            outmap[h][r] = set()
        inmap[t][r].add(h)
        outmap[h][r].add(t)
    return inmap, outmap 
        

def gen_xp_query(outmap: defaultdict(dict), x: int, num: int, subset: str, verbose: bool):
    assert not (subset=='train' and x==1), '1-hop projection train query does not need generation function'
    queries, answers = defaultdict(set), defaultdict(set)  # struc_q_dict, q_a_dict
    testfact = set()
    cdd_eids = random.sample(list(outmap.keys()), num) if len(outmap.keys()) >= num else random.choices(list(outmap.keys()), k=num)
    for eid in tqdm(cdd_eids, desc='generating %s %s query' % (str(x)+'p', subset), disable=not verbose):
        r_path, nxt_eid = [], [eid]
        while len(r_path) < x:
            cur_eid_cdd = [_id for _id in nxt_eid if len(outmap[_id])>0] # next nodes who have out edge
            if len(cur_eid_cdd) == 0:
                break
            cur_eid = random.choice(cur_eid_cdd)
            r = random.choice(list(outmap[cur_eid].keys()))
            r_path.append(r)
            nxt_eid = outmap[cur_eid][r]

        if len(r_path) == x:
            cur_q = (eid, tuple(r_path))
            queries[get_xp_struc(x)].add(cur_q)

    for struc, qs in queries.items(): 
        for q in tqdm(qs, desc='finding answers for %s %s query' % (str(x)+'p', subset), disable=not verbose):
            eid, rs = q  # 'xp' structure
            cur_eid = set([eid])
            for r in rs:
                nxt_eid = set()
                for eid in cur_eid:
                    if r not in outmap[eid]:
                        continue
                    nxt_eid |= outmap[eid][r]
                cur_eid = nxt_eid
            assert len(cur_eid) > 0
            answers[q] |= cur_eid
    # we ignore adding testfact in xp query generation

    return queries, answers, testfact


def gen_xi_query(inmap: defaultdict(dict), outmap: defaultdict(dict), x: int, num: int, subset: str, verbose: bool):
    queries, answers = defaultdict(set), defaultdict(set)  # struc_q_dict, q_a_dict
    testfact = set()
    cdd_eids = set([eid for eid, v in inmap.items() if len(v)>=x]) # ignore if a node has multiple same-type edge
    if len(cdd_eids) == 0: 
        return queries, answers, testfact

    cdd_eids = random.sample(list(cdd_eids), num) if len(cdd_eids) >= num else random.choices(list(cdd_eids), k=num)
    for eid in tqdm(cdd_eids, desc='generating %s %s query' % (str(x)+'i', subset), disable=not verbose):
        cur_q = []
        rs = random.sample(list(inmap[eid].keys()), x)
        for r in rs:
            src_eid = random.choice(list(inmap[eid][r]))
            cur_q.append((src_eid, (r,)))
        cur_q = tuple(cur_q)
        queries[get_xi_struc(x)].add(cur_q)

    # find all answer
    for struc, qs in queries.items():
        for q in tqdm(qs, desc='finding answers for %s %s query' % (str(x)+'i', subset), disable=not verbose):
            ans = []
            for path in q:
                eid = path[0]
                r = path[1][0]
                ans.append(outmap[eid][r])
            ans = set.intersection(*ans)
            assert len(ans) > 0
            answers[q] |= ans
            if subset == 'test':
                for path in q:
                    eid = path[0]
                    r = path[1][0]
                    testfact |= set([(eid, r, a) for a in ans])
    
    return queries, answers, testfact


def gen_npp_xi_query(inmap: defaultdict(dict), outmap: defaultdict(dict), n: int, x: int, num: int, subset: str, verbose: bool):
    assert n<=x
    queries, answers = defaultdict(set), defaultdict(set)  # struc_q_dict, q_a_dict
    testfact = set()
    cdd_eids = set([eid for eid, v in inmap.items() if len(v)>=x]) # ignore if a node has multiple same-type edge
    if len(cdd_eids) == 0: 
        return queries, answers, testfact

    cdd_eids = random.sample(list(cdd_eids), num) if len(cdd_eids) >= num else random.choices(list(cdd_eids), k=num)
    for eid in tqdm(cdd_eids, desc='generating %s %s query' % (str(n)+'pp.'+str(x)+'i', subset), disable=not verbose):
        cur_q = []
        rs = random.sample(list(inmap[eid].keys()), x)
        for r in rs:
            src_eid = random.choice(list(inmap[eid][r]))
            cur_q.append((src_eid, (r,)))

        n_cnt = 0
        for i, (src_eid, rs) in enumerate(cur_q):
            if n_cnt >= n: break
            assert len(rs) == 1
            r2 = rs[0]
            r1 = random.choice(list(inmap[src_eid].keys()))
            src_eid = random.choice(list(inmap[src_eid][r1])) 
            cur_q[i] = (src_eid, (r1, r2))
            n_cnt += 1

        if n_cnt == n: # find enough npp path
            cur_q = tuple(cur_q)
            queries[get_npp_xi_struc(n, x)].add(cur_q)
    
    # find all answer
    for struc, qs in queries.items():
        for q in tqdm(qs, desc='finding answers for %s %s query' % (str(n)+'pp.'+str(x)+'i', subset), disable=not verbose):
            ans = []
            for path in q:
                eids = set([path[0]])
                rs = path[1]
                for r in rs:
                    ans_curpath = set()
                    for eid in eids:
                        if r not in outmap[eid]:
                            continue # no go through this path
                        ans_curpath |= outmap[eid][r]
                        if subset == 'test':
                            testfact |= set([(eid, r, a) for a in outmap[eid][r]])
                    eids = ans_curpath
                ans.append(ans_curpath)
            ans = set.intersection(*ans)
            assert len(ans) > 0
            answers[q] |= ans
    
    return queries, answers, testfact

def extend_xi_to_xip(queries: defaultdict(set), answers: defaultdict(set), outmap: defaultdict(dict), subset: str, verbose: bool):
    xip_queries, xip_answers, testfact = defaultdict(set), defaultdict(set), set()
    if len(queries) == 0 or len(answers) == 0:
        return xip_queries, xip_answers, testfact    
        
    for struc, qs in queries.items():
        xip_struc = (struc, ('r',))
        for q in tqdm(qs, desc='extending ..xi to ..xip', disable=not verbose):
            ans = answers[q]
            assert len(ans) > 0
            cdd_rs = [set(outmap[a].keys()) for a in ans]
            r2freq = defaultdict(int)
            for r_in_1path in cdd_rs:
                for r in r_in_1path:
                    r2freq[r] += 1
            freq2r = defaultdict(set)
            for r, freq in r2freq.items():
                freq2r[freq].add(r)
            r = random.choice(list(freq2r[max(freq2r.keys())]))
            xip_q = (q, (r,))
            xip_queries[xip_struc].add(xip_q)
            
            # find all answers
            xip_a = set()
            for a in ans:
                if r in outmap[a]:
                    xip_a |= outmap[a][r]
                    if subset == 'test':
                        testfact |= set([(a, r, _a) for _a in outmap[a][r]])
            xip_answers[xip_q] |= xip_a
            
    return xip_queries, xip_answers, testfact    


