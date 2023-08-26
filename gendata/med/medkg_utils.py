from email.policy import default
import os
import random
import pickle
from collections import defaultdict
import gendata.cyber.cyberkg_utils as cyber

ent_prefix_delimiter = cyber.ent_prefix_delimiter 

# original relation name to a clean name
# between Compound:Disease, pos/neg is determined by human side
# for other pairs, pos/neg is determined by their interactions
clean_r_map = {
    # 'bioarx::HumGenHumGen:Gene:Gene': 'neutral:Gene:Gene',
    # 'bioarx::VirGenHumGen:Gene:Gene': 'neutral:Gene:Gene',
    # 'bioarx::DrugVirGen:Compound:Gene': 'neutral:Compound:Gene',
    # 'bioarx::DrugHumGen:Compound:Gene': 'neutral:Compound:Gene',
    # 'bioarx::Covid2_acc_host_gene::Disease:Gene': 'neutral:Disease:Gene',
    # 'bioarx::Coronavirus_ass_host_gene::Disease:Gene': 'neutral:Disease:Gene',
    'DRUGBANK::x-atc::Compound:Atc': 'neutral:Compound:Atc',
    'DRUGBANK::ddi-interactor-in::Compound:Compound': 'negative:Compound:Compound',
    'DRUGBANK::target::Compound:Gene': 'negative:Compound:Gene',
    'DRUGBANK::enzyme::Compound:Gene': 'positive:Compound:Gene',
    'DRUGBANK::carrier::Compound:Gene': 'negative:Compound:Gene',
    'DRUGBANK::treats::Compound:Disease': 'positive:Compound:Disease',
    'GNBR::E::Compound:Gene' : 'neutral:Compound:Gene',
    'GNBR::A+::Compound:Gene' : 'positive:Compound:Gene',
    'GNBR::N::Compound:Gene' : 'negative:Compound:Gene',
    'GNBR::K::Compound:Gene' : 'neutral:Compound:Gene',
    'GNBR::A-::Compound:Gene' : 'negative:Compound:Gene',
    'GNBR::E+::Compound:Gene' : 'positive:Compound:Gene',
    'GNBR::B::Compound:Gene' : 'negative:Compound:Gene',
    'GNBR::E-::Compound:Gene' : 'negative:Compound:Gene',
    'GNBR::O::Compound:Gene' : 'neutral:Compound:Gene',
    'GNBR::Z::Compound:Gene' : 'positive:Compound:Gene',
    'GNBR::T::Compound:Disease' : 'positive:Compound:Disease',
    'GNBR::C::Compound:Disease' : 'positive:Compound:Disease',
    'GNBR::Sa::Compound:Disease' : 'negative:Compound:Disease',
    'GNBR::Pa::Compound:Disease' : 'positive:Compound:Disease',
    'GNBR::Mp::Compound:Disease' : 'neutral:Compound:Disease',
    'GNBR::Pr::Compound:Disease' : 'positive:Compound:Disease',
    'GNBR::J::Compound:Disease' : 'negative:Compound:Disease',
    'GNBR::L::Gene:Disease' : 'negative:Gene:Disease',
    'GNBR::U::Gene:Disease' : 'neutral:Gene:Disease',
    'GNBR::Y::Gene:Disease' : 'neutral:Gene:Disease',
    'GNBR::J::Gene:Disease' : 'positive:Gene:Disease',
    'GNBR::Te::Gene:Disease' : 'negative:Gene:Disease',
    'GNBR::Md::Gene:Disease' : 'neutral:Gene:Disease',
    'GNBR::G::Gene:Disease' : 'positive:Gene:Disease',
    'GNBR::D::Gene:Disease' : 'negative:Gene:Disease',
    'GNBR::X::Gene:Disease' : 'positive:Gene:Disease',
    'GNBR::Ud::Gene:Disease' : 'positive:Gene:Disease',
    'GNBR::V+::Gene:Gene' : 'positive:Gene:Gene',
    'GNBR::Q::Gene:Gene' : 'positive:Gene:Gene',
    'GNBR::Rg::Gene:Gene' : 'negative:Gene:Gene',
    'GNBR::B::Gene:Gene' : 'negative:Gene:Gene',
    'GNBR::I::Gene:Gene' : 'neutral:Gene:Gene',
    'GNBR::E+::Gene:Gene' : 'positive:Gene:Gene',
    'GNBR::H::Gene:Gene' : 'neutral:Gene:Gene',
    'GNBR::W::Gene:Gene' : 'positive:Gene:Gene',
    'GNBR::E::Gene:Gene' : 'neutral:Gene:Gene',
    'GNBR::in_tax::Gene:Tax' : 'neutral:Gene:Tax',
    'Hetionet::GpBP::Gene:Biological Process' : 'neutral:Gene:Biological Process',
    # 'Hetionet::GiG::Gene:Gene' : 'neutral:Gene:Gene'
    'Hetionet::CrC::Compound:Compound' : 'neutral:Compound:Compound',
    'Hetionet::DdG::Disease:Gene' : 'negative:Disease:Gene',
    'Hetionet::DpS::Disease:Symptom' : 'neutral:Disease:Symptom',
    'Hetionet::DlA::Disease:Anatomy' : 'neutral:Disease:Anatomy',
    'Hetionet::CtD::Compound:Disease' : 'positive:Compound:Disease',
    'Hetionet::CbG::Compound:Gene' : 'negative:Compound:Gene',
    'Hetionet::CuG::Compound:Gene' : 'positive:Compound:Gene',
    'Hetionet::DrD::Disease:Disease' : 'neutral:Disease:Disease',
    'Hetionet::DaG::Disease:Gene' : 'neutral:Disease:Gene',
    'Hetionet::CpD::Compound:Disease' : 'positive:Compound:Disease',
    'Hetionet::AdG::Anatomy:Gene' : 'negative:Anatomy:Gene',
    'Hetionet::AuG::Anatomy:Gene' : 'positive:Anatomy:Gene',
    # 'Hetionet::GcG::Gene:Gene' : 'neutral:Gene:Gene',
    'Hetionet::GpMF::Gene:Molecular Function' : 'neutral:Gene:Molecular Function',
    'Hetionet::PCiC::Pharmacologic Class:Compound' : 'neutral:Pharmacologic Class:Compound',
    'Hetionet::GpCC::Gene:Cellular Component' : 'neutral:Gene:Cellular Component',
    # 'Hetionet::Gr>G::Gene:Gene' : 'negative:Gene:Gene',
    'Hetionet::CdG::Compound:Gene' : 'negative:Compound:Gene',
    'Hetionet::DuG::Disease:Gene' : 'positive:Disease:Gene',
    'Hetionet::GpPW::Gene:Pathway' : 'neutral:Gene:Pathway',
    # 'Hetionet::CcSE::Compound:Side Effect' : 'positive:Compound:Side Effect',
    # 'Hetionet::AeG::Anatomy:Gene' :  'neutral:Anatomy:Gene',
}
#----------------------------------------------------------------------#
#            Below are functions used for query generation             #
#----------------------------------------------------------------------#

# NOTE: manually setup the exact entity categories as in entset.pkl

dz_1h_evi_cate = ['Gene', 'Anatomy', 'Symptom'] 
dz_2h_evi_cate = ['Pathway', 'Molecular Function', 'Biological Process', 'Tax', 'Cellular Component']


def get_dz_1h_evi_path(kg_path, evi_path_dz: defaultdict):
    # return: 
    #   dz_1h_evi_path: {dz (int): set( (evi_id, (rid,)) )}
    factset = pickle.load(open(os.path.join(kg_path, 'id_factset.pkl'), 'rb'))
    entset = pickle.load(open(os.path.join(kg_path, 'id_entset.pkl'), 'rb'))
    rel2id = pickle.load(open(os.path.join(kg_path, 'rel2id.pkl'), 'rb'))
    rel_dict = pickle.load(open(os.path.join(kg_path, 'rel_dict.pkl'), 'rb'))

    all_evi_ents = set()
    for cate in dz_1h_evi_cate:
        all_evi_ents |= entset[cate]

    dz_1h_rid = set()
    for evi_cate in dz_1h_evi_cate:
        dz_1h_rid |= set([rel2id[r_name] for r_name in rel_dict[evi_cate+':Disease']])
    
    dz_1h_evi_path = defaultdict(set)
    for rid in dz_1h_rid:
        for h, r, t in factset[rid]:
            assert r == rid
            assert h in all_evi_ents
            assert t in entset['Disease']
            dz_id, evi_path = t, (h, (r,))
            dz_1h_evi_path[dz_id].add(evi_path)
            evi_path_dz[evi_path].add(dz_id)

    return dz_1h_evi_path, evi_path_dz


def get_dz_2h_evi_path(kg_path, evi_path_dz: defaultdict):
    # NOTE: assume all evi pass through Gene entity
    # return: 
    #   dz_2h_evi_path: {dz (int): set( (evi_id, (rid, rid)) )}
    factset = pickle.load(open(os.path.join(kg_path, 'id_factset.pkl'), 'rb'))
    entset = pickle.load(open(os.path.join(kg_path, 'id_entset.pkl'), 'rb'))
    rel2id = pickle.load(open(os.path.join(kg_path, 'rel2id.pkl'), 'rb'))
    rel_dict = pickle.load(open(os.path.join(kg_path, 'rel_dict.pkl'), 'rb'))

    all_evi_ents = set()
    for cate in dz_2h_evi_cate:
        all_evi_ents |= entset[cate]

    gene_dz_dict = defaultdict(lambda: defaultdict(set))  # {gene_id: {rid: set(dz_id)}}
    for r_name in rel_dict['Gene:Disease']:
        rid = rel2id[r_name]
        for h, r, t in factset[rid]:
            assert r == rid
            assert h in entset['Gene']
            assert t in entset['Disease']
            gene_dz_dict[h][r].add(t)

    evi_gene_rid = set()
    for evi_cate in dz_2h_evi_cate:
        evi_gene_rid |= set([rel2id[r_name] for r_name in rel_dict[evi_cate+':Gene']])
    
    dz_2h_evi_path = defaultdict(set)
    for eg_rid in evi_gene_rid:
        for h, r, t in factset[eg_rid]:
            assert r == eg_rid
            assert h in all_evi_ents
            assert t in entset['Gene']
            if t in gene_dz_dict:
                for gd_rid, dz_ids in gene_dz_dict[t].items():
                    evi_path = (h, (eg_rid, gd_rid))
                    for dz_id in dz_ids:
                        dz_2h_evi_path[dz_id].add(evi_path)
                        evi_path_dz[evi_path].add(dz_id)
    return dz_2h_evi_path, evi_path_dz


def gen_dz_drug_dict(kg_path): # int dict
    factset = pickle.load(open(os.path.join(kg_path, 'id_factset.pkl'), 'rb'))
    entset = pickle.load(open(os.path.join(kg_path, 'id_entset.pkl'), 'rb'))
    rel2id = pickle.load(open(os.path.join(kg_path, 'rel2id.pkl'), 'rb'))
    rel_dict = pickle.load(open(os.path.join(kg_path, 'rel_dict.pkl'), 'rb'))

    dz_drug_dict = defaultdict(lambda: defaultdict(set))  # {dz_id: {rid: set(drug_id)}}
    for r_name in rel_dict['Disease:Compound']:
        assert r_name in rel2id
        rid = rel2id[r_name]
        for h, r, t in factset[rid]:
            assert h in entset['Disease']
            assert t in entset['Compound']
            assert r == rid
            dz_drug_dict[h][r].add(t)

    return dz_drug_dict


# generate single query
def gen_1_xi_q(dz_1h_evi_path: dict,
            dz_id: int,
            x: int,
            must_have_logics=[]):
    assert x >= 0, 'path number cannot be negative value'
    logics, rst_x = [], x

    for p in must_have_logics:   # targeted logic path
        assert type(p[0]) == int
        assert type(p[1]) == tuple
        assert len(p[1]) == 1
        logics.append(p)
        rst_x -= 1

    # if subset == 'test':
    if rst_x > 0:
        all_1h_path = dz_1h_evi_path[dz_id] - set(logics)
        if len(all_1h_path) < rst_x: 
            return None # not enough logical paths for building query
        to_add_path = random.sample(list(all_1h_path), rst_x)
        for p in to_add_path:
            assert type(p) == tuple
            logics.append(p)

    logics = tuple(logics)
    assert len(logics) == x, logics

    return logics


def gen_1_npp_xi_q(dz_1h_evi_path: dict,
                dz_2h_evi_path: dict,
                dz_id: int,
                x: int,                 # all evidence num
                n: int,
                must_have_logics=[]):
    assert x >= n and n >= 0, 'path number cannot be negative value'

    xi_must_have_logics = []
    for p in must_have_logics:
        assert type(p[0]) == int
        assert type(p[1]) == tuple
        if len(p[1]) == 1: 
            xi_must_have_logics.append(p)

    must_have_logics = list(set(must_have_logics) - set(xi_must_have_logics))

    logics = gen_1_xi_q(dz_1h_evi_path, dz_id, x-n, must_have_logics=xi_must_have_logics)
    if logics is None: return None
    logics = list(logics)

    for p in must_have_logics:  # targeted logic path
        assert type(p[0]) == int
        assert type(p[1]) == tuple
        assert len(p[1]) == 2
        logics = [p] + logics   # put longer path ahead
        n -= 1

    all_2h_path = list(dz_2h_evi_path[dz_id])
    if n > 0 and len(all_2h_path) < n: return None
    
    toadd_2h_path = random.sample(all_2h_path, n)

    logics = toadd_2h_path + logics  # put longer path ahead
    logics = tuple(logics)
    assert len(logics) == x, logics
    return logics


def gen_dz_ans_q(args, logics, evi_path_dz):
    # logics: tuple, one query
    test_facts = set()
    ans_dzs = []  
    for lp in logics:
        assert type(lp)==tuple
        assert type(lp[0])==int
        assert type(lp[1])==tuple
        assert lp in evi_path_dz

        cur_ans = evi_path_dz[lp]
        assert len(cur_ans) > 0
        ans_dzs.append(cur_ans)
    
        # only add 1-hop facts
        if len(lp[1]) == 1:
            for a in cur_ans:
                test_facts.add((lp[0], lp[1][0], a))

    ans_dzs = set.intersection(*ans_dzs)
    assert len(ans_dzs) > 0

    entset = pickle.load(open(os.path.join(args.kg_path, 'id_entset.pkl'), 'rb'))
    for a in ans_dzs:
        assert a in entset['Disease']
    return ans_dzs, test_facts