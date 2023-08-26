'''
This file contains functions used for query perturbation (evasion)
'''
from collections import defaultdict
import os, sys
sys.path.append(os.path.abspath('..'))

import copy
import pickle
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn

import gendata.cyber.cyberkg_utils as cyber
import gendata.med.medkg_utils as med
import helper.atkutils as atkutils
from helper.utils import query_name_dict, flatten
from model.geo import KGReasoning


# TODO: consider multiple hop evasion logic paths
    # step1: args.eva_evi_cates
    # step2: for each cate in args.eva_evi_cates, find cdd_evi_eids and specific rpath
    # step3: change the search_box function
def evasion(args, 
            model: KGReasoning, 
            test_q_atk: defaultdict(set), 
            test_a_atk: defaultdict(set),
            tar_ans: int = None,
            tar_A2B_r: int = None,
            ):
    ''' Evasion attacker's jobs:
        - optimize best embeddings and search the trigger structure
        - build new test_q/a set with trigger (not save)
        - evaluate under different QA steps (use saved model)
        NOTE: only consider xi and xip struc
              for xip struc, attach eva_q at its xi part 
              (since the A2B path is already set as tar_A2B_r in init)
    '''
    S = args.eva_num if args.under_attack else args.eva_num_at
    model.eval()
    
    q2struc = {}
    for q_struc, qs in test_q_atk.items():
        for q in qs:
            q2struc[q] = q_struc

    id2ent = pickle.load(open(os.path.join(args.atk_kg_path, 'id2ent.pkl'), 'rb'))
    id2rel = pickle.load(open(os.path.join(args.atk_kg_path, 'id2rel.pkl'), 'rb'))
    rel2id = pickle.load(open(os.path.join(args.atk_kg_path, 'rel2id.pkl'), 'rb'))
    # cdd_eids = list(set(id_entset['campaign']) 
    #                 | set(id_entset['product']) 
    #                 | set(id_entset['version']) 
    #             )
    # cdd_rids = [rel2id[r_name] for r_name in atkutils.evi2cve_rels]
    if args.domain == 'cyber':
        A2B_dict = cyber.gen_cve_miti_dict(args.kg_path)
    elif args.domain == 'med':
        A2B_dict = med.gen_dz_drug_dict(args.kg_path)

    cdd_eids = list(id2ent.keys())
    cdd_rids = list(id2rel.keys())

    eva_test_q_atk, eva_test_a_atk = defaultdict(set), defaultdict(set)
    for q, ans in tqdm(test_a_atk.items(), desc='generating evasion trigger', disable=not args.verbose):
        A = ans
        q_struc = q2struc[q]
        struc_name = query_name_dict[q_struc]
        xi_q = None
        xi_q_struc = copy.deepcopy(q_struc)
        struc_type = None

        if args.atk_obj == 'targeted': # NOTE: on both xi/xip struc, use the taskA tar_ans as obj
            A = [tar_ans]

        if struc_name.endswith('i'):    # ..xi
            evi_num = len(q)
            xi_q = q
            struc_type = 'xi'
            # if args.atk_obj == 'targeted':
            #     A = [tar_ans]
        elif struc_name.endswith('ip'): # ..xip
            evi_num = len(q[0]) 
            xi_q = q[0]  # xip --> xi
            xi_q_struc = copy.deepcopy(q_struc[0])
            struc_type = 'xip'
            # if args.atk_obj == 'targeted':
            #     if args.domain == 'cyber':
            #         A = A2B_dict[tar_ans]
            #     elif args.domain == 'med':
            #         A = A2B_dict[tar_ans][tar_A2B_r]

            # xi-->p path is 1-hop A2B rel
            A2B_r_center_embedding = model.relation_embedding[tar_A2B_r].unsqueeze(0)  # (1, D)
            if args.sur_model == 'box':
                A2B_r_offset_embedding = model.offset_embedding[tar_A2B_r].unsqueeze(0)    # (1, D)
        else:
            raise NotImplementedError('we dont support other query structures besides ...xi & ...xip')
        # eva_evi_num = int(np.ceil(evi_num * args.trigger_ratio))

        # NOTE: only for '...xi' & '...xip' query
        # NOTE: only consider GQE and Query2Box
        eva_e_center_embedding = nn.Parameter(torch.zeros(S, model.entity_dim))
        eva_r_center_embedding = nn.Parameter(torch.zeros(S, model.entity_dim))
        eva_r_offset_embedding = nn.Parameter(torch.zeros(S, model.entity_dim))  # Query2Box
        nn.init.uniform_(
                tensor=eva_e_center_embedding, 
                a=-model.embedding_range.item(), 
                b=model.embedding_range.item()
            )
        nn.init.uniform_(
                tensor=eva_r_center_embedding, 
                a=-model.embedding_range.item(), 
                b=model.embedding_range.item()
            )
        nn.init.uniform_(  # Query2Box
            tensor=eva_r_offset_embedding, 
            a=0., 
            b=model.embedding_range.item()
        )
        answer_embedding = model.entity_embedding[list(A)].unsqueeze(1) # (ans_num, 1, D)
        ans_num = len(answer_embedding)

        if args.cuda:
            eva_e_center_embedding = eva_e_center_embedding.detach().cuda().requires_grad_()
            eva_r_center_embedding = eva_r_center_embedding.detach().cuda().requires_grad_()
            eva_r_offset_embedding = eva_r_offset_embedding.detach().cuda().requires_grad_()  # Query2Box
            answer_embedding = answer_embedding.detach().cuda()

        eva_optimizer = torch.optim.Adam([eva_e_center_embedding, 
                                        eva_r_center_embedding, 
                                        eva_r_offset_embedding
                                        ], 
                                        lr=args.learning_rate)

        for _ in range(args.eva_optim_steps):
            assert struc_type in ['xi', 'xip'], 'not support evasion on other query types'
            eva_optimizer.zero_grad()

            # print(torch.sum(eva_e_center_embedding), torch.sum(eva_r_center_embedding), torch.sum(eva_r_offset_embedding))
            
            center_embedding_list = []
            offset_embedding_list = [] # Query2Box

            # for i in range(len(xi_q)): # NOTE: already changed to ..xi struc whatever the org format
            flatten_xi_q = torch.LongTensor(flatten(xi_q)).unsqueeze(0) # make it 2D as an one-instance "batch"
            if args.cuda: flatten_xi_q = flatten_xi_q.cuda()
            if args.sur_model == 'box':
                cen_embed, off_embed, _ = model.embed_query_box(flatten_xi_q, 
                                                                xi_q_struc,
                                                                0)
                center_embedding_list.append(cen_embed)
                offset_embedding_list.append(off_embed)
            elif args.sur_model == 'vec':
                cen_embed, _ = model.embed_query_vec(flatten_xi_q, 
                                                    xi_q_struc,
                                                    0)
                center_embedding_list.append(cen_embed)

            for i in range(S):
                center_embedding_list.append(eva_e_center_embedding[i].unsqueeze(0) + eva_r_center_embedding[i].unsqueeze(0))
                if args.sur_model == 'box':
                    offset_embedding_list.append(model.func(eva_r_offset_embedding[i]).unsqueeze(0))

            query_center_embedding = model.center_net(torch.stack(center_embedding_list))  # (1, D)
            if args.sur_model == 'box':
                query_offset_embedding = model.offset_net(torch.stack(offset_embedding_list))  # (1, D)

            # if struc_type == 'xip':
            #     query_center_embedding = query_center_embedding + A2B_r_center_embedding
            #     if args.sur_model == 'box':
            #         query_offset_embedding = query_offset_embedding + model.func(A2B_r_offset_embedding)

            all_center_embeddings = torch.cat([query_center_embedding], dim=0).unsqueeze(1).repeat(ans_num, 1, 1) # (ans_num, 1, D)
            if args.sur_model == 'box':
                all_offset_embeddings = torch.cat([query_offset_embedding], dim=0).unsqueeze(1).repeat(ans_num, 1, 1) # (ans_num, 1, D)
                    
            if args.sur_model == 'box':
                positive_logit = model.cal_logit_box(answer_embedding, all_center_embeddings, all_offset_embeddings) # (ans_num, 1)
            elif args.sur_model == 'vec':
                positive_logit = model.cal_logit_vec(answer_embedding, all_center_embeddings)
            
            loss = torch.mean(positive_logit) if args.atk_obj=='untargeted' else -torch.mean(positive_logit)
            loss.backward()
            eva_optimizer.step()  

        tri_q, tri_q_struc = inversion(args, model, eva_e_center_embedding, eva_r_center_embedding, 
                                                    eva_r_offset_embedding, cdd_eids, cdd_rids)
        # NOTE: add ppp first, then pp, then p
        eva_q_struc = None
        if struc_type == 'xi':
            eva_q = add_q_or_qstruc_in_correct_pos(q, tri_q)
            eva_q_struc = add_q_or_qstruc_in_correct_pos(q_struc, tri_q_struc)
        elif struc_type == 'xip':
            eva_q = (add_q_or_qstruc_in_correct_pos(q[0], tri_q), q[1])
            eva_q_struc = (add_q_or_qstruc_in_correct_pos(q_struc[0], tri_q_struc), q_struc[1])
        assert eva_q_struc is not None, 'testset contains other query structures'
        eva_test_q_atk[eva_q_struc].add(eva_q)
        eva_test_a_atk[eva_q] = ans

    return eva_test_q_atk, eva_test_a_atk
    

# TODO: another way is consider all e+r such that e->func(r) close to q_embed
def inversion(args, model: KGReasoning, e_center_embedding, r_center_embedding, r_offset_embedding, cdd_eids, cdd_rids):
    # NOTE: whatever org query structure, here we only match embedding distance
    # thus need to transfer query to single embeddings as inputs
    cdd_e_cen_embeddings = model.entity_embedding[cdd_eids]
    cdd_r_cen_embeddings = model.relation_embedding[cdd_rids]
    if args.sur_model == 'box':
        cdd_r_off_embeddings = model.offset_embedding[cdd_rids]

    eva_evi_num = len(e_center_embedding)
    tri_q = []
    tri_q_struc = tuple([('e', ('r',)) for _ in range(eva_evi_num)]) 
    for i in range(eva_evi_num):
        cur_e_cen_embedding = e_center_embedding[i].unsqueeze(0)  # (1, D)
        cur_r_cen_embedding = r_center_embedding[i].unsqueeze(0)  # (1, D)
        cur_r_off_embedding = r_offset_embedding[i].unsqueeze(0)  # (1, D)

        e_dist = torch.cdist(cur_e_cen_embedding, cdd_e_cen_embeddings).squeeze(0)  # (cdd_num)
        r_dist = torch.cdist(cur_r_cen_embedding, cdd_r_cen_embeddings).squeeze(0)  # (cdd_num)
        if args.sur_model == 'box':
             r_dist *= torch.cdist(cur_r_off_embedding, cdd_r_off_embeddings).squeeze(0)  # (cdd_num)

        tri_e = cdd_eids[int(torch.argsort(e_dist, descending=False)[0].item())]
        tri_r = cdd_rids[int(torch.argsort(r_dist, descending=False)[0].item())]
        tri_q.append((tri_e, (tri_r,)))

    print('*'*100)
    print('bait evidence', tri_q)
    print('*'*100)
    return tuple(tri_q), tri_q_struc


def add_q_or_qstruc_in_correct_pos(org_q: tuple, eva_q: tuple):
    # support both q_struc and int-based q
    # NOTE: input q/q_struc must be ...xi type
    rst_q = []
    len2logic = defaultdict(list)
    for l in list(org_q) + list(eva_q):
        assert type(l[0]) == int or type(l[0]) == str
        assert type(l[1]) == tuple and len(l[1]) >= 1 
        len2logic[len(l[1])].append(l)
    for k in sorted(list(len2logic.keys()), reverse=True):  # descending length
        for l in len2logic[k]:
            rst_q.append(l)
    assert len(org_q) + len(eva_q) == len(rst_q), '%d, %d, %d' % (len(org_q), len(eva_q), len(rst_q))

    return tuple(rst_q)