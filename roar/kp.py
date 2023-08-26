'''
This file contains functions used for KG poisoning
'''
from collections import defaultdict
from email.policy import default
import os, sys
sys.path.append(os.path.abspath('..'))

import time
import torch
import pickle
import logging
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from main.kgr import KRL
from model.geo import KGReasoning
import gendata.cyber.cyberkg_utils as cyber
import gendata.med.medkg_utils as med
import helper.utils as util
import helper.atkutils as atkutils


def pturb_kge(args, 
            model: KGReasoning, 
            optimizer, 
            krl: KRL,
            atk_pkg: dict, 
            loaders: dict,
            eval_answers: dict):
    tar_path: tuple = atk_pkg['tar_path']
    tar_ans: int = atk_pkg['tar_ans']
    tar_A2B_r: int = atk_pkg['tar_A2B_r']

    train_loader_ben = loaders['train_loader_ben']
    train_loader_atk = loaders['train_loader_atk']
    test_loader_ben_A = loaders['test_loader_ben_%s' % krl.taskA]
    test_loader_ben_B = loaders['test_loader_ben_%s' % krl.taskB]
    if args.attack == 'kgp':
        test_loader_atk_A = loaders['test_loader_atk_%s' % krl.taskA]
        test_loader_atk_B = loaders['test_loader_atk_%s' % krl.taskB]
    elif args.attack == 'cop':
        test_loader_atk_A = loaders['eva_test_loader_atk_%s' % krl.taskA]
        test_loader_atk_B = loaders['eva_test_loader_atk_%s' % krl.taskB]

    lr = args.learning_rate
    model.entity_embedding.requires_grad = True
    for name, param in model.named_parameters():
        if 'center_net' in name or 'offset_net' in name:
            param.requires_grad = False
        if 'relation_embedding' in name or 'offset_embedding' in name:
            param.requires_grad = False
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    # set freezing entities
    kg_tar_ents = set(get_tarevi_id(tar_path))
    # if args.atk_obj == 'targeted':
    #     kg_tar_ents |= set([tar_ans])
        # if args.domain == 'cyber':
        #     A2B_dict = cyber.gen_cve_miti_dict(args.kg_path)
        #     kg_tar_ents |= A2B_dict[tar_ans]
        # elif args.domain == 'med':
        #     A2B_dict = med.gen_dz_drug_dict(args.kg_path)
        #     kg_tar_ents |= A2B_dict[tar_ans][tar_A2B_r]

    fixed_eid = list(set(range(krl.nentity)) - kg_tar_ents)
    fixed_entity_embedding = model.entity_embedding.clone().data[fixed_eid]
    training_logs = []

    # Training Loop
    for step in range(0, args.atk_steps):
        log = model.roar_train_step(
            model, optimizer, train_loader_atk, train_loader_ben, args, fixed_eid=fixed_eid, fixed_entity_embedding=fixed_entity_embedding)

        training_logs.append(log)
        if step % args.log_steps == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

            krl.log_metrics('Training average', step, metrics)
            training_logs = []

    if args.do_test:
        logging.info('')
        logging.info('-------------------- Evaluating on Roar-Optimized Surrogate KRL Model on Test Dataset...')
        logging.info('')
        krl.evaluate(model, eval_answers['test_a_ben_%s' % krl.taskA], test_loader_ben_A, util.query_name_dict, 'benign %s' % krl.taskA, step)
        krl.evaluate(model, eval_answers['test_a_ben_%s' % krl.taskB], test_loader_ben_B, util.query_name_dict, 'benign %s' % krl.taskB, step)
        if args.attack == 'kgp':
            krl.evaluate(model, eval_answers['test_a_atk_%s' % krl.taskA], test_loader_atk_A, util.query_name_dict, 'attack %s' % krl.taskA, step)
            krl.evaluate(model, eval_answers['test_a_atk_%s' % krl.taskB], test_loader_atk_B, util.query_name_dict, 'attack %s' % krl.taskB, step)
        elif args.attack == 'cop':
            krl.evaluate(model, eval_answers['eva_test_a_atk_%s' % krl.taskA], test_loader_atk_A, util.query_name_dict, 'attack %s' % krl.taskA, step)
            krl.evaluate(model, eval_answers['eva_test_a_atk_%s' % krl.taskB], test_loader_atk_B, util.query_name_dict, 'attack %s' % krl.taskB, step)

    new_facts = get_atk_facts(args, model, atk_pkg, kg_tar_ents)
    
    id_factset = pickle.load(open(os.path.join(args.kg_path, 'id_factset.pkl'), 'rb'))
    factset = pickle.load(open(os.path.join(args.kg_path, 'factset.pkl'), 'rb'))
    id2rel = pickle.load(open(os.path.join(args.atk_kg_path, 'id2rel.pkl'), 'rb'))
    id2ent = pickle.load(open(os.path.join(args.atk_kg_path, 'id2ent.pkl'), 'rb'))
    for new_f in new_facts:
        h, r, t = new_f
        id_factset[r].add(new_f)
        factset[id2rel[r]].add((id2ent[h], id2rel[r], id2ent[t]))
    with open(os.path.join(args.atk_kg_path, 'id_factset.pkl'), 'wb') as pklfile:
        pickle.dump(id_factset, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.atk_kg_path, 'factset.pkl'), 'wb') as pklfile:
        pickle.dump(factset, pklfile, protocol=pickle.HIGHEST_PROTOCOL) 

    # add new facts
    train_q = pickle.load(open(os.path.join(args.atk_q_path, "train_queries.pkl"), 'rb'))
    train_a = pickle.load(open(os.path.join(args.atk_q_path, "train_answers.pkl"), 'rb'))
    test_facts = pickle.load(open(os.path.join(args.atk_q_path, 'test_facts.pkl'), 'rb'))

    fact_dict = cyber.gen_factdict(args.atk_kg_path)
    for h, r_ts in fact_dict.items():
        for r, ts in r_ts.items():
            for t in ts:
                if (h, r, t) not in test_facts:
                    cur_q = (h, (r,))
                    cur_a = t
                    train_q[('e', ('r',))].add(cur_q)
                    train_a[cur_q].add(cur_a)
    with open(os.path.join(args.atk_q_path, 'train_queries.pkl'), 'wb') as pklfile:
        pickle.dump(train_q, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.atk_q_path, 'train_answers.pkl'), 'wb') as pklfile:
        pickle.dump(train_a, pklfile, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        train_q_pf = pickle.load(open(os.path.join(args.atk_q_path, "train_queries_poisonfact.pkl"), 'rb'))
        train_a_pf = pickle.load(open(os.path.join(args.atk_q_path, "train_answers_poisonfact.pkl"), 'rb'))
    except:
        train_q_pf = defaultdict(set)
        train_a_pf = defaultdict(set)
    for h, r, t in new_facts:
        cur_a = fact_dict[h][r]
        cur_q = (h, (r,))
        train_q_pf[('e', ('r',))].add(cur_q)
        train_a_pf[cur_q] |= cur_a
        
    with open(os.path.join(args.atk_q_path, 'train_queries_poisonfact.pkl'), 'wb') as pklfile:
        pickle.dump(train_q_pf, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.atk_q_path, 'train_answers_poisonfact.pkl'), 'wb') as pklfile:
        pickle.dump(train_a_pf, pklfile, protocol=pickle.HIGHEST_PROTOCOL)



def get_tarevi_id(tar_path):
    # NOTE: only for ..xi type targeted logic path
    assert type(tar_path[0]) == int, 'not implement targeted logic path besides e->r->r->... format'
    return [tar_path[0]]


def get_atk_facts(args,
                model: KGReasoning, 
                atk_pkg: dict,
                kg_tar_ents):
    # rel2id = pickle.load(open(os.path.join(args.atk_kg_path, 'rel2id.pkl'), 'rb'))
    id2rel = pickle.load(open(os.path.join(args.atk_kg_path, 'id2rel.pkl'), 'rb'))
    id_entset = pickle.load(open(os.path.join(args.atk_kg_path, 'id_entset.pkl'), 'rb'))
    entid2cate = pickle.load(open(os.path.join(args.atk_kg_path, 'entid2cate.pkl'), 'rb'))
    r_embeddings = model.relation_embedding.data
    if args.sur_model == 'box':
        r_offset_embeddings = model.offset_embedding.data
    # r_embeddings = model.relation_embedding.data
    # r_offset_embeddings = model.offset_embedding.data

    atk_cdd_eids = set() 
    for k, v in id_entset.items():
        atk_cdd_eids |= v

    assert len(kg_tar_ents)>0, 'need target evidence/answer entities to perturb'
    atk_cdd_eids = atk_cdd_eids - kg_tar_ents        
    
    new_facts = set()
    # cdd_rels = rel2id.items()
    for _tar_eid in kg_tar_ents:
        tar_embedding = model.entity_embedding.data[_tar_eid] # (D, )
        # cdd_rels = atkutils.get_cdd_rel(args, [entid2cate[_tar_eid]])
        cdd_rels = id2rel.keys()
        # print('\n targeted eid cate: %s \tcdd_rels: %s' % (entid2cate[_tar_eid], ', '.join([str(ele) for ele in cdd_rels])))
        fact_score_dict = {} # tuple(f, rev_f): score
        t1 = time.time()

        for _cdd_eid in atk_cdd_eids:
            cdd_embedding = model.entity_embedding.data[_cdd_eid] # (D, )
            for _rid in cdd_rels:
                # _rev_rid = cyber.get_rev_rel(rel2id, id2rel, _rid)
                r_embedding = r_embeddings[_rid]
                # r_rev_embedding = r_embeddings[_rev_rid]

                if args.sur_model == 'box':
                    r_offset_embedding = r_offset_embeddings[_rid]
                    # r_rev_offset_embedding = r_offset_embeddings[_rev_rid]
                    offset_embedding = model.func(r_offset_embedding)
                    # rev_offset_embedding = model.func(r_rev_offset_embedding)

                    logit_h = model.cal_logit_box(cdd_embedding, tar_embedding+r_embedding, offset_embedding) 
                    logit_t = model.cal_logit_box(tar_embedding, cdd_embedding+r_embedding, offset_embedding)
                    # logit += model.cal_logit_box(tar_embedding, cdd_embedding+r_rev_embedding, rev_offset_embedding)
                elif args.sur_model == 'vec':
                    logit_h = model.cal_logit_vec(cdd_embedding, tar_embedding+r_embedding) 
                    logit_t = model.cal_logit_vec(tar_embedding, cdd_embedding+r_embedding)

                # (e_tar, r, e) and (e, r, e_tar)
                fact_score_dict.update({tuple([_tar_eid, _rid, _cdd_eid]): logit_h.item()})
                fact_score_dict.update({tuple([_cdd_eid, _rid, _tar_eid]): logit_t.item()})

        print('--- time for tar_eid (%d, %s): %f seconds' % (_tar_eid, entid2cate[_tar_eid], round((time.time() - t1), 2)))
        
    top_f = sorted(fact_score_dict.keys(), key=fact_score_dict.get, reverse=True)[:args.atk_budget]
    new_facts |= set(top_f)

    print('*'*100)
    print('new_facts', new_facts)
    print('*'*100)
    return new_facts