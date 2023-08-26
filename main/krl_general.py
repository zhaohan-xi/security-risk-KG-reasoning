#!/usr/bin/python3

from email.policy import default
import os, sys
sys.path.append(os.path.abspath('..'))

import yaml
import pickle
import logging
import torch
import numpy as np
from collections import defaultdict

from config.config import parse_args
import helper.utils as util
import gendata.cyber.cyberkg_utils as cyber
from torch.utils.data import DataLoader
from loader.loader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from model.geo import KGReasoning

class KRL:
    def __init__(self, args, expt: str):
        self.args = args
        self.expt = expt

        self.cur_time = util.parse_time()
        if self.args.save_path is None:
            self.args.save_path = os.path.join(
                util.project_root_path, self.args.log_folder, self.args.domain, self.expt, self.args.model, self.cur_time)
        os.makedirs(self.args.save_path, exist_ok=True)
        print ("logging to", self.args.save_path)

        self.set_logger()
        self.update_ent_rel_num()

        logging.info('-------------------------------   KRL Setting   -------------------------------')
        logging.info('Domain: %s' % self.args.domain)
        logging.info('KG Path: %s' % self.args.kg_path)
        logging.info('QA Path: %s' % self.args.q_path)
        logging.info('# entity: %d' % self.args.nentity)
        logging.info('# relation: %d' % self.args.nrelation)
        logging.info('KRL Model: %s' % self.args.model)
        logging.info('# KRL steps: %d' % self.args.krl_train_steps)


    def run(self):
        _data = self.load_data(self.args.q_path)
        train_q = _data['train_queries']
        train_a = _data['train_answers']
        test_q = _data['test_queries']
        test_a = _data['test_answers']
        
        train_q, train_a = self.gen_facts(self.args.kg_path)

        if self.args.do_train:
            train_loader = self.gen_loader(train_q, train_a, True)

        if self.args.do_test:
            test_loader = self.gen_loader(test_q, test_a, False)

        model = KGReasoning(
            nentity=self.nentity,
            nrelation=self.nrelation,
            hidden_dim=self.args.hidden_dim,
            gamma=self.args.gamma,
            model=self.args.model,
            cen_layer=self.args.cen_layer, 
            off_layer=self.args.off_layer, 
            prj_layer=self.args.prj_layer, 
            use_cuda = self.args.cuda,
            box_mode=util.eval_tuple(self.args.box_mode),
            beta_mode = util.eval_tuple(self.args.beta_mode),
            test_batch_size=self.args.test_batch_size,
            query_name_dict = util.query_name_dict
        )

        if self.args.cuda:
            model = model.cuda()

        if self.args.do_train:
            lr = self.args.learning_rate
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=lr
            )
            # warm_up_steps = self.args.qa_train_steps // 2

        org_requires_grad = {}
        for name, param in model.named_parameters():
            org_requires_grad[name] = param.requires_grad

        if self.args.krl_ckpt_path is not None: # retrain whole model or test
            logging.info('Retrain/test with pretrained model %s...' % self.args.krl_ckpt_path)
            checkpoint = torch.load(os.path.join(self.args.krl_ckpt_path, 'krl_ckpt'))
            model.load_state_dict(checkpoint['model_state_dict'])
            if self.args.do_train:
                for name, param in model.named_parameters():
                    param.requires_grad = org_requires_grad[name]
        else:
            logging.info('Ramdomly Initializing KRL Model %s' % self.args.model)

        logging.info('Model Parameter Configuration:')
        num_params = 0
        for name, param in model.named_parameters():
            logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
            if param.requires_grad:
                num_params += np.prod(param.size())
        logging.info('Parameter Number: %d' % num_params)

        if self.args.model == 'box':
            logging.info('box mode = %s' % self.args.box_mode)
        elif self.args.model == 'beta':
            logging.info('beta mode = %s' % self.args.beta_mode)

        if self.args.do_train:
            logging.info('Start Training...')
            logging.info('learning rate = %f' % lr)
        logging.info('batch_size = %d' % self.args.batch_size)
        logging.info('hidden_dim = %d' % self.args.hidden_dim)
        logging.info('gamma = %f' % self.args.gamma)
        

        # used for valid formulation
        if self.args.do_valid:
            val_data = self.validate_preparation(self.args.kg_path)


        if self.args.do_train:
            TRAIN_STEPS = self.args.krl_train_steps
            training_logs = []

            # Training Loop
            for step in range(0, TRAIN_STEPS):
                log = model.train_step(model, optimizer, train_loader, self.args, step)
                training_logs.append(log)

                if step > 0 and step % (TRAIN_STEPS // 5) == 0:
                    lr = lr / 2
                    logging.info('Change learning rate to %f at step %d' % (lr, step))
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=lr
                    )
                
                if (step > 0 and step % self.args.save_checkpoint_steps == 0) or step==TRAIN_STEPS-1:
                    save_variable_list = {
                        'step': step, 
                        'learning_rate': lr,
                        # 'warm_up_steps': warm_up_steps
                    }
                    self.save_model(model, optimizer, save_variable_list)

                if step % self.args.valid_steps == 0 or step==TRAIN_STEPS-1:
                    if self.args.do_valid:
                        self.validate_formula(model, step, val_data)

                    if self.args.do_test:
                        logging.info('Evaluating on Test Dataset...')
                        self.evaluate(model, test_a, test_loader, util.query_name_dict, 'test', step)

                if step % self.args.log_steps == 0 or step==TRAIN_STEPS-1:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

                    self.log_metrics('Training average', step, metrics)
                    training_logs = []
            
        print('KRL Done at step %d' % step)
        logging.info('>>> KRL Done at step %d <<<' % step)
        logging.info('='*80 + '\n')

        self.model = model

        
    @staticmethod
    def load_data(q_path):
        '''
        Load queries for general QA tasks
        '''
        logging.info("loading train data from %s" % q_path)
        train_q = pickle.load(open(os.path.join(q_path, "train_queries.pkl"), 'rb'))
        train_a = pickle.load(open(os.path.join(q_path, "train_answers.pkl"), 'rb'))

        logging.info("loading benign test data from %s" % q_path)
        test_q = pickle.load(open(os.path.join(q_path, 'test_queries.pkl'), 'rb'))
        test_a = pickle.load(open(os.path.join(q_path, 'test_answers.pkl'), 'rb'))

        rst = {
            'train_queries': train_q, 'train_answers': train_a,
            'test_queries': test_q, 'test_answers': test_a,
        }
        return rst

    @staticmethod
    def gen_facts(kg_path):
        # 1-hop projection queries for knowledge graph embedding
        fact_dict = cyber.gen_factdict(kg_path)
        train_q = defaultdict(set) # {q_struc: set(nested int tuples)}
        train_a = defaultdict(set)
        for h, r_ts in fact_dict.items():
            for r, ts in r_ts.items():
                for t in ts:
                    cur_q = (h, (r,))
                    cur_a = t
                    train_q[('e', ('r',))].add(cur_q)
                    train_a[cur_q].add(cur_a)
    
        return train_q, train_a


    def gen_loader(self, queries: defaultdict(set), answers: defaultdict(set), for_train: bool, msg: str = ''):
        if len(msg) > 0: msg = ' - ' + msg
        loader = None
        if for_train:
            logging.info('')
            logging.info('Training query statistics%s' % msg)
            for query_structure in queries:
                logging.info(util.query_name_dict[query_structure]+": "+str(len(queries[query_structure])))
            loader = SingledirectionalOneShotIterator(DataLoader(
                                        TrainDataset(
                                            util.flatten_query(queries), 
                                            self.nentity, 
                                            self.nrelation, 
                                            self.args.negative_sample_size, 
                                            answers),
                                        batch_size=self.args.batch_size,
                                        shuffle=True,
                                        num_workers=self.args.cpu_num,
                                        collate_fn=TrainDataset.collate_fn
                                    ))
        else:
            logging.info('')
            logging.info('Testing query statistics%s' % msg)
            for query_structure in queries:
                logging.info(util.query_name_dict[query_structure]+": "+str(len(queries[query_structure])))
            loader = DataLoader(
                TestDataset(
                    util.flatten_query(queries), 
                    self.args.nentity, 
                    self.args.nrelation, 
                ), 
                batch_size=self.args.test_batch_size,
                num_workers=self.args.cpu_num, 
                collate_fn=TestDataset.collate_fn
            )
        return loader


    def save_model(self, model, optimizer, save_variable_list, name='krl_ckpt'):
        '''
        Save the parameters of the model and the optimizer,
        as well as some other variables such as step and learning_rate
        '''
        os.makedirs(self.args.save_path, exist_ok=True)

        argparse_dict = vars(self.args)
        with open(os.path.join(self.args.save_path, 'config.yaml'), 'w') as f:
            # json.dump(argparse_dict, f)
            yaml.dump(argparse_dict, f)

        torch.save({
            **save_variable_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(self.args.save_path, name)
        )


    def log_metrics(self, mode, step, metrics, print_log=True):
        '''
        Print the evaluation logs
        '''
        if print_log:
            for metric in metrics:
                logging.info('%s %s at step %d: %.4f' % (mode, metric, step, metrics[metric]))


    def set_logger(self):
        '''
        Write logs to console and log file
        '''
        if self.args.do_train:
            log_file = os.path.join(self.args.save_path, 'train.log')
        else:
            log_file = os.path.join(self.args.save_path, 'test.log')

        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='a+'
        )
        if self.args.print_on_screen:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)


    def update_ent_rel_num(self):
        path = self.args.atk_kg_path if self.args.under_attack else self.args.kg_path
        ent2id = pickle.load(open(os.path.join(path, "ent2id.pkl"), 'rb'))
        rel2id = pickle.load(open(os.path.join(path, "rel2id.pkl"), 'rb'))
        self.args.nentity = len(ent2id)
        self.args.nrelation = len(rel2id)
        self.nentity = self.args.nentity
        self.nrelation = self.args.nrelation

    @staticmethod
    def validate_preparation(kg_path):
        factset = pickle.load(open(os.path.join(kg_path, 'id_factset.pkl'), 'rb'))
        id2ent = pickle.load(open(os.path.join(kg_path, 'id2ent.pkl'), 'rb'))
        id2rel = pickle.load(open(os.path.join(kg_path, 'id2rel.pkl'), 'rb'))
        rids = sorted(list(id2rel.keys()))
        nfact = 0
        for _, facts in factset.items():
            nfact += len(facts)

        # topological property
        degree = defaultdict(lambda: defaultdict(int))  # {eid: {rid: degree}}
        indegree = defaultdict(lambda: defaultdict(int))  # {eid: {rid: degree}}

        factdict = cyber.gen_factdict(kg_path)
        for rid in rids:
            assert rid in factset
            
            for eid in id2ent:
                degree[eid][rid] += len(factdict[eid][rid])
                for t_eid in factdict[eid][rid]:
                    indegree[t_eid][rid] += 1
        data = {
            'rids': rids,
            'factset': factset,
            'nentity': len(id2ent),
            'degree': degree,
            'nfact': nfact,
            'indegree': indegree,
        }
        return data

    @staticmethod
    def validate_formula(model: KGReasoning, step: int, data: dict):
        """
        Note: if we set x = sigma - norm(h_r, t_r)
        we have  x = log(N * A[h][t]/(d_h * d_ht))
        """
        rids = data['rids']
        factset = data['factset']
        nentity = data['nentity']
        degree = data['degree']
        nfact = data['nfact']

        em_scores = defaultdict(list)   # {rid: [float, float, ...]}
        tp_scores = defaultdict(list)   # {rid: [float, float, ...]}
        df_scores = defaultdict(list)   # {rid: [float, float, ...]}
        for rid in rids:
            for h, r, t in factset[rid]:
                assert r == rid
                r_h_em = model.entity_embedding[h] + model.relation_embedding[r]  # (D,)
                r_t_em = model.entity_embedding[t] # (D,)

                em_score = model.gamma - torch.norm(r_h_em - r_t_em, p=1, dim=-1)
                em_scores[r].append(em_score)

                tp_score = torch.log(torch.tensor(nentity / degree[h][r]))
                # tp_score = torch.log(torch.tensor(nentity / (degree[h][r]+degree[t][r])))

                tp_scores[r].append(tp_score)
                df_scores[r].append(torch.abs(em_score - tp_score))

        df, w_df, ratio, w_ratio = [], [], [], []
        for rid in rids:
            em_s_mean = torch.mean(torch.tensor(em_scores[rid])).item()
            tp_s_mean = torch.mean(torch.tensor(tp_scores[rid])).item()
            df_s_mean = torch.mean(torch.tensor(df_scores[rid])).item()
            logging.info('Step %d\trel %d\t emb %.2f\t topo %.2f\tdiff %.2f\t diff/topo %.2f\t' % (
                step, rid, em_s_mean, tp_s_mean, df_s_mean, df_s_mean/tp_s_mean))
            df.append(df_s_mean)
            ratio.append(df_s_mean/tp_s_mean)
            w_df.append(df_s_mean * len(factset[rid]) / nfact)
            w_ratio.append((df_s_mean / tp_s_mean) * len(factset[rid]) / nfact)
        logging.info('Step %d\t mean diff %.2f\tweighted diff %.2f\t mean ratio %.2f\t weighted ratio %.2f' % (
            step, np.mean(df), np.sum(w_df), np.mean(ratio), np.sum(w_ratio)))


    def evaluate(self, model: KGReasoning, answers, dataloader, query_name_dict, mode, step, print_log=True):
        '''
        Evaluate queries in dataloader
        '''
        average_metrics = defaultdict(float)
        all_metrics = defaultdict(float)

        metrics, query_rank_dict = model.test_step(
            model, answers, self.args, dataloader, query_name_dict, use_neg_sample_idxs=None)
        num_query_structures = 0
        num_queries = 0
        for query_structure in metrics:
            self.log_metrics(mode+" "+query_name_dict[query_structure], step, metrics[query_structure], print_log=print_log)
            for metric in metrics[query_structure]:
                all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
                if metric != 'num_queries':
                    average_metrics[metric] += metrics[query_structure][metric]
            num_queries += metrics[query_structure]['num_queries']
            num_query_structures += 1

        for metric in average_metrics:
            average_metrics[metric] /= num_query_structures
            all_metrics["_".join(["average", metric])] = average_metrics[metric]
        self.log_metrics('%s average'%mode, step, average_metrics, print_log=print_log)

        # with open(os.path.join(self.args.save_path, 'query_rank_dict_%s_%d.pkl' % ('miti' if 'mitigation' in mode else 'cve' ,step)), 'wb') as pklfile:
        #     pickle.dump(query_rank_dict, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        #     print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
            
        return all_metrics

if __name__ == '__main__':
    args = parse_args()
    yml_dict = yaml.load(open(args.yaml, 'r'), Loader=yaml.FullLoader)
    kwargs= args.__dict__
    kwargs.update(**yml_dict)

    krl = KRL(args, 'krl-general')
    krl.run()