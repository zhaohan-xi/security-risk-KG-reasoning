# KRL for task-AB QA tasks

from email.policy import default
import os, sys
sys.path.append(os.path.abspath('..'))

import yaml
import shutil
import pickle
import logging
import torch
import numpy as np
from datetime import date, datetime
from collections import defaultdict

from config.config import parse_args
import helper.utils as util
from gendata.cyber.cyberkg_utils import zeroday_metrics
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

        if self.args.domain == 'cyber':
            self.taskA = 'cve'
            self.taskB = 'miti'
        elif self.args.domain == 'med':
            self.taskA = 'dz'
            self.taskB = 'drug'
        else:
            raise NotImplementedError('Not implemented other domain for KRL')

        logging.info('-------------------------------   KRL Setting   -------------------------------')
        logging.info('Domain: %s' % self.args.domain)
        logging.info('KG Path: %s' % self.args.kg_path)
        logging.info('QA Path: %s' % self.args.q_path)
        logging.info('# entity: %d' % self.args.nentity)
        logging.info('# relation: %d' % self.args.nrelation)
        logging.info('KRL Model: %s' % self.args.model)
        logging.info('# KRL steps: %d' % self.args.krl_train_steps)
    

    def run(self, eva_test_set: dict = None):
        _data = self.load_data()
        train_q = _data['train_queries']
        train_a = _data['train_answers']
        # train_q_pf = _data['train_queries_pf']
        # train_a_pf = _data['train_answers_pf']
        test_q_ben_A = _data['test_queries_ben_%s' % self.taskA]
        test_a_ben_A = _data['test_answers_ben_%s' % self.taskA]
        test_q_ben_B = _data['test_queries_ben_%s' % self.taskB]
        test_a_ben_B = _data['test_answers_ben_%s' % self.taskB]
        test_q_atk_A = _data['test_queries_atk_%s' % self.taskA] if eva_test_set is None else eva_test_set['eva_test_q_atk_%s' % self.taskA]
        test_a_atk_A = _data['test_answers_atk_%s' % self.taskA] if eva_test_set is None else eva_test_set['eva_test_a_atk_%s' % self.taskA]
        test_q_atk_B = _data['test_queries_atk_%s' % self.taskB] if eva_test_set is None else eva_test_set['eva_test_q_atk_%s' % self.taskB]
        test_a_atk_B = _data['test_answers_atk_%s' % self.taskB] if eva_test_set is None else eva_test_set['eva_test_a_atk_%s' % self.taskB]

        if self.args.do_train:
            train_loader = self.gen_loader(train_q, train_a, True)

            train_q_f = defaultdict(set)
            train_a_f = defaultdict(set)
            train_q_f[('e', ('r',))] = train_q[('e', ('r',))]
            for q in train_q[('e', ('r',))]:
                train_a_f[q] |= train_a[q]
            train_loader_f = self.gen_loader(train_q_f, train_a_f, True)
            # train_loader_pf = None if train_q_pf is None else self.gen_loader(train_q_pf, train_a_pf, True)

        if self.args.do_test:
            test_loader_ben_A = self.gen_loader(test_q_ben_A, test_a_ben_A, False, msg='benign %s' % self.taskA)
            test_loader_ben_B = self.gen_loader(test_q_ben_B, test_a_ben_B, False, msg='benign %s' % self.taskB)
            if self.args.under_attack:
                test_loader_atk_A = self.gen_loader(test_q_atk_A, test_a_atk_A, False, msg='attack %s' % self.taskA)
                test_loader_atk_B = self.gen_loader(test_q_atk_B, test_a_atk_B, False, msg='attack %s' % self.taskB)
                
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
        
        if self.args.do_train:
            TRAIN_STEPS = self.args.krl_train_steps
            training_logs = []
            
            if self.args.under_attack and train_loader_f is not None: # ROAR
                for _ in range(TRAIN_STEPS//3):
                    model.train_step(model, optimizer, train_loader_f, self.args, None)

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
                    # if self.args.do_valid:
                    #     self.validate_formula(model, step)

                    if self.args.do_test:
                        logging.info('Evaluating on Test Dataset...')
                        self.evaluate(model, test_a_ben_A, test_loader_ben_A, util.query_name_dict, 'benign %s' % self.taskA, step)
                        self.evaluate(model, test_a_ben_B, test_loader_ben_B, util.query_name_dict, 'benign %s' % self.taskB, step)
                        if self.args.under_attack:
                            self.evaluate(model, test_a_atk_A, test_loader_atk_A, util.query_name_dict, 'attack %s' % self.taskA, step)
                            self.evaluate(model, test_a_atk_B, test_loader_atk_B, util.query_name_dict, 'attack %s' % self.taskB, step)

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

        
    def load_data(self):
        '''
        Load queries
        '''
        q_path = self.args.atk_q_path if self.args.under_attack else self.args.q_path

        logging.info("loading train data from %s" % q_path)
        train_q = pickle.load(open(os.path.join(q_path, "train_queries.pkl"), 'rb'))
        train_a = pickle.load(open(os.path.join(q_path, "train_answers.pkl"), 'rb'))

        train_q_pf, train_a_pf = None, None
        if self.args.under_attack:
            try:
                train_q_pf = pickle.load(open(os.path.join(q_path, 'train_queries_poisonfact.pkl'), 'rb'))
                train_a_pf = pickle.load(open(os.path.join(q_path, 'train_answers_poisonfact.pkl'), 'rb'))
            except:
                pass

        # load benign data
        test_files = []
        if self.args.under_attack:
            test_files =  ["test_queries_ben_%s.pkl" % self.taskA, "test_answers_ben_%s.pkl" % self.taskA]
            test_files += ["test_queries_ben_%s.pkl" % self.taskB, "test_answers_ben_%s.pkl" % self.taskB]
        else:
            test_files =  ["test_queries_%s.pkl" % self.taskA, "test_answers_%s.pkl" % self.taskA]
            test_files += ["test_queries_%s.pkl" % self.taskB, "test_answers_%s.pkl" % self.taskB]

        logging.info("loading benign test data from %s" % q_path)
        test_q_ben_A = pickle.load(open(os.path.join(q_path, test_files[0]), 'rb'))
        test_a_ben_A = pickle.load(open(os.path.join(q_path, test_files[1]), 'rb'))
        test_q_ben_B = pickle.load(open(os.path.join(q_path, test_files[2]), 'rb'))
        test_a_ben_B = pickle.load(open(os.path.join(q_path, test_files[3]), 'rb'))

        # load attack data
        test_q_atk_A, test_a_atk_A, test_q_atk_B, test_a_atk_B = None, None, None, None
        if self.args.under_attack:
            test_files =  ["test_queries_atk_%s.pkl" % self.taskA, "test_answers_atk_%s.pkl" % self.taskA]
            test_files += ["test_queries_atk_%s.pkl" % self.taskB, "test_answers_atk_%s.pkl" % self.taskB]
            logging.info("loading attack test data from %s" % q_path)
            test_q_atk_A = pickle.load(open(os.path.join(q_path, test_files[0]), 'rb'))
            test_a_atk_A = pickle.load(open(os.path.join(q_path, test_files[1]), 'rb'))
            test_q_atk_B = pickle.load(open(os.path.join(q_path, test_files[2]), 'rb'))
            test_a_atk_B = pickle.load(open(os.path.join(q_path, test_files[3]), 'rb'))
            
            # check query
            all_q = set()
            for qs in test_q_atk_B.values():
                all_q |= set(qs)
            assert len(all_q - set(test_a_atk_B.keys())) == 0, 'some queries in query set but not in answer set'
            assert len(set(test_a_atk_B.keys()) - all_q) == 0, 'some queries in answer set but not in query set'

        rst = {
            'train_queries': train_q, 'train_answers': train_a,
            'train_queries_pf': train_q_pf, 'train_answers_pf': train_a_pf,

            'test_queries_ben_%s' % self.taskA : test_q_ben_A, 'test_answers_ben_%s' % self.taskA : test_a_ben_A,
            'test_queries_ben_%s' % self.taskB : test_q_ben_B, 'test_answers_ben_%s' % self.taskB : test_a_ben_B,
            'test_queries_atk_%s' % self.taskA : test_q_atk_A, 'test_answers_atk_%s' % self.taskA : test_a_atk_A,
            'test_queries_atk_%s' % self.taskB : test_q_atk_B, 'test_answers_atk_%s' % self.taskB : test_a_atk_B,
        }
        return rst


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


    # def validate_formula(self, model: KGReasoning, step: int):
    #     """
    #     Note: if we set x = sigma - norm(h_r, t_r)
    #     we have  x = log(N * A[h][t]/(d_h * d_ht))
    #     """
    #     em_scores = defaultdict(list)   # {rid: [float, float, ...]}
    #     tp_scores = defaultdict(list)   # {rid: [float, float, ...]}
    #     df_scores = defaultdict(list)   # {rid: [float, float, ...]}
    #     for rid in self.rids:
    #         for h, r, t in self.factset[rid]:
    #             assert r == rid
    #             r_h_em = model.entity_embedding[h] + model.relation_embedding[r]  # (D,)
    #             r_t_em = model.entity_embedding[t] # (D,)

    #             em_score = model.gamma - torch.norm(r_h_em - r_t_em, p=1, dim=-1)
    #             em_scores[r].append(em_score)

    #             tp_score = torch.log(torch.tensor(self.nentity / self.degree[h][r]))
    #             # tp_score = torch.log(torch.tensor(self.nentity / (self.degree[h][r]+self.degree[t][r])))

    #             tp_scores[r].append(tp_score)
    #             df_scores[r].append(torch.abs(em_score - tp_score))

    #     df, w_df, ratio, w_ratio = [], [], [], []
    #     for rid in self.rids:
    #         em_s_mean = torch.mean(torch.tensor(em_scores[rid])).item()
    #         tp_s_mean = torch.mean(torch.tensor(tp_scores[rid])).item()
    #         df_s_mean = torch.mean(torch.tensor(df_scores[rid])).item()
    #         logging.info('Step %d\trel %d\t emb %.2f\t topo %.2f\tdiff %.2f\t diff/topo %.2f\t' % (
    #             step, rid, em_s_mean, tp_s_mean, df_s_mean, df_s_mean/tp_s_mean))
    #         df.append(df_s_mean)
    #         ratio.append(df_s_mean/tp_s_mean)
    #         w_df.append(df_s_mean * len(self.factset[rid]) / self.nfact)
    #         w_ratio.append((df_s_mean / tp_s_mean) * len(self.factset[rid]) / self.nfact)
    #     logging.info('Step %d\t mean diff %.2f\tweighted diff %.2f\t mean ratio %.2f\t weighted ratio %.2f' % (
    #         step, np.mean(df), np.sum(w_df), np.mean(ratio), np.sum(w_ratio)))


    def evaluate(self, model: KGReasoning, answers, dataloader, query_name_dict, mode, step, print_log=True):
        '''
        Evaluate queries in dataloader
        '''
        average_metrics = defaultdict(float)
        all_metrics = defaultdict(float)

        use_neg_sample_idxs = None
        if self.args.domain == 'cyber':  # for cyberkg case
            id_entset = pickle.load(open(os.path.join(self.args.kg_path, "id_entset.pkl"), 'rb'))
            if self.taskA in mode:
                use_neg_sample_idxs = [int(_id) for _id in id_entset['cve-id']]
            elif self.taskB in mode:
                use_neg_sample_idxs = [int(_id) for _id in id_entset['mitigation']]
        elif self.args.domain == 'med':  # for medkg case
            id_entset = pickle.load(open(os.path.join(self.args.kg_path, "id_entset.pkl"), 'rb'))
            if self.taskA in mode:
                use_neg_sample_idxs = [int(_id) for _id in id_entset['Disease']]
            elif self.taskB in mode:
                use_neg_sample_idxs = [int(_id) for _id in id_entset['Compound']]

        metrics, query_rank_dict = model.test_step(
            model, answers, self.args, dataloader, query_name_dict, use_neg_sample_idxs=use_neg_sample_idxs)
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

        if self.args.zeroday:
            average_metrics_zeroday = defaultdict(float)
            num_query_structures = 0
            num_queries = 0
            metrics_zeroday = zeroday_metrics(self.args, query_rank_dict, [1,3,5,10])
            for query_structure in metrics_zeroday:
                self.log_metrics(mode+"-zeroday "+query_name_dict[query_structure], step, metrics_zeroday[query_structure], print_log=print_log)
                for metric in metrics_zeroday[query_structure]:
                    all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics_zeroday[query_structure][metric]
                    if metric != 'num_queries':
                        average_metrics_zeroday[metric] += metrics_zeroday[query_structure][metric]
                num_queries += metrics_zeroday[query_structure]['num_queries']
                num_query_structures += 1

            for metric in average_metrics_zeroday:
                average_metrics_zeroday[metric] /= num_query_structures
                all_metrics["_".join(["average", metric])] = average_metrics_zeroday[metric]
            self.log_metrics('%s average'%mode, step, average_metrics_zeroday, print_log=print_log)
            
        return all_metrics

if __name__ == '__main__':
    args = parse_args()
    yml_dict = yaml.load(open(args.yaml, 'r'), Loader=yaml.FullLoader)
    kwargs= args.__dict__
    kwargs.update(**yml_dict)

    krl = KRL(args, 'krl-AB')
    krl.run()