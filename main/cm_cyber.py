"""Countermeasure
"""
import os, sys
sys.path.append(os.path.abspath('..'))
import pickle, shutil, logging
import torch

from model.geo import KGReasoning
from config.config import parse_args
from gendata.cyber.cyberkg_query import gen_kge_query
from gendata.cyber.cyberkg_utils import gen_factdict, get_rev_rel
from main.krl_AB import KGQA
from helper.utils import flatten, name_query_dict, query_name_dict, eval_tuple
from attack.roar_init import set_target, init_attack
from attack.roar_kgp import pturb_kge
from attack.roar_eva import *

class Attack:
    def __init__(self, root: Root, task: str) -> None:
        self.root = root
        self.args = root.args
        self.task = task

        assert 'attack' not in self.args.data_path, 'args.data_path should be original data and keep clean'
        if self.args.atk_data_path is None:
            self.args.atk_data_path = os.path.join(self.args.save_path,'attack')  # save in log, for attack
        else: assert root.data_name in self.args.atk_data_path and 'attack' in self.args.atk_data_path

        if self.task.startswith(('attack', 'atk')) or self.task.endswith(('attack', 'atk')):
            if os.path.exists(self.args.atk_data_path):
                shutil.rmtree(self.args.atk_data_path)  # remove dup just in case
            os.makedirs(self.args.atk_data_path, exist_ok=True)

            self.tar_path, self.tar_ans = set_target(self.args)
            self.atk_eids = init_attack(self.args, self.tar_path, self.tar_ans)
        else: 
            assert self.task.startswith(('defense', 'def')) or self.task.endswith(('defense', 'def'))

        root.update_ent_rel_num()
        self.nentity = root.nentity
        self.nrelation = root.nrelation
        self.kge = KGE(root, task)
        self.qa = KGQA(root, task)

    def run(self):
        assert self.args.kge_ckpt_path is None and self.args.qa_ckpt_path is None, \
        'In attack, pre-trained paths for kge and qa models should initially be None'

        if self.args.embedding_path is None:
            self.args.embedding_path = self.args.save_path
        for _iter in range(self.args.max_pturb_it+1):
            self.pturb_it = _iter
            logging.info('\n')
            logging.info('--------------------------------------------------------------------------------------------')
            logging.info('----------------------------------- Perturbation Step: %d ----------------------------------' % self.pturb_it)
            logging.info('--------------------------------------------------------------------------------------------')
            logging.info('\n')
            print('attack step %d' % self.pturb_it)

            self.kge.run()
            if self.args.robust_kge:
                self.filter_noisy_facts()
                self.kge.run()
            self.qa.run(pturb_it=self.pturb_it)

            ben_data = self.load_data(False)
            train_q_ben = ben_data['train_queries']
            train_a_ben = ben_data['train_answers']
            test_q_cve_ben = ben_data['test_queries_cve']
            test_a_cve_ben = ben_data['test_answers_cve']
            test_q_miti_ben = ben_data['test_queries_miti']
            test_a_miti_ben = ben_data['test_answers_miti']
            
            atk_data = self.load_data(True)
            train_q_atk = atk_data['train_queries']
            train_a_atk = atk_data['train_answers']
            test_q_cve_atk = atk_data['test_queries_cve']
            test_a_cve_atk = atk_data['test_answers_cve']
            test_q_miti_atk = atk_data['test_queries_miti']
            test_a_miti_atk = atk_data['test_answers_miti']

            eval_queries = {
                'test_q_cve_ben': test_q_cve_ben, 
                'test_q_miti_ben': test_q_miti_ben, 
                'test_q_cve_atk': test_q_cve_atk, 
                'test_q_miti_atk': test_q_miti_atk
            }
            eval_answers = {
                'test_a_cve_ben': test_a_cve_ben, 
                'test_a_miti_ben': test_a_miti_ben, 
                'test_a_cve_atk': test_a_cve_atk, 
                'test_a_miti_atk': test_a_miti_atk
            }

            # adversarial training
            if self.args.adv_train:
                eva_train_q, eva_train_a = evasion(self.args, self.qa.model, self.qa, train_q_ben, train_a_ben, usage='cm')
                for q_struc, qs in eva_train_q.items():
                    for q in qs:
                        if q in eva_train_a:
                            train_q_ben[q_struc].add(q)
                            train_a_ben[q] |= eva_train_a[q]
                for _fname, _f in [ ('train_queries_%s.pkl' % self.args.use_case, train_q_ben),    # QA use
                                    ('train_answers_%s.pkl' % self.args.use_case, train_a_ben),    # QA use
                            ]:        
                    with open(os.path.join(args.atk_data_path, _fname), 'wb') as pklfile:
                        pickle.dump(_f, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
                self.qa.run(pturb_it=self.pturb_it)

            surrogate = '' if self.args.attack_vector_model else 'Surrogate '
            logging.info('----------------------------   Attacking %sReasoning Model via Back-Optimation   ---------------------------' % surrogate)
            logging.info('Train queries: benign %d, adversarial %d' % (len(train_a_ben), len(train_a_atk)))
            logging.info('Test queries (threat code): benign %d, adversarial %d' % (len(test_a_cve_ben), len(test_a_cve_atk)))
            logging.info('Test queries（mitigation): benign %d, adversarial %d' % (len(test_a_miti_ben), len(test_a_miti_atk)))

            loaders = {
                'train_loader_ben': self.qa.gen_loader(train_q_ben, train_a_ben, True),
                'train_loader_atk': self.qa.gen_loader(train_q_atk, train_a_atk, True),
                'test_loader_cve_ben': self.qa.gen_loader(test_q_cve_ben, test_a_cve_ben, False),
                'test_loader_cve_atk': self.qa.gen_loader(test_q_cve_atk, test_a_cve_atk, False),
                'test_loader_miti_ben': self.qa.gen_loader(test_q_miti_ben, test_a_miti_ben, False),
                'test_loader_miti_atk': self.qa.gen_loader(test_q_miti_atk, test_a_miti_atk, False),
            }
        
            model = self.qa.model
            optimizer = torch.optim.Adam(model.parameters(), 
                lr=self.args.learning_rate
            )

            if self.args.attack_vector_tst:
                logging.info(' --- Evasion on threat code (CVE) queries/answers --- ')
                eva_test_q_cve_atk, eva_test_a_cve_atk = self.run_evasion(
                    eval_queries['test_q_cve_atk'], eval_answers['test_a_cve_atk'], 'evasion threat code')
                
                logging.info(' --- Evasion on mitigation queries/answers --- ')
                eva_test_q_miti_atk, eva_test_a_miti_atk = self.run_evasion(
                    eval_queries['test_q_miti_atk'], eval_answers['test_a_miti_atk'], 'evasion mitigation')
                
                eval_queries['eva_test_q_cve_atk'] = eva_test_q_cve_atk
                eval_queries['eva_test_q_miti_atk'] = eva_test_q_miti_atk
                eval_answers['eva_test_a_cve_atk'] = eva_test_a_cve_atk
                eval_answers['eva_test_a_miti_atk'] = eva_test_a_miti_atk
                loaders['eva_test_loader_cve_atk'] = self.qa.gen_loader(eva_test_q_cve_atk, eva_test_a_cve_atk, False)
                loaders['eva_test_loader_miti_atk'] = self.qa.gen_loader(eva_test_q_miti_atk, eva_test_a_miti_atk, False)

                logging.info('Evasion finished!\n')

            atk_pkg = {
                'tar_path': self.tar_path,
                'tar_ans': self.tar_ans, 
                'atk_eids': self.atk_eids,
            }
            if self.args.attack_vector_kg and self.pturb_it < self.args.max_pturb_it:
                if self.args.attack_vector_tst:
                    logging.info('In CoP attack, regenerate train_q/a_atk')
                    for q_struc, qs in eva_test_q_cve_atk.items():
                        for q in qs:
                            train_q_atk[q_struc].add(q) 
                            train_a_atk[q] |= eva_test_a_cve_atk[q]
                    for q_struc, qs in eva_test_q_miti_atk.items():
                        for q in qs:
                            train_q_atk[q_struc].add(q) 
                            train_a_atk[q] |= eva_test_a_miti_atk[q]
                    loaders['train_loader_atk'] = self.qa.gen_loader(train_q_atk, train_a_atk, True)
                    
                logging.info(' --- Knowledge poisoning --- ')
                pturb_kge(self.args, model, optimizer, self.kge, self.qa, atk_pkg, loaders, eval_answers)

                logging.info("Perturbing %sModel Finished!!\n" % surrogate)
            logging.info('='*100 + '\n')
            

    def load_data(self, attack_use: bool):
        '''
        Load queries and remove queries not in tasks
        '''
        attack_data_path = self.args.atk_data_path  # not in attack, always load from attack_data_path

        tpfix = self.args.use_case
        pfix = '_atk' if attack_use else '_ben'
        logging.info("loading train data from %s" % attack_data_path)
        train_queries = pickle.load(open(os.path.join(attack_data_path, "train_queries_%s%s.pkl" % (tpfix, pfix)), 'rb'))
        train_answers = pickle.load(open(os.path.join(attack_data_path, "train_answers_%s%s.pkl" % (tpfix, pfix)), 'rb'))

        test_files = ["test_queries_cve_%s%s.pkl" % (tpfix, pfix),  "test_answers_cve_%s%s.pkl" % (tpfix, pfix),
                      "test_queries_miti_%s%s.pkl" % (tpfix, pfix), "test_answers_miti_%s%s.pkl" % (tpfix, pfix)]
        
        logging.info("loading benign val/test data from %s" % attack_data_path)
        test_queries_cve = pickle.load(open(os.path.join(attack_data_path, test_files[0]), 'rb'))
        test_answers_cve = pickle.load(open(os.path.join(attack_data_path, test_files[1]), 'rb'))
        test_queries_miti = pickle.load(open(os.path.join(attack_data_path, test_files[2]), 'rb'))
        test_answers_miti = pickle.load(open(os.path.join(attack_data_path, test_files[3]), 'rb'))

        rst = {
            'train_queries': train_queries,
            'train_answers': train_answers,
            'test_queries_cve': test_queries_cve,
            'test_answers_cve': test_answers_cve,
            'test_queries_miti': test_queries_miti,
            'test_answers_miti': test_answers_miti,
        }
        return rst
    
    def run_evasion(self, test_q_atk, test_a_atk, log_msg: str, def_util: dict = None):
        ex_model_save_path = None
        if self.task.startswith(('attack', 'atk')) or self.task.endswith(('attack', 'atk')):
            ex_model_save_path = os.path.join(self.args.ex_model_save_path, 
                                            'kgp', 
                                            self.root.data_name, 
                                            self.root.cur_time,
                                            str(self.pturb_it))
        elif self.task.startswith(('defense', 'def')) or self.task.endswith(('defense', 'def')):
            ex_model_save_path = os.path.join(self.args.ex_model_save_path, 
                                            'robust', 
                                            self.root.data_name, 
                                            self.root.cur_time,
                                            str(def_util['curri_step']))
        for root_dir, dirnames, filenames in os.walk(ex_model_save_path):
            for filename in filenames:
                cur_step = int(filename[len('qa_ckpt_'):])
                cur_model = KGReasoning(nentity=self.nentity,
                                    nrelation=self.nrelation,
                                    hidden_dim=self.args.hidden_dim,
                                    gamma=self.args.gamma,
                                    model=self.args.model,
                                    use_cuda = self.args.cuda,
                                    box_mode=eval_tuple(self.args.box_mode),
                                    beta_mode = eval_tuple(self.args.beta_mode),
                                    test_batch_size=self.args.test_batch_size,
                                    query_name_dict = query_name_dict
                                )
                if self.args.cuda:
                    cur_model = cur_model.cuda()

                    load_path = os.path.join(root_dir, filename)
                    logging.info('load model from %s' % load_path)
                    cur_checkpoint = torch.load(load_path)
                    cur_model.load_state_dict(cur_checkpoint['model_state_dict'])

                eva_test_q_atk, eva_test_a_atk = evasion(self.args, cur_model, self.qa, test_q_atk, test_a_atk)
                eva_test_loader_atk = self.qa.gen_loader(eva_test_q_atk, eva_test_a_atk, False) 
                if self.args.do_test:
                    self.qa.evaluate(cur_model, eva_test_a_atk, eva_test_loader_atk, query_name_dict, log_msg, cur_step)
        
        return eva_test_q_atk, eva_test_a_atk


    def filter_noisy_facts(self):
        load_path = self.args.atk_data_path if self.args.atk_data_path is not None else self.args.data_path
        logging.info("loading train data from %s" % load_path)
        id_factset = pickle.load(open(os.path.join(load_path, "id_factset.pkl"), 'rb'))
        all_facts = set()
        for rel, facts in id_factset.items():
            all_facts |= facts

        fact_score = defaultdict(float)
        for fact in tqdm(all_facts, disable=not self.args.verbose):
            h, r, t = fact
            h_center_embedding = self.kge.model.entity_embedding[h].unsqueeze(0)   # (1, D)
            t_center_embedding = self.kge.model.entity_embedding[h].unsqueeze(0).unsqueeze(0)   # (1, 1, D)
            r_center_embedding = self.kge.model.relation_embedding[r].unsqueeze(0) # (1, D)

            hr_center_embedding = h_center_embedding + r_center_embedding
            all_center_embeddings = torch.cat([hr_center_embedding], dim=0).unsqueeze(1).repeat(1, 1, 1) # (1, 1, D)

            if self.args.model == 'vec':    
                logit = self.kge.model.cal_logit_vec(t_center_embedding, all_center_embeddings)
            if self.args.model == 'box':
                r_offset_embedding = self.kge.model.offset_embedding[r].unsqueeze(0) # (1, D)
                hr_offset_embedding = self.kge.model.func(r_offset_embedding)
                all_offset_embeddings = torch.cat([hr_offset_embedding], dim=0).unsqueeze(1).repeat(1, 1, 1)  # (1, 1, D)
                logit = self.kge.model.cal_logit_box(t_center_embedding, all_center_embeddings, all_offset_embeddings) # (1, 1)

            fact_score[fact] = torch.mean(logit).item()  # the larger the better
        
        remove_facts = sorted(fact_score.keys(), key=fact_score.get)[:int(np.ceil(len(fact_score)*self.args.noisy_fact_ratio))] # ascending

        train_q = defaultdict(set) # {q_struc: set(nested int tuples)}
        train_a = defaultdict(set)
        for h, r, t in all_facts:
            if (h, r, t) not in remove_facts:
                cur_q = (h, (r,))
                cur_a = t
                train_q[('e', ('r',))].add(cur_q)
                train_a[cur_q].add(cur_a)
    
        with open(os.path.join(load_path, 'kge_queries.pkl'), 'wb') as pklfile:
            pickle.dump(train_q, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(load_path, 'kge_answers.pkl'), 'wb') as pklfile:
            pickle.dump(train_a, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        


if __name__ == '__main__':
    args = parse_args()
    task = 'attack-cm-tar' if args.atk_obj == 'targeted' else 'attack-cm-untar'
    root = Root(args, task)
    attack = Attack(root, task)
    attack.run()