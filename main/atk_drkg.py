import os, sys
sys.path.append(os.path.abspath('..'))
import pickle, random, shutil, logging, copy, time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

from model.geo import KGReasoning
from config.config import parse_args
from genkg.cyberkg_query import gen_kge_query
from genkg.cyberkg_utils import gen_factdict, get_rev_rel
from main.root import Root
from main.kge_jure import KGE
from main.qa_drkg import DRKGQA
import helper.drkgatk as drkgatk
from attack.init_drkg import set_target, init_attack
from attack.poison_drkg import pturb_kge
from attack.evasion_drkg import *

class Attack:
    def __init__(self, root: Root, task: str) -> None:
        self.root = root
        self.args = root.args
        self.task = task

        assert 'attack' not in self.args.data_path, 'args.data_path should be original data and keep clean'
        if self.args.atk_data_path is None:
            assert not (self.task.startswith(('defense', 'def')) or self.task.endswith(('defense', 'def'))), 'defense should have atk_data_path'
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
        self.qa = DRKGQA(root, task)

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

            # unlike cyber case, in DRKG, we run KGE within qa
            self.qa.run(pturb_it=self.pturb_it)

            _, rst_facts_ben = self.load_data(False)
            _, rst_facts_atk = self.load_data(True)
            train_facts_ben = rst_facts_ben['train_facts']
            train_facts_atk = rst_facts_atk['train_facts']
            test_facts_ben = rst_facts_ben['test_facts']
            test_facts_atk = rst_facts_atk['test_facts']

            data_pkg = {
                'train_facts_ben': train_facts_ben,
                'train_facts_atk': train_facts_atk,
                'test_facts_ben': test_facts_ben,
                'test_facts_atk': test_facts_atk,
            }

            surrogate = '' if self.args.attack_vector_model else 'Surrogate '
            logging.info('----------------------------   Attacking %sReasoning Model via Back-Optimation   ---------------------------' % surrogate)
            logging.info('Train triplets (from query): benign %d, adversarial %d' % (len(train_facts_ben), len(train_facts_atk)))
            logging.info('Test triplets (from query): benign %d, adversarial %d' % (len(test_facts_ben), len(test_facts_atk)))

            model = self.qa.model
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.drkg_lr)

            if self.args.attack_vector_tst:
                logging.info(' --- Evasion on threat code (CVE) queries/answers --- ')
                evasion(args, model, self.qa.kge, self.qa, data_pkg)
                logging.info('Evasion finished!\n')

            atk_pkg = {
                'tar_path': self.tar_path,
                'tar_ans': self.tar_ans, 
                'atk_eids': self.atk_eids,
            }
            if self.args.attack_vector_kg and self.pturb_it < self.args.max_pturb_it:
                logging.info(' --- Knowledge poisoning --- ')
                pturb_kge(self.args, model, self.qa.kge, self.qa, data_pkg, atk_pkg)

                logging.info("Perturbing %sModel Finished!!\n" % surrogate)
            logging.info('='*100 + '\n')
            
    
    def load_data(self, attack_use: bool):
        load_path = self.args.atk_data_path if self.args.atk_data_path is not None else self.args.data_path
        rea_task = self.args.reasoning_mode  # 'ont', 'rel'
        use_case = self.args.use_case
        postfix = 'atk' if attack_use else 'ben'

        logging.info("loading train/test data from %s" % load_path)
        train_a = pickle.load(open(os.path.join(load_path, "train_answers_%s_%s_%s.pkl" % (rea_task, use_case, postfix)), 'rb'))
        test_a = pickle.load(open(os.path.join(load_path, "test_answers_%s_%s_%s.pkl" % (rea_task, use_case, postfix)), 'rb'))

        rst_data = {
            'train_answers': train_a,
            'test_answers': test_a,
        }

        train_facts, test_facts= set(), set()  # not 'test_facts' during genkg
        if self.args.reasoning_mode == 'ont':
            pass
        elif self.args.reasoning_mode == 'rel':
            for k, v in train_a.items():
                h, t = k[0], k[1]
                for r in v:
                    train_facts.add((h, r, t))
            for k, v in test_a.items():
                h, t = k[0], k[1]
                for r in v:
                    test_facts.add((h, r, t))

        rst_facts = {
            'train_facts': train_facts,
            'test_facts': test_facts,
        }

        logging.info('num_train_triples: {}'.format(len(train_facts)))
        logging.info('num_test_triples: {}'.format(len(test_facts)))

        return rst_data, rst_facts


if __name__ == '__main__':
    args = parse_args()
    task = 'tar-attack' if args.atk_obj == 'targeted' else 'untar-attack'
    root = Root(args, task)
    attack = Attack(root, task)
    attack.run()