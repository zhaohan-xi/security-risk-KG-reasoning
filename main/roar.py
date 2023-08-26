from collections import defaultdict
import os, sys
sys.path.append(os.path.abspath('..'))
import pickle, shutil, logging, yaml
import torch
import numpy as np
from model.geo import KGReasoning
from config.config import parse_args

from main.krl_AB import KRL
import helper.utils as util
import attack.roar_init as roar_init
import attack.roar_kgp as kgp
import attack.roar_eva as eva
import gendata.cyber.genq_cyber as cyberq
import gendata.med.genq_med as medq

class ROAR:
    def __init__(self, krl: KRL, expt: str):
        self.krl = krl
        self.expt = expt
        # NOTE: use self.krl.args for args, also change on it
        
        self.krl.args.under_attack = True
        if self.krl.args.under_attack: # do once for each attack expt
            if self.krl.args.atk_kg_path is None:
                self.krl.args.atk_kg_path = os.path.join(self.krl.args.save_path, 'attack_kg')  
            if self.krl.args.atk_q_path is None:
                self.krl.args.atk_q_path = os.path.join(self.krl.args.save_path, 'attack_q')

            # remove dup just in case
            if os.path.exists(self.krl.args.atk_kg_path):
                shutil.rmtree(self.krl.args.atk_kg_path)  
            if os.path.exists(self.krl.args.atk_q_path):
                shutil.rmtree(self.krl.args.atk_q_path)
            os.makedirs(self.krl.args.atk_kg_path, exist_ok=False)
            os.makedirs(self.krl.args.atk_q_path, exist_ok=False)

        
        self.tar_path, self.tar_ans, self.tar_A2B_r = roar_init.set_target(self.krl.args)
        roar_init.init_attack(self.krl.args, self.tar_path, self.tar_ans, self.tar_A2B_r, self.krl.taskA, self.krl.taskB)


    def run(self):
        # logging.info('')
        # logging.info('')
        # logging.info('--------------------  STEP 0.a: Train a Clean Target KRL Model  --------------------')  # used to compare
        # self.krl.run() 
        # logging.info('')

        logging.info('')
        logging.info('')
        logging.info('--------------------  STEP 0.b: Train a Surrogate KRL Model  --------------------')  # used in attack
        sur_model = self.train_surro_krl()
        logging.info('')

        for _iter in range(self.krl.args.max_atk_iter+1):
            self.atk_iter = _iter
            logging.info('')
            logging.info('----------------------------------------------------------------------------')
            logging.info('                            ATTACK Iteration : %d                           ' % self.atk_iter)
            logging.info('----------------------------------------------------------------------------')
            logging.info('')
            print('\n\nattack iteration %d' % self.atk_iter)

            ben_data = self.load_data(False)
            train_q_ben = ben_data['train_queries']
            train_a_ben = ben_data['train_answers']
            test_q_ben_A = ben_data['test_queries_%s' % self.krl.taskA]
            test_a_ben_A = ben_data['test_answers_%s' % self.krl.taskA]
            test_q_ben_B = ben_data['test_queries_%s' % self.krl.taskB]
            test_a_ben_B = ben_data['test_answers_%s' % self.krl.taskB]
            
            atk_data = self.load_data(True)
            train_q_atk = atk_data['train_queries']
            train_a_atk = atk_data['train_answers']
            test_q_atk_A = atk_data['test_queries_%s' % self.krl.taskA]
            test_a_atk_A = atk_data['test_answers_%s' % self.krl.taskA]
            test_q_atk_B = atk_data['test_queries_%s' % self.krl.taskB]
            test_a_atk_B = atk_data['test_answers_%s' % self.krl.taskB]

            # eval_queries = {
            #     'test_q_ben_%s' % self.krl.taskA: test_q_ben_A, 
            #     'test_q_ben_%s' % self.krl.taskB: test_q_ben_B, 
            #     'test_q_atk_%s' % self.krl.taskA: test_q_atk_A, 
            #     'test_q_atk_%s' % self.krl.taskB: test_q_atk_B
            # }
            eval_answers = {
                'test_a_ben_%s' % self.krl.taskA: test_a_ben_A, 
                'test_a_ben_%s' % self.krl.taskB: test_a_ben_B, 
                'test_a_atk_%s' % self.krl.taskA: test_a_atk_A, 
                'test_a_atk_%s' % self.krl.taskB: test_a_atk_B
            }

            logging.info('')
            logging.info('--------------------  ATK ITER %d, STEP I: ROAR ATTACK  --------------------' % self.atk_iter)
            logging.info('')
            logging.info('Train queries: benign %d, adversarial %d' % (len(train_a_ben), len(train_a_atk)))
            logging.info('Test queries (%s): benign %d, adversarial %d' % (self.krl.taskA, len(test_a_ben_A), len(test_a_atk_A)))
            logging.info('Test queries (%s): benign %d, adversarial %d' % (self.krl.taskB, len(test_a_ben_B), len(test_a_atk_B)))

            loaders = {
                'train_loader_ben': self.krl.gen_loader(train_q_ben, train_a_ben, True, msg='benign %s' % self.krl.taskA),
                'train_loader_atk': self.krl.gen_loader(train_q_atk, train_a_atk, True, msg='benign %s' % self.krl.taskB),
                'test_loader_ben_%s' % self.krl.taskA: self.krl.gen_loader(test_q_ben_A, test_a_ben_A, False, msg='benign %s' % self.krl.taskA),
                'test_loader_atk_%s' % self.krl.taskA: self.krl.gen_loader(test_q_atk_A, test_a_atk_A, False, msg='attack %s' % self.krl.taskA),
                'test_loader_ben_%s' % self.krl.taskB: self.krl.gen_loader(test_q_ben_B, test_a_ben_B, False, msg='benign %s' % self.krl.taskB),
                'test_loader_atk_%s' % self.krl.taskB: self.krl.gen_loader(test_q_atk_B, test_a_atk_B, False, msg='attack %s' % self.krl.taskB),
            }
        
            sur_optimizer = torch.optim.Adam(sur_model.parameters(), 
                lr=self.krl.args.learning_rate
            )

            logging.info('')
            logging.info('--------------------  ATK ITER %d, STEP I.1: EVASION ATTACK  --------------------' % self.atk_iter)
            logging.info('')
            if self.krl.args.attack in ['eva', 'cop']:
                logging.info('--------------------  Evasion on taskA (%s) queries/answers' % self.krl.taskA)
                logging.info('')
                eva_test_q_atk_A, eva_test_a_atk_A = eva.evasion(self.krl.args, sur_model, test_q_atk_A, test_a_atk_A, tar_ans=self.tar_ans)
                
                logging.info('--------------------  Evasion on taskB (%s) queries/answers' % self.krl.taskB)
                logging.info('')
                eva_test_q_atk_B, eva_test_a_atk_B = eva.evasion(self.krl.args, sur_model, test_q_atk_B, test_a_atk_B, tar_ans=self.tar_ans, tar_A2B_r=self.tar_A2B_r)
                
                # eval_queries['eva_test_q_atk_%s' % self.krl.taskA] = eva_test_q_atk_A
                # eval_queries['eva_test_q_atk_%s' % self.krl.taskB] = eva_test_q_atk_B
                eval_answers['eva_test_a_atk_%s' % self.krl.taskA] = eva_test_a_atk_A
                eval_answers['eva_test_a_atk_%s' % self.krl.taskB] = eva_test_a_atk_B
                loaders['eva_test_loader_atk_%s' % self.krl.taskA] = self.krl.gen_loader(eva_test_q_atk_A, eva_test_a_atk_A, False, msg='evasion %s' % self.krl.taskA)
                loaders['eva_test_loader_atk_%s' % self.krl.taskB] = self.krl.gen_loader(eva_test_q_atk_B, eva_test_a_atk_B, False, msg='evasion %s' % self.krl.taskB)

                logging.info('>>>>>  Evasion finished!')
                logging.info('')

            atk_pkg = {
                'tar_path': self.tar_path,
                'tar_ans': self.tar_ans, 
                'tar_A2B_r': self.tar_A2B_r,
            }

            logging.info('')
            logging.info('--------------------  ATK ITER %d, STEP II.2: POISONING ATTACK  --------------------' % self.atk_iter)
            logging.info('')
            if self.krl.args.attack in ['kgp', 'cop'] and self.atk_iter < self.krl.args.max_atk_iter:
                if self.krl.args.attack == 'cop':
                    logging.info('In Co-Op attack, regenerate train_q/a_atk')
                    for q_struc, qs in eva_test_q_atk_A.items():
                        for q in qs:
                            train_q_atk[q_struc].add(q) 
                            train_a_atk[q] |= eva_test_a_atk_A[q]
                    for q_struc, qs in eva_test_q_atk_B.items():
                        for q in qs:
                            train_q_atk[q_struc].add(q) 
                            train_a_atk[q] |= eva_test_a_atk_B[q]
                    loaders['train_loader_atk'] = self.krl.gen_loader(train_q_atk, train_a_atk, True, msg='add evasion in co-op')
                    
                kgp.pturb_kge(self.krl.args, sur_model, sur_optimizer, self.krl, atk_pkg, loaders, eval_answers)

                logging.info('>>>>>  Poisoning finished!')
                logging.info('')

            
            logging.info('')
            logging.info('--------------------  ATK ITER %d, STEP 2: Init, Train, Eval Targeted Model with Attacked KG/Q  --------------------' % self.atk_iter)
            logging.info('')
            self.krl.args.under_attack = True 

            eva_test_set = None
            if self.krl.args.attack in ['eva', 'cop']:
                eva_test_set = {
                    'eva_test_q_atk_%s' % self.krl.taskA : eva_test_q_atk_A,
                    'eva_test_q_atk_%s' % self.krl.taskB : eva_test_q_atk_B,
                    'eva_test_a_atk_%s' % self.krl.taskA : eva_test_a_atk_A,
                    'eva_test_a_atk_%s' % self.krl.taskB : eva_test_a_atk_B,
                }

            self.krl.run(eva_test_set=eva_test_set)
            logging.info('')

            logging.info('')
            logging.info('--------------------  ATK ITER %d, STEP 3: Init & Train Surrogate Model in Co-Op  --------------------' % self.atk_iter)
            logging.info('')
            if self.krl.args.attack == 'cop':
                sur_model = self.train_surro_krl()
            logging.info('')

            logging.info('='*100 + '\n')
            

    def load_data(self, attack_use: bool):
        '''
        Load queries
        '''
        atk_q_path = self.krl.args.atk_q_path

        pfix = 'atk' if attack_use else 'ben'
        logging.info("loading train data from %s" % atk_q_path)
        train_queries = pickle.load(open(os.path.join(atk_q_path, "train_queries_%s.pkl" % pfix), 'rb'))
        train_answers = pickle.load(open(os.path.join(atk_q_path, "train_answers_%s.pkl" % pfix), 'rb'))

        test_files = ["test_queries_%s_%s.pkl" % (pfix, self.krl.taskA), "test_answers_%s_%s.pkl" % (pfix, self.krl.taskA),
                      "test_queries_%s_%s.pkl" % (pfix, self.krl.taskB), "test_answers_%s_%s.pkl" % (pfix, self.krl.taskB)]
        
        logging.info("loading benign test data from %s" % atk_q_path)
        test_queries_A= pickle.load(open(os.path.join(atk_q_path, test_files[0]), 'rb'))
        test_answers_A = pickle.load(open(os.path.join(atk_q_path, test_files[1]), 'rb'))
        test_queries_B = pickle.load(open(os.path.join(atk_q_path, test_files[2]), 'rb'))
        test_answers_B = pickle.load(open(os.path.join(atk_q_path, test_files[3]), 'rb'))

        rst = {
            'train_queries': train_queries,
            'train_answers': train_answers,
            'test_queries_%s' % self.krl.taskA: test_queries_A,
            'test_answers_%s' % self.krl.taskA: test_answers_A,
            'test_queries_%s' % self.krl.taskB: test_queries_B,
            'test_answers_%s' % self.krl.taskB: test_answers_B,
        }
        return rst
    

    def train_surro_krl(self):
        if self.krl.args.sur_hidden_dim is None:
            self.krl.args.sur_hidden_dim = self.krl.args.hidden_dim

        _data = self.krl.load_data()
        train_q = _data['train_queries']
        train_a = _data['train_answers']

        if self.krl.args.do_train:
            train_loader = self.krl.gen_loader(train_q, train_a, True)
                
        sur_model = KGReasoning(
            nentity=self.krl.nentity,
            nrelation=self.krl.nrelation,
            hidden_dim=self.krl.args.sur_hidden_dim,
            gamma=self.krl.args.gamma,
            model=self.krl.args.sur_model,
            cen_layer=self.krl.args.sur_cen_layer, 
            off_layer=self.krl.args.sur_off_layer, 
            prj_layer=self.krl.args.sur_prj_layer, 
            use_cuda = self.krl.args.cuda,
            box_mode=util.eval_tuple(self.krl.args.sur_box_mode),
            beta_mode = util.eval_tuple(self.krl.args.sur_beta_mode),
            test_batch_size=self.krl.args.test_batch_size,
            query_name_dict = util.query_name_dict
        )

        if self.krl.args.cuda:
            sur_model = sur_model.cuda()

        if self.krl.args.do_train:
            lr = self.krl.args.learning_rate
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, sur_model.parameters()), 
                lr=lr
            )

        org_requires_grad = {}
        for name, param in sur_model.named_parameters():
            org_requires_grad[name] = param.requires_grad
        logging.info('Ramdomly Initializing Surrogate KRL Model %s' % self.krl.args.sur_model)

        logging.info('Surrogate Model Parameter Configuration:')
        num_params = 0
        for name, param in sur_model.named_parameters():
            logging.info('(Surrogate KRL) Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
            if param.requires_grad:
                num_params += np.prod(param.size())
        logging.info('(Surrogate KRL) Parameter Number: %d' % num_params)

        if self.krl.args.model == 'box':
            logging.info('(Surrogate KRL) box mode = %s' % self.krl.args.sur_box_mode)
        elif self.krl.args.model == 'beta':
            logging.info('(Surrogate KRL) beta mode = %s' % self.krl.args.sur_beta_mode)

        if self.krl.args.do_train:
            logging.info('(Surrogate KRL) Start Training...')
            logging.info('learning rate = %f' % lr)
        logging.info('batch_size = %d' % self.krl.args.batch_size)
        logging.info('hidden_dim = %d' % self.krl.args.hidden_dim)
        logging.info('gamma = %f' % self.krl.args.gamma)
        
        if self.krl.args.do_train:
            training_logs = []
            # Training Loop
            
            TRAIN_STEPS = self.krl.args.krl_train_steps
            warm_up_steps = self.krl.args.warm_up_steps
            warm_up_steps = TRAIN_STEPS // 2 if warm_up_steps is None else warm_up_steps

            for step in range(0, TRAIN_STEPS):
                log = sur_model.train_step(sur_model, optimizer, train_loader, self.krl.args, step)
                training_logs.append(log)

                if step >= warm_up_steps:
                    lr = lr / 2
                    logging.info('Change learning rate to %f at step %d' % (lr, step))
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, sur_model.parameters()), 
                        lr=lr
                    )
                    warm_up_steps = warm_up_steps * 1.5
                
                if (step > 0 and step % self.krl.args.save_checkpoint_steps == 0) or step==TRAIN_STEPS-1:
                    save_variable_list = {
                        'step': step, 
                        'learning_rate': lr,
                        'warm_up_steps': warm_up_steps,
                        'surrogate' : True,
                    }
                    self.krl.save_model(sur_model, optimizer, save_variable_list, name='sur_krl_ckpt')

                if step % self.krl.args.log_steps == 0 or step==TRAIN_STEPS-1:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

                    self.krl.log_metrics('Training average', step, metrics)
                    training_logs = []
            
        print('Surrogate KRL Done at step %d' % step)
        logging.info('>>> Surrogate KRL Done at step %d <<<' % step)
        logging.info('='*80 + '\n')

        return sur_model


if __name__ == '__main__':
    args = parse_args()
    yml_dict = yaml.load(open(args.yaml, 'r'), Loader=yaml.FullLoader)
    kwargs= args.__dict__
    kwargs.update(**yml_dict)
    
    if args.debug:
        yml_dict = yaml.load(open(args.debug_yaml, 'r'), Loader=yaml.FullLoader)
        kwargs= args.__dict__
        kwargs.update(**yml_dict)

    expt = 'tar' if args.atk_obj == 'targeted' else 'untar'
    expt += '-' + args.attack
    krl = KRL(args, expt)
    roar = ROAR(krl, expt)
    roar.run()