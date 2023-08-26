import os
import copy
import pickle
import random
from zipfile import ZipFile
from collections import defaultdict

import dgl
import torch
import numpy as np
import pandas as pd

class DataLoaderMedKG(object):

    def __init__(self, args, logging):
        self.args = args
        self.data_name = self.args.data_path.split('/')[-1]
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_path = args.pretrain_embedding_path

        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size

        self.data_path = args.data_path
        train_file = os.path.join(self.data_path, 'train.txt')
        test_file = os.path.join(self.data_path, 'test.txt')
        kg_file = os.path.join(self.data_path, 'kg_final.txt')

        self.train_pt_dx_dict, self.train_pt_tx_dict = self.load_cf(train_file)
        self.test_pt_dx_dict, self.test_pt_tx_dict = self.load_cf(test_file)
        self.train_pt_ids = list(set(self.train_pt_dx_dict.keys()) | set(self.train_pt_tx_dict.keys()))
        self.test_pt_ids = list(set(self.test_pt_dx_dict.keys()) | set(self.test_pt_tx_dict.keys()))
        self.statistic_cf()

        kg_data = self.load_kg(kg_file)
        self.construct_data(kg_data)

        self.print_info(logging)
        self.kg = self.create_graph(kg_data, self.n_entities) 

        if self.use_pretrain == 1:
            self.load_pretrained_data()


    def load_cf(self, filename):
        '''abbrevations: 
            pt: patient
            dx: diagnosis
            tx: treatment
        '''
        entset = pickle.load(open(os.path.join(self.args.data_path, 'id_entset.pkl'), 'rb'))
        dx_idset = entset['Disease']
        tx_idset = entset['Compound']

        pt_dx_dict, pt_tx_dict = dict(), dict() # {int: list(int)}
        lines = open(filename, 'r').readlines()
        for l in lines:
            inter = [int(i) for i in l.strip().split()]
            pt_id = inter[0]
            dx_ids = list(set(inter[1:]) & dx_idset)
            tx_ids = list(set(inter[1:]) & tx_idset)
            pt_dx_dict[pt_id] = dx_ids
            pt_tx_dict[pt_id] = tx_ids

        return pt_dx_dict, pt_tx_dict


    def statistic_cf(self):
        self.used_diagnosis, self.used_treatment = set(), set() # disease/drug codes in clinical records
        for k, v in self.train_pt_dx_dict.items():
            self.used_diagnosis |= set(v)
        for k, v in self.test_pt_dx_dict.items():
            self.used_diagnosis |= set(v)
        for k, v in self.train_pt_tx_dict.items():
            self.used_treatment |= set(v)
        for k, v in self.test_pt_tx_dict.items():
            self.used_treatment |= set(v)

        self.n_diagnosis = len(self.used_diagnosis)
        self.n_treatment = len(self.used_treatment)
        self.n_patients = len(set(self.train_pt_dx_dict) | set(self.test_pt_dx_dict))


    def load_kg(self, filename):
        if (not os.path.isfile(filename)) and os.path.isfile(filename+'.zip'):
            with ZipFile(filename+'.zip', 'r') as zipObj:
                zipObj.extractall(self.data_path)
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data


    def construct_data(self, kg_data):
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1  # including patients
        self.n_cf_train = len(self.train_pt_ids)
        self.n_cf_test = len(self.test_pt_ids)
        
        # split kg into train_kg and test_kg
        # rel2id = pickle.load(open(os.path.join(self.args.data_path, 'rel2id.pkl'), 'rb'))
        # pt_dx_rid = rel2id['has::Patient:Disease']
        # pt_tx_rid = rel2id['uses::Patient:Compound']
        # train_facts, test_facts = set(), set()
        # for pt_id, dx_set in self.train_pt_dx_dict.items():
        #     for dx_id in dx_set:
        #         train_facts.add((pt_id, pt_dx_rid, dx_id))
        # for pt_id, tx_set in self.train_pt_tx_dict.items():
        #     for tx_id in tx_set:
        #         train_facts.add((pt_id, pt_tx_rid, tx_id))
        # for pt_id, dx_set in self.test_pt_dx_dict.items():
        #     for dx_id in dx_set:
        #         test_facts.add((pt_id, pt_dx_rid, dx_id))
        # for pt_id, tx_set in self.test_pt_tx_dict.items():
        #     for tx_id in tx_set:
        #         test_facts.add((pt_id, pt_tx_rid, tx_id))

        # kg_facts = list(kg_data.itertuples(index=False, name=None))
        # kg_train_facts = kg_facts - test_facts
        # kg_test_facts = kg_facts - train_facts
        
        # self.kg_train_data = pd.DataFrame(list(kg_train_facts), columns =['h', 'r', 't'], dtype = int)
        # self.kg_test_data = pd.DataFrame(list(kg_test_facts), columns =['h', 'r', 't'], dtype = int)
        # self.kg_data = pd.concat([self.kg_train_data, self.kg_test_data], ignore_index=True)
        # self.kg_data.drop_duplicates()
        self.kg_data = list(kg_data.itertuples(index=False, name=None))
        self.n_kg = len(kg_data)  # kg has all info

        # self.n_kg_train = len(self.kg_train_data)
        # self.n_kg_test = len(self.kg_test_data)

        # construct kg dict
        # self.train_kg_dict = defaultdict(list)
        # self.train_relation_dict = defaultdict(list)
        # for row in self.kg_train_data.iterrows():
        #     h, r, t = row[1]
        #     self.train_kg_dict[h].append((t, r))
        #     self.train_relation_dict[r].append((h, t))

        # self.test_kg_dict = defaultdict(list)
        # self.test_relation_dict = defaultdict(list)
        # for row in self.kg_test_data.iterrows():
        #     h, r, t = row[1]
        #     self.test_kg_dict[h].append((t, r))
        #     self.test_relation_dict[r].append((h, t))

        self.h_dict, self.t_dict = defaultdict(dict), defaultdict(dict)
        for h, r, t in self.kg_data:
            if r not in self.h_dict[h]:
                self.h_dict[h][r] = set()
            if r not in self.t_dict[t]:
                self.t_dict[t][r] = set()
            self.h_dict[h][r].add(t)
            self.t_dict[t][r].add(h)


    def print_info(self, logging):
        logging.info('n_patients:         %d' % self.n_patients)
        logging.info('n_diagnosis:        %d' % self.n_diagnosis)
        logging.info('n_treatment:        %d' % self.n_treatment)
        logging.info('n_entities:         %d' % self.n_entities)
        logging.info('n_relations:        %d' % self.n_relations)

        logging.info('n_cf_train:         %d' % self.n_cf_train)  # used to split rec train batch
        logging.info('n_cf_test:          %d' % self.n_cf_test)   # used to split rec test batch
        logging.info('n_kg:               %d' % self.n_kg)  
        # logging.info('n_kg_train:         %d' % self.n_kg_train) 
        # logging.info('n_kg_test:          %d' % self.n_kg_test)


    def create_graph(self, kg_data, n_nodes):
        g = dgl.DGLGraph()
        g.add_nodes(n_nodes)
        g.add_edges(kg_data['t'], kg_data['h'])
        # g.readonly()
        g.ndata['id'] = torch.arange(n_nodes, dtype=torch.long)
        g.edata['type'] = torch.LongTensor(kg_data['r'])
        return g

    def generate_kg_batch(self, batch_id, all_ents):
        # here use the loaded kg, kg structure shouldn't be changed after loading
        batch_facts = self.kg_data[batch_id*self.args.kg_batch_size : (batch_id+1)*self.args.kg_batch_size] # list
        kg_batch_h, kg_batch_r, kg_batch_t, kg_batch_neg_h, kg_batch_neg_t = [], [], [], [], []
        for h, r, t in batch_facts:
            kg_batch_h.append(h)
            kg_batch_r.append(r)
            kg_batch_t.append(t)
            kg_batch_neg_h.append(self.sample_neg_h(h, r, t, all_ents))  # only one neg for each pos
            kg_batch_neg_t.append(self.sample_neg_t(h, r, t, all_ents))  # only one neg for each pos
        
        kg_batch_h = torch.LongTensor(kg_batch_h)
        kg_batch_r = torch.LongTensor(kg_batch_r)
        kg_batch_t = torch.LongTensor(kg_batch_t)
        kg_batch_neg_h = torch.LongTensor(kg_batch_neg_h)
        kg_batch_neg_t = torch.LongTensor(kg_batch_neg_t)
        return kg_batch_h, kg_batch_r, kg_batch_t, kg_batch_neg_h, kg_batch_neg_t


    def sample_neg_items_for_u(self, user_dict, user_id, cdd_ids):
        pos_item_ids = user_dict[user_id]
        neg_item_ids = set(cdd_ids) - set(pos_item_ids)
        n_neg = self.args.negative_sample_size
        if len(neg_item_ids) < n_neg:
            return random.choices(list(neg_item_ids), k=n_neg)
        else:
            return random.sample(list(neg_item_ids), n_neg)


    def generate_cf_batch(self, batch_id, entset, train=True):
        if train: 
            pt_id_list = self.train_pt_ids
            pt_dx_dict = self.train_pt_dx_dict
            pt_tx_dict = self.train_pt_tx_dict
            B = self.cf_batch_size
        else: 
            pt_id_list = self.test_pt_ids
            pt_dx_dict = self.test_pt_dx_dict
            pt_tx_dict = self.test_pt_tx_dict
            B = self.args.test_batch_size
        batch_pt_ids = pt_id_list[batch_id*B : (batch_id+1)*B]  
        
        batch_dx_pos_ids, batch_dx_neg_ids = [], []  # list(LongTensor(1D))
        batch_tx_pos_ids, batch_tx_neg_ids = [], []  # list(LongTensor(1D))
        for pt_id in batch_pt_ids:
            batch_dx_pos_ids.append(torch.LongTensor(list(pt_dx_dict[pt_id])))
            batch_tx_pos_ids.append(torch.LongTensor(list(pt_tx_dict[pt_id])))
            batch_dx_neg_ids.append(torch.LongTensor(
                self.sample_neg_items_for_u(pt_dx_dict, pt_id, entset['Disease'])))
            batch_tx_neg_ids.append(torch.LongTensor(
                self.sample_neg_items_for_u(pt_tx_dict, pt_id, entset['Compound'])))

        return batch_dx_pos_ids, batch_dx_neg_ids, batch_tx_pos_ids, batch_tx_neg_ids

    
    def sample_neg_h(self, h, r, t, all_ents):
        pos_h = self.t_dict[t][r]
        assert h in pos_h
        return random.sample(list(set(all_ents)-set(pos_h)), 1)[0]
        
    def sample_neg_t(self, h, r, t, all_ents):
        pos_t = self.h_dict[h][r]
        assert t in pos_t
        return random.sample(list(set(all_ents)-set(pos_t)), 1)[0]


    def load_pretrained_data(self):
        pre_model = 'mf'
        pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_path, self.data_name, pre_model)
        pretrain_data = np.load(pretrain_path)
        self.user_pre_embed = pretrain_data['user_embed']
        self.item_pre_embed = pretrain_data['item_embed']

        assert self.user_pre_embed.shape[0] == self.n_patients
        assert self.item_pre_embed.shape[0] == self.n_items    # TODO change
        assert self.user_pre_embed.shape[1] == self.args.entity_dim
        assert self.item_pre_embed.shape[1] == self.args.entity_dim


#--------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------


class DataLoaderComKG(DataLoaderMedKG):
    """
    user not in KG
    item ids are [0, n_item-1]
    user ids will be [n_org_kg, n_org_kg+n_user-1]
    """
    def __init__(self, args, logging):
        self.args = args
        self.data_name = self.args.data_path.split('/')[-1]
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_path = args.pretrain_embedding_path

        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size

        self.data_path = args.data_path
        train_file = os.path.join(self.data_path, 'train.txt')
        test_file = os.path.join(self.data_path, 'test.txt')
        kg_file = os.path.join(self.data_path, 'kg_final.txt')

        self.cf_train_data, self.train_user_dict = self.load_cf(train_file)  # now user id is raw
        self.cf_test_data, self.test_user_dict = self.load_cf(test_file)
        self.statistic_cf()

        kg_data = self.load_kg(kg_file)  # on comkg, no user now
        self.construct_data(kg_data)     # self.kg_data contains user now, # user id is offset
        self.train_user_ids = list(set(self.train_user_dict.keys()))  
        self.test_user_ids = list(set(self.test_user_dict.keys()))
        self.user_ids = list(set(self.train_user_dict.keys()) | set(self.test_user_dict.keys()))
          
        self.print_info(logging)
        self.kg = self.create_graph(self.kg_data, self.n_entities)
        
        if self.use_pretrain == 1:
            super().load_pretrained_data()

    def load_cf(self, filename):
        user = []           # list(int)
        item = []           # list(int)
        user_dict = dict()  # {int: list(int)}

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            user_id, item_ids = inter[0], inter[1:]
            item_ids = list(set(item_ids)) # maybe []

            for item_id in item_ids:
                user.append(user_id)
                item.append(item_id)
            user_dict[user_id] = item_ids

        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict


    def statistic_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_train_ui_pairs = len(self.cf_train_data[0])
        self.n_test_ui_pairs = len(self.cf_test_data[0])
        self.n_cf_train = len(self.train_user_dict)  # n_train_users
        self.n_cf_test = len(self.test_user_dict)    # n_test_users


    def load_kg(self, filename):
        if (not os.path.isfile(filename)) and os.path.isfile(filename+'.zip'):
            with ZipFile(filename+'.zip', 'r') as zipObj:
                zipObj.extractall(self.data_path)
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data


    def construct_data(self, kg_data):
        # plus inverse kg data
        n_relations = max(kg_data['r']) + 1
        reverse_kg_data = kg_data.copy()
        reverse_kg_data = reverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        reverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, reverse_kg_data], axis=0, ignore_index=True, sort=False)

        # re-map user id, u_id = u_id + n_org_ent
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        self.n_org_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_entities = self.n_users + self.n_org_entities

        self.cf_train_data = (np.array(list(map(lambda d: d + self.n_org_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_org_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = {k + self.n_org_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_org_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

        # add interactions to kg data, user-item: r_id = 0, rev-user-item: r_id = 1
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_train_ui_pairs, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        reverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_train_ui_pairs, 3), dtype=np.int32), columns=['h', 'r', 't'])
        reverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        reverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        cf2kg_test_data = pd.DataFrame(np.zeros((self.n_test_ui_pairs, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_test_data['h'] = self.cf_test_data[0]
        cf2kg_test_data['t'] = self.cf_test_data[1]

        reverse_cf2kg_test_data = pd.DataFrame(np.ones((self.n_test_ui_pairs, 3), dtype=np.int32), columns=['h', 'r', 't'])
        reverse_cf2kg_test_data['h'] = self.cf_test_data[1]
        reverse_cf2kg_test_data['t'] = self.cf_test_data[0]

        self.kg_data = pd.concat([kg_data, cf2kg_train_data, reverse_cf2kg_train_data, cf2kg_test_data, reverse_cf2kg_test_data], ignore_index=True)
        self.kg_data = self.kg_data.drop_duplicates() 
        self.kg_data = list(self.kg_data.itertuples(index=False, name=None))

        self.n_kg = len(self.kg_data)
        
        # construct kg dict
        self.h_dict, self.t_dict = defaultdict(dict), defaultdict(dict)
        for h, r, t in self.kg_data:
            if r not in self.h_dict[h]:
                self.h_dict[h][r] = set()
            if r not in self.t_dict[t]:
                self.t_dict[t][r] = set()
            self.h_dict[h][r].add(t)
            self.t_dict[t][r].add(h)


    def print_info(self, logging):
        logging.info('n_users:            %d' % self.n_users)
        logging.info('n_items:            %d' % self.n_items)
        logging.info('n_org_entities:     %d' % self.n_org_entities)
        logging.info('n_entities:         %d' % self.n_entities)
        logging.info('n_relations:        %d' % self.n_relations)

        logging.info('n_train_ui_pairs:   %d' % self.n_train_ui_pairs)
        logging.info('n_test_ui_pairs:    %d' % self.n_test_ui_pairs)

        logging.info('n_cf_train:         %d' % self.n_cf_train)
        logging.info('n_cf_test:          %d' % self.n_cf_test)
        logging.info('n_kg:               %d' % self.n_kg)


    def create_graph(self, kg_data: list, n_nodes):
        return super().create_graph(pd.DataFrame(kg_data, columns =['h', 'r', 't'], dtype = int), n_nodes)


    def generate_kg_batch(self, batch_id, all_ents):
        return super().generate_kg_batch(batch_id, all_ents)


    def generate_cf_batch(self, batch_id, item_entset: set, train=True):
        if train: 
            user_id_list = self.train_user_ids
            user_dict = self.train_user_dict
            B = self.cf_batch_size
        else: 
            user_id_list = self.test_user_ids
            user_dict = self.test_user_dict
            B = self.args.test_batch_size
        batch_user_ids = user_id_list[batch_id*B : (batch_id+1)*B]  
        
        batch_item_pos_ids, batch_item_neg_ids = [], []  # list(LongTensor(1D))
        for u_id in batch_user_ids:
            batch_item_pos_ids.append(torch.LongTensor(list(user_dict[u_id])))
            batch_item_neg_ids.append(torch.LongTensor(
                self.sample_neg_items_for_u(user_dict, u_id, item_entset)))

        return batch_user_ids, batch_item_pos_ids, batch_item_neg_ids