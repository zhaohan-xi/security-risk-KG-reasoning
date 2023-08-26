#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict

import random
import numpy as np
import torch

from torch.utils.data import Dataset
from helper.utils import list2tuple, tuple2list, flatten

# 'queries' == 'answers' in drkg case
# for ontological reasoning, queries: {(drug_eid, rid): set(ans eids)}
# for relational reasoning, queries: {(drug_eid, rpp_eid): set(ans rids)}

class TestDataset(Dataset):
    def __init__(self, args, queries: defaultdict, nentity, nrelation):
        self.args = args
        self.nentity = nentity
        self.nrelation = nrelation

        self.facts = []
        if self.args.reasoning_mode == 'ont':
            for k, v in queries.items():
                for eid in v:
                    self.facts.append((k[0], k[1], eid))
        elif self.args.reasoning_mode == 'rel':
            for k, v in queries.items():
                for rid in v:
                    self.facts.append((k[0], rid, k[1]))

    def __len__(self):
        return len(self.facts)
    
    def __getitem__(self, idx: int):
        pos_fact = torch.LongTensor(self.facts[idx])
        neg_facts = []

        ph, pr, pt = self.facts[idx]
        if self.args.reasoning_mode == 'ont':
            neg_ts = list(set(range(self.nentity)) - set([pt]))
            for nt in neg_ts:
                neg_facts.append((ph, pr, nt))
        elif self.args.reasoning_mode == 'rel':
            neg_rs = list(set(range(self.nrelation)) - set([pr]))
            for nr in neg_rs:
                neg_facts.append((ph, nr, pt))

        neg_facts = torch.LongTensor(neg_facts)
        return pos_fact, neg_facts

    
class TrainDataset(Dataset):
    def __init__(self, args, queries: defaultdict, nentity, nrelation):
        self.args = args
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = self.args.negative_sample_size
        
        self.facts = []
        if self.args.reasoning_mode == 'ont':
            for k, v in queries.items():
                for eid in v:
                    self.facts.append((k[0], k[1], eid))
        elif self.args.reasoning_mode == 'rel':
            for k, v in queries.items():
                for rid in v:
                    self.facts.append((k[0], rid, k[1]))
                    
    def __len__(self):
        return len(self.facts)
    
    def __getitem__(self, idx: int):
        pos_fact = torch.LongTensor(self.facts[idx])
        neg_facts = []

        ph, pr, pt = self.facts[idx]
        cdd_eids = set(range(self.nentity))
        neg_hs = random.choices(list(cdd_eids - set([ph])), k=self.negative_sample_size//2)
        neg_ts = random.choices(list(cdd_eids - set([pt])), k=self.negative_sample_size//2)
        for nh in neg_hs:
            neg_facts.append((nh, pr, pt))
        for nt in neg_ts:
            neg_facts.append((ph, pr, nt))

        neg_facts = torch.LongTensor(neg_facts)
        # pos_fact.shape: (3), 
        # neg_facts.shape: (neg_size, 3)
        return pos_fact, neg_facts
    