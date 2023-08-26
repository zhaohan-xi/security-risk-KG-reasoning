import os, sys
sys.path.append(os.path.abspath('..'))
import argparse
import gendata.med.medkg_query as qa

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_path', default='/data/zhaohan/adv-reasoning/save/data/medkg/', type=str, help='previously saved KG')
    parser.add_argument('--q_path', default='/data/zhaohan/adv-reasoning/save/data/medQ_AB', type=str, help='save path for generated cyber queries')

    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args(args)


def query_req():
    '''
    Even though we may use 'xip' structures to query mitigation, 
    but those could be extended by 'xi' query for cve-id query,
    hence we only need to write 'xi' structures as requirements
    '''
    train_reqs = [
        # [('e', ('r',)), 500_000], 
        # [('e', ('r', 'r',)), 500_000],
        # [('e', ('r', 'r', 'r',)), 500_000],

        # adding all '1p' queries by default
        ['2i', 200000], ['1pp.2i', 200000],
        ['3i', 200000], ['1pp.3i', 200000], 
        # ['5i', 30000], ['1pp.5i', 8000], ['2pp.5i', 8000], ['1ppp.5i', 8000], ['1ppp.1pp.5i', 8000],
    ]
    
    test_reqs = [
        ['2i', 300], ['1pp.2i', 300], 
        ['3i', 300], ['1pp.3i', 300], 
        ['5i', 300], ['2pp.5i', 300], 
    ]

    return train_reqs, test_reqs

if __name__ == '__main__':
    args = parse_args()

    print('#-------- sampling med-QA set  ------#')

    # QA generation, test qa first
    train_reqs, test_reqs = query_req()
    qa.gen_qa_set(args, test_reqs, 'test')
    qa.gen_qa_set(args, train_reqs, 'train')

    # python genq_med.py --verbose