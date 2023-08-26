import os, sys
sys.path.append(os.path.abspath('..'))
import argparse
import gendata.cyber.cyberkg_query as qa
import gendata.cyber.cyberkg_zeroday as zeroday

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_path', default='/data/zhaohan/adv-reasoning/save/data/cyberkg', type=str, help='previously saved KG')
    parser.add_argument('--q_path', default='/data/zhaohan/adv-reasoning/save/data/cyberQ_AB', type=str, help='save path for generated cyber queries')

    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--zeroday', action='store_true')
    parser.add_argument('--zeroday_ratio', default=0.25, type=float, help='how much answers (CVE-IDs) are set as zero-day data')

    return parser.parse_args(args)


def query_req(args):
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
        ['2i', 40000], ['1pp.2i', 20000], ['1ppp.2i', 20000],
        ['3i', 30000], ['1pp.3i', 20000], ['1ppp.3i', 20000],
        # ['5i', 30000], ['1pp.5i', 8000], ['2pp.5i', 8000], ['1ppp.5i', 8000], ['1ppp.1pp.5i', 8000],
    ]
    
    test_reqs = [
        ['2i', 300], ['1pp.2i', 300], ['1ppp.2i', 300],
        ['3i', 300], ['1pp.3i', 300], ['1ppp.3i', 300],
        ['5i', 300], ['2pp.5i', 300], ['1ppp.1pp.5i', 300],
    ]

    if args.zeroday:
        test_reqs = [
        ['2i', 200], ['1pp.2i', 200], ['1ppp.2i', 200],
        ['3i', 200], ['1pp.3i', 200], ['1ppp.3i', 200],
        ['5i', 200], ['2pp.5i', 200], ['1ppp.1pp.5i', 200],
    ]

    return train_reqs, test_reqs

if __name__ == '__main__':
    args = parse_args()

    print('\n#-------- sampling cyber-QA set  ------#')

    ### QA generation, test qa first
    train_reqs, test_reqs = query_req(args)
    if args.zeroday:
        args.q_path = args.q_path + '_zeroday'
        zeroday.gen_test_zeroday_set(args, test_reqs)
    else:
        qa.gen_qa_set(args, test_reqs, 'test')
    qa.gen_qa_set(args, train_reqs, 'train')
    
    
    # NOTE: dont indenpently run this one, better run genkg_cyber.py first to get complete KG

    # python genq_cyber.py --verbose