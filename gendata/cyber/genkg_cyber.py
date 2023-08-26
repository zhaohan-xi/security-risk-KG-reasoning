from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
sys.path.append(os.path.abspath('..'))
import argparse
from helper.utils import set_global_seed
import gendata.cyber.cyberkg_backbone as bb
import gendata.cyber.cyberkg_bron as bron


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_path', default='/data/zhaohan/adv-reasoning/data/cyberkg-raw', type=str, 
        help='save path for raw files, including (1) crawled cve webpage information (under raw_path/cve), (2) downloaded cwe files, (3) extracted cve evidence, (4) mitigation info from NVD')
    parser.add_argument('--kg_path', default='/data/zhaohan/adv-reasoning/save/data/cyberkg', type=str, help='save path for generated cyberkg and queries')
    parser.add_argument('--bron_path', default='/data/zhaohan/adv-reasoning/data/bron_output_data', type=str, 
        help='loading path for generated BRON graph data, originally locating at ~/BRON/full_data/full_output_data')

    parser.add_argument('--cve_start_year', default=2000, type=int, help='strating year of used CVE-IDs')
    parser.add_argument('--cve_end_year', default=2022, type=int, help='ending year (included) of used CVE-IDs')
    # parser.add_argument('--pd_num_thre', nargs='+', default=[3, 9999], type=int, help='constrain how much products a vendor can have')
    # parser.add_argument('--ver_num_thre', nargs='+', default=[5, 999], type=int, help='constrain how much versions a product can have')
    parser.add_argument('--mini_miti_thre', default=2, type=int, help='constrain minimum number of cve-ids a mitigation code can be used to')
    parser.add_argument('--full', action='store_true', help='no filtering of KG, kept original info (maybe noisy), which has 156k CVEs, 754k entities and 4.8milli facts with reverse relations')

    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--verbose', action='store_true')
    # parser.add_argument('--use_case', default='cyberkg', type=str, help="'cyberkg', 'cyberkg_L', 'zeroday'")
    # parser.add_argument('--zeroday_ratio', default=0.25, type=float, help='how much answers (CVE-IDs) are set as zero-day data')
    
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    set_global_seed(args.seed)

    print('#-------- building cyber-kg ------#')

    # step1: 1hop info (product, campaign, mitigation)
    entset, factset, cve_desc = bb.get_cve_info(args)
    entset, factset, nvd_cve_miti_map = bb.augment_cve_nvdmiti(args, entset, factset)
    # entset, factset, miti_keywords, cwe_miti_dict = bb.augment_cwe_miti(args, entset, factset)
    if not args.full:  # filter a smaller KG
        entset, factset = bb.refine_kg_by_1hop_cve_info(args, entset, factset)

    # step2: add multiple hop info (BRON technique-centric evidence)
    entset, factset = bron.augment_cve_evidence_by_BRON(args, entset, factset)

    # step3: proceed to save
    factset = bb.add_rev_rel(factset)
    ent2id, id2ent, rel2id, id2rel, entid2cate = bb.gen_id_map(entset, factset)
    bb.save_kg(args, entset, factset, ent2id, id2ent, rel2id, id2rel, entid2cate)
    # bb.save_file(args, cve_evi_by_type, 'cve_evi_by_type.pkl', format='pickle')
    bb.save_file(args, nvd_cve_miti_map, 'nvd_cve_miti_map.pkl', format='pickle')
    # bb.save_file(args, cwe_miti_dict, 'cwe_miti_dict.pkl', format='pickle')
    # bb.save_file(args, miti_keywords, 'miti_keywords.pkl', format='pickle')

    # python genkg_cyber.py --verbose