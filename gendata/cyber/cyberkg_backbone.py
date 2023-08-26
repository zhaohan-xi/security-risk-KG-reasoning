import os, ast, pickle, json
import pandas as pd
from tqdm import tqdm
from datetime import date, datetime
from collections import defaultdict, OrderedDict
import genkg.cyberkg_utils as cyber

ent_prefix = cyber.ent_prefix
ent_prefix_delimiter = cyber.ent_prefix_delimiter
rel_dict = cyber.rel_dict
rev_rel_prefix = cyber.rev_rel_prefix

#--------------------------------------------------------------------------------------------------#
# Functions in blow block are used to extract and process raw info (all string-based format)
#--------------------------------------------------------------------------------------------------#

def get_cve_info(args):
    '''
    Collect Entities & Facts
    
    entset = { key: set() }
    factset = { key: set() }

    entset keys:'cve-id' + keys of ent_prefix
    factset keys: all relations in rel_dict

    all reversed relations will also be added besides 'cve:is:related:to:cve'

    NOTE: we don't filter cve or other entities when building KG, 
        just ignoring 'None' items
    '''
    start_year, end_year = args.cve_start_year, args.cve_end_year
    # pd_thre, ver_thre = args.pd_num_thre, args.ver_num_thre

    cve_paths = [os.path.join(args.raw_path, 'cve', str(y)) for y in range(start_year, end_year+1)]
    load_paths = []
    for path in cve_paths:
        for root, _, filenames in os.walk(path):
            load_paths.extend([os.path.join(root, filename) for filename in filenames])
    
    entset = defaultdict(set)
    factset = defaultdict(set)
    cve_desc = defaultdict(str)
    cve_count = 0  # count all crawled cve_num
    for path in tqdm(load_paths, desc='parsing crawled CVE data'):
        data = pd.read_csv(path, delimiter='|',  index_col=0, header=0)
        for index, row in data.iterrows():
            cve_count += 1

            # I. CVE id
            cve_id = row['cve-id']
            assert cve_id.startswith('CVE') # valid, not 'None'
            entset['cve-id'].add(cve_id)

            # II. Campaign (vulnerability types)
            if not row['vulner-types'] == 'None':
                vulner_types = row['vulner-types'].split(',')
                vulner_types = [
                    cyber.extend_evidence_name(
                        vul_type, ent_prefix['campaign'], ent_prefix_delimiter
                        ) for vul_type in vulner_types]
                entset['campaign'] |= set(vulner_types)
                for vul_type in vulner_types:
                    factset[rel_dict['cve-id:campaign']].add((cve_id, rel_dict['cve-id:campaign'], vul_type))

            # III. CWE id
            if not row['cwe-id'] == 'None':
                cwe_id = cyber.extend_weakness_name(row['cwe-id'] , ent_prefix['weakness'], ent_prefix_delimiter)
                entset['weakness'].add(cwe_id)
                factset[rel_dict['weakness:cve-id']].add((cwe_id, rel_dict['weakness:cve-id'], cve_id))

                # if row['cwe-rel']!='None':
                #     for rel_cwe_row in row['cwe-rel'].split(';'):
                #         rel, rel_cwe_id, rel_cwe_name = rel_cwe_row.split(',')
                #         rel = 'cwe:is:' + rel + ':cwe'
                #         entity_set['cwe-id'].add(rel_cwe_id)
                #         fact_set[rel].add((cwe_id, rel, rel_cwe_id))

            # IV. Vendor, Product, Version
            if not row['pd-info'] == 'None':
                pd_info = [p.split(',') for p in set(row['pd-info'].split(';'))]
                
                for p in pd_info:
                    _, vendor, product, version = p
                    if (vendor == 'None') or (product == 'None') or (version == 'None'): continue
                    org_product_name = product

                    # avoid dup name between vendor & products
                    vendor = cyber.extend_vendor_name(vendor, ent_prefix['vendor'], ent_prefix_delimiter)
                    product = cyber.extend_product_name(product, ent_prefix['product'], ent_prefix_delimiter)
                    entset['vendor'].add(vendor)
                    entset['product'].add(product)
                    factset[rel_dict['vendor:product']].add((vendor, rel_dict['vendor:product'], product))
                    factset[rel_dict['cve-id:vendor']].add((cve_id, rel_dict['cve-id:vendor'], vendor))
                    factset[rel_dict['cve-id:product']].add((cve_id, rel_dict['cve-id:product'], product))

                    if len(version)>0 and version!='-' and version!='*':
                        version = cyber.extend_version_name(org_product_name, version, ent_prefix['version'], 
                                                            ent_prefix_delimiter, cyber.ver_delimiter)
                        entset['version'].add(version)
                        factset[rel_dict['product:version']].add((product, rel_dict['product:version'], version))
                        factset[rel_dict['cve-id:version']].add((cve_id, rel_dict['cve-id:version'], version))

            # V. CVE-CVE and CVE texts
            # NOTE: CVEs in entset may not in cve_desc
            if not row['cve-desc'] == 'None':
                cve_desc[cve_id] = row['cve-desc'].strip().strip("'").strip('"') 
                for related_cve in cyber.cve_from_desc(row['cve-desc']):
                    if int(related_cve.split('-')[1]) < start_year: continue
                    assert related_cve.startswith('CVE')
                    entset['cve-id'].add(related_cve)
                    factset[rel_dict['cve-id:cve-id']].add((cve_id, rel_dict['cve-id:cve-id'], related_cve))
                    factset[rel_dict['cve-id:cve-id']].add((related_cve, rel_dict['cve-id:cve-id'], cve_id))
                    
    os.makedirs(args.kg_path, exist_ok=True)
    print('\noriginal crawled CVE number %d\n' % cve_count)

    # verification: no open edge
    all_ents = set()
    for v in entset.values():
        all_ents |= v
    for rel, facts in factset.items():
        for h, r, t in facts:
            assert r == rel, (r, rel)
            assert h in all_ents, (h, r)
            assert t in all_ents, (t, r)
    return entset, factset, cve_desc


def refine_kg_by_1hop_cve_info(args, entset, factset):
    # NOTE: refine KG by 1hop cve info (vendor, pd, version, campaign, mitigation, cwe)
    #       should included all above type of nodes before using this func

    # step1: no 'None' left from raw crawled info
    for k, v in entset.items():
        assert v != 'None', "'None' exist in entset['%s']" % k
    for k, v in factset.items():
        for h, r, t in v:
            assert h != 'None', "'None' exist in factset['%s']" % k
            assert r != 'None', "'None' exist in factset['%s']" % k
            assert t != 'None', "'None' exist in factset['%s']" % k
    
    cve_vd_map = defaultdict(set)
    cve_pd_map = defaultdict(set)
    cve_ver_map = defaultdict(set)
    cve_cam_map = defaultdict(set)
    cve_cwe_map = defaultdict(set)
    cve_miti_map = defaultdict(set)
    for h, r, t in factset[rel_dict['cve-id:vendor']]:
        assert r == rel_dict['cve-id:vendor']
        assert h in entset['cve-id']
        assert t in entset['vendor']
        cve_vd_map[h].add(t)
    for h, r, t in factset[rel_dict['cve-id:product']]:
        assert r == rel_dict['cve-id:product']
        assert h in entset['cve-id']
        assert t in entset['product']
        cve_pd_map[h].add(t)
    for h, r, t in factset[rel_dict['cve-id:version']]:
        assert r == rel_dict['cve-id:version']
        assert h in entset['cve-id']
        assert t in entset['version']
        cve_ver_map[h].add(t)
    for h, r, t in factset[rel_dict['cve-id:campaign']]:
        assert r == rel_dict['cve-id:campaign']
        assert h in entset['cve-id']
        assert t in entset['campaign']
        cve_cam_map[h].add(t)
    for h, r, t in factset[rel_dict['weakness:cve-id']]:
        assert r == rel_dict['weakness:cve-id']
        assert h in entset['weakness']
        assert t in entset['cve-id']
        cve_cwe_map[t].add(h)
    for h, r, t in factset[rel_dict['mitigation:cve-id']]:
        assert r == rel_dict['mitigation:cve-id']
        assert h in entset['mitigation']
        assert t in entset['cve-id']
        cve_miti_map[t].add(h)

    keep_cve_from_vd = set([k for k, v in cve_vd_map.items() if len(v)>=1])
    keep_cve_from_pd = set([k for k, v in cve_pd_map.items() if len(v)>=1])
    keep_cve_from_ver = set([k for k, v in cve_ver_map.items() if len(v)>=4])
    keep_cve_from_cam = set([k for k, v in cve_cam_map.items() if len(v)>=1])
    keep_cve_from_cwe = set([k for k, v in cve_cwe_map.items() if len(v)>=1])
    keep_cve_from_miti = set([k for k, v in cve_miti_map.items() if len(v)>=1])


    # keep_cve_from_pd = set([k for k, v in cve_pd_map.items() if len(v)>=2])
    keep_cve_from_ver = set([k for k, v in cve_ver_map.items() if len(v)>=2])
    keep_cve_from_ver &= set([k for k, v in cve_ver_map.items() if len(v)<=50])
    keep_cve_from_miti &= set([k for k, v in cve_miti_map.items() if len(v)<=30])

    keep_cve = keep_cve_from_vd & keep_cve_from_pd & keep_cve_from_ver & keep_cve_from_cam & keep_cve_from_cwe & keep_cve_from_miti
    entset['cve-id'] = entset['cve-id'] & keep_cve

    # clean vd, pd, ver entities who doesn't connect any cves
    # tip: cwe,cam,miti nodes can be isolated after disconnecting 
    # with any cve, but vd,pd,ver nodes may not
    keep_vd = set()
    for h, r, t in factset[rel_dict['cve-id:vendor']]:
        assert h.startswith('CVE')
        if h in entset['cve-id']:
            keep_vd.add(t)
    keep_pd = set()
    for h, r, t in factset[rel_dict['cve-id:product']]:
        assert h.startswith('CVE')
        if h in entset['cve-id']:
            keep_pd.add(t)
    keep_ver = set()
    for h, r, t in factset[rel_dict['cve-id:version']]:
        assert h.startswith('CVE')
        if h in entset['cve-id']:
            keep_ver.add(t)

    entset['vendor'] = entset['vendor'] & keep_vd
    entset['product'] = entset['product'] & keep_pd
    entset['version'] = entset['version'] & keep_ver

    # clean facts whose entities not in entset
    all_ent = set()
    for k, v in entset.items():
        all_ent = all_ent | v
    for k, v in factset.items():
        to_pop = set()
        for f in v:
            h, r, t = f
            if (h not in all_ent) or (t not in all_ent):
                to_pop.add(f)
        factset[k] = factset[k] - to_pop

    # clean isolated entities
    all_ent = set()
    for rel in factset:
        for fact in factset[rel]:
            all_ent.add(fact[0])
            all_ent.add(fact[2])
    for cate in entset:
        entset[cate] = entset[cate] & all_ent
            
    return entset, factset


# NOTE: deprecated
def parse_cwe_info(args):
    cwe_path = os.path.join(args.raw_path, 'cwe')
    cwe_dict = OrderedDict()
    for root, _, filenames in os.walk(cwe_path):
        for filename in filenames:
            csv_path = os.path.join(root, filename)
            data = pd.read_csv(csv_path, delimiter=',', index_col=False, header=0)
            for index, row in data.iterrows():
                cwe_id = str(row['CWE-ID'])
                cwe_name = row['Name']
                rel_cwe = row['Related Weaknesses']
                miti = row['Potential Mitigations']
                rel_cve = row['Observed Examples']
                
                assert isinstance(cwe_name, str)
                if cwe_id not in cwe_dict:
                    cwe_dict[cwe_id] = {
                        'name': cwe_name,
                        'rel-cwe': set(),  # set(tuple(rel, cwe-id))
                        'rel-cve': set(),  # set(tuple(cve-id, desc))
                        'mitigation': set(), # set(tuple(phase, strategy, desc))
                    }
                if isinstance(rel_cwe, str): # not nan
                    rel_cwe = [s.split(':') for s in rel_cwe.strip(':').split('::')]
                    for s in rel_cwe:
                        to_add = ['None', 'None']
                        for i in range(len(s)):
                            if s[i] == 'NATURE' and i<len(s)-1:
                                to_add[0] = s[i+1]
                            elif s[i] == 'CWE ID' and i<len(s)-1:
                                to_add[1] = s[i+1]
                        cwe_dict[cwe_id]['rel-cwe'].add(tuple(to_add))
                if isinstance(rel_cve, str):
                    rel_cve = [s.split(':') for s in rel_cve.strip(':').split('::')]
                    for s in rel_cve:
                        to_add = ['None', 'None']
                        for i in range(len(s)):
                            if s[i] == 'REFERENCE' and i<len(s)-1:
                                to_add[0] = s[i+1]
                            elif s[i] == 'DESCRIPTION' and i<len(s)-1:
                                to_add[1] = s[i+1]
                        cwe_dict[cwe_id]['rel-cve'].add(tuple(to_add))
                if isinstance(miti, str):
                    miti = [s.split(':') for s in miti.strip(':').split('::')]
                    for s in miti:
                        to_add = ['None', 'None', 'None']
                        for i in range(len(s)):
                            if s[i] == 'PHASE' and i<len(s)-1:
                                to_add[0] = s[i+1]
                            elif s[i] == 'STRATEGY' and i<len(s)-1:
                                to_add[1] = s[i+1]
                            elif s[i] == 'DESCRIPTION' and i<len(s)-1:
                                to_add[2] = s[i+1]
                        cwe_dict[cwe_id]['mitigation'].add(tuple(to_add))
                    
    # with open(os.path.join(args.save_path, 'cwe_dict.pkl'), 'wb') as pklfile:
    #     pickle.dump(cwe_dict, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    #     print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
    return cwe_dict


# NOTE: deprecated
def augment_cve_evidence(args, entset, factset, cve_desc, cwe_dict):
    cve_evi_path = os.path.join(args.raw_path, 'cve-evidence')
    all_ents = set()
    for k, v in entset.items():
        for ent in v:
            all_ents.add(ent.split(':')[-1])
    load_paths = []
    for root, _, filenames in os.walk(cve_evi_path):
        load_paths.extend([os.path.join(root, filename) for filename in filenames])

    evi_info = defaultdict(dict)
    # reltype2fq = {}
    for f in tqdm(load_paths, desc='parsing CVE-related evidence'):
        cve_id = f.split('/')[-1].split('.')[0]
        if cve_id in entset['cve-id']:
            with open(f, 'r') as file:
                data = ast.literal_eval(file.read().replace('\n', ''))
                for ent in data['entities']:
                    evi = ent['canonicalForm']
                    if len(evi) == 0 or evi in all_ents or \
                    len(evi.strip('1234567890.,v-; ')) == 0 or \
                    len(evi.strip('CVE1234567890-, ')) == 0:
                        continue
                    if evi not in evi_info:
                        evi_info[evi] = {
                            'type2subtype': defaultdict(set),
                            'subType': set(),
                            'cve-ids': set(),
                        }
                    evi_info[evi]['type2subtype'][ent['type']].add(ent['subType'])
                    evi_info[evi]['subType'].add(ent['subType'])
                    evi_info[evi]['cve-ids'].add(cve_id)
    evi2fq = {}
    for evi in evi_info:
        evi2fq[evi] = len(evi_info[evi]['cve-ids'])

    type2evi = defaultdict(set)
    subtype2evi = defaultdict(set)
    type2subtype = defaultdict(set)
    for evi, fq in evi2fq.items():
        if fq >= 5:
            for t in evi_info[evi]['type2subtype'].keys():
                type2evi[t].add(evi)
                for subt in evi_info[evi]['type2subtype'][t]:
                    subtype2evi[subt].add(evi)
                    type2subtype[t].add(subt)

    # filtered types: [
    # 'ThreatActor', 'MiscEntity', 'TTP', 'Identity', 'DomainName', 
    # 'Vulnerability', 'Product', 'IpAddress', 'MalwareFamily', 
    # 'Campaign', 'Protocol', 'AvSignature', 'FileName', 'Person']
    # 
    # where the 
    #     'ThreatActor': we will remove non-sense terms like '15.0.0.249 allow attacker' and strip('-')
    #      we will also map org category to a new name that identical to entset.keys()

    evi_name_map = {  # org name : new name in entset
        'ThreatActor': 'threat-actor',
        'TTP': 'TTP',
        'Identity': 'threat-actor',
        'DomainName': 'threat-actor',
        'Vulnerability': 'incident',
        'IpAddress': 'threat-actor',
        'MalwareFamily': 'threat-actor',
        'Campaign': 'campaign',
        'FileName': 'threat-actor',
        'Product': 'threat-actor',
        'Person': 'threat-actor',
        'FilePath': 'threat-actor',
    }
    # print(type2evi.keys())
    tmp_evi_name_map = {}
    for k, v in evi_name_map.items():
        if k in type2evi.keys():
            tmp_evi_name_map[k] = v
    evi_name_map = tmp_evi_name_map

    # filter non-sense evidence, TODO: add more filtering condition
    cve_evidence_by_type = defaultdict(set) # org type name, no prefix
    for _type in evi_name_map.keys():
        if _type == 'ThreatActor':
            for evi in type2evi[_type]:
                if evi.strip('1234567890. ')=='allow attacker': 
                    continue
                cve_evidence_by_type[_type].add(evi)
        else:
            cve_evidence_by_type[_type] = type2evi[_type]

    # add evi into entset and cve-evi into factset
    for _type, evidences in cve_evidence_by_type.items(): # no prefix
        assert _type in evi_name_map, evi_name_map[_type] in ent_prefix
        _type = evi_name_map[_type]
        
        for cve_id, desc in cve_desc.items():
            for evi in evidences:
                if evi in desc:
                    assert cve_id in entset['cve-id']
                    add_evi= cyber.extend_evidence_name(evi, ent_prefix[_type], ent_prefix_delimiter)
                    entset[_type].add(add_evi)
                    add_rel = rel_dict['cve-id:%s' % _type]
                    factset[add_rel].add((cve_id, add_rel, add_evi))

    # add evi-evi into factset, based on entset/rel_dict keys
    cve_evi_maps = defaultdict(dict)  # {cve-id: {evi-type: [evi]}}
    for rel, facts in factset.items():
        if rel not in [rel_dict['cve-id:threat-actor'], 
                       rel_dict['cve-id:campaign'], 
                       rel_dict['cve-id:TTP'], 
                       rel_dict['cve-id:incident']]:
            continue
        for h, r, t in facts:  # no rev relation so far
            assert r==rel and h in entset['cve-id'] and t not in entset['cve-id']
            for cate in entset.keys():
                if t in entset[cate]:
                    if cate not in cve_evi_maps[h]:
                        cve_evi_maps[h][cate] = []
                    cve_evi_maps[h][cate].append(t)

    for cve_id, maps in cve_evi_maps.items():
        if 'threat-actor' in maps.keys() and 'incident' in maps.keys():
            add_rel = rel_dict['threat-actor:incident']
            for h in maps['threat-actor']:
                for t in maps['incident']:
                    factset[add_rel].add((h, add_rel, t))

        if 'threat-actor' in maps.keys() and 'TTP' in maps.keys():
            add_rel = rel_dict['threat-actor:TTP']
            for h in maps['threat-actor']:
                for t in maps['TTP']:
                    factset[add_rel].add((h, add_rel, t))

        if 'threat-actor' in maps.keys() and 'campaign' in maps.keys():
            add_rel = rel_dict['threat-actor:campaign']
            for h in maps['threat-actor']:
                for t in maps['campaign']:
                    factset[add_rel].add((h, add_rel, t))

        if 'incident' in maps.keys() and 'TTP' in maps.keys():
            add_rel = rel_dict['incident:TTP']
            for h in maps['incident']:
                for t in maps['TTP']:
                    factset[add_rel].add((h, add_rel, t))

        if 'incident' in maps.keys() and 'campaign' in maps.keys():
            add_rel = rel_dict['incident:campaign']
            for h in maps['incident']:
                for t in maps['campaign']:
                    factset[add_rel].add((h, add_rel, t))

        if 'TTP' in maps.keys() and 'campaign' in maps.keys():
            add_rel = rel_dict['TTP:campaign']
            for h in maps['TTP']:
                for t in maps['campaign']:
                    factset[add_rel].add((h, add_rel, t))

    # adding cwe related info from cwe_dict
    #     adding cwe_name -> cve_id    (entity_set, fact_set)
    #     adding cwe_id -> cve_id      (entity_set, fact_set)
    #     adding cwe_id -> rel_cwe_id  (entity_set, fact_set)

    # for f in factset['cwe:refers:to:cve']:
    #     h, _, t  = f
    #     if h in cwe_dict: # the downloaded cwe csv files lack some cwe details
    #         cwe_name = cyber.extend_evidence_name(cwe_dict[h]['name'], 'cve:keyword:cwe:name:', ent_prefix_delimiter) 
    #         entset['cve-keyword'].add(cwe_name)
    #         factset['cve:keyword:cwe:name:describes:cve'].add((cwe_name, 'cve:keyword:cwe:name:describes:cve', t))
    # if cwe_dict is not None:
    #     for cwe_id in cwe_dict.keys():
    #         related_cwe = False
    #         for cve_id, _ in cwe_dict[cwe_id]['rel-cve']:
    #             if cve_id in entset['cve-id']:
    #                 related_cwe = True
    #                 entset['cwe-id'].add(cwe_id)
    #                 factset['cwe:refers:to:cve'].add((cwe_id, 'cwe:refers:to:cve', cve_id))
    #         if related_cwe:
    #             for rel, rel_cwe_id in cwe_dict[cwe_id]['rel-cwe']:
    #                 rel = 'cwe:is:' + rel + ':cwe'
    #                 entset['cwe-id'].add(rel_cwe_id)
    #                 factset[rel].add((cwe_id, rel, rel_cwe_id))

    return entset, factset, cve_evidence_by_type


def augment_cve_nvdmiti(args, entset, factset):
    load_path = os.path.join(args.raw_path, 'nvd_cve_info')
    load_paths = []
    for root, _, filenames in os.walk(load_path):
        load_paths.extend([os.path.join(root, filename) for filename in filenames])

    nvd_cve_miti_map = defaultdict(list)  # only keep not 'None'
    for p in load_paths:
        data = json.load(open(p, 'r'))
        for cve_id, info in data.items():
            for _, v in info['external'].items():
                if not v['url'] == 'None':
                    nvd_cve_miti_map[cve_id].append(v['url'])
    
    # each url as a mitigation code
    for cve_id in entset['cve-id']:
        if cve_id in nvd_cve_miti_map.keys():
            for url in nvd_cve_miti_map[cve_id]:
                entset['mitigation'].add(url)
                factset[rel_dict['mitigation:cve-id']].add((url, rel_dict['mitigation:cve-id'], cve_id))

    # clean kg by mitigation info
    entset, factset = clean_kg_by_miti(args, entset, factset)
    return entset, factset, nvd_cve_miti_map

# NOTE: deprecated
def clean_kg_by_miti(args, entset, factset):
    # remove mitigations with low usage frequency
    # remove cve-ids without any (NVD) mitigation info

    # step1. in 1 miti - N cve, constrain N >= args.mini_miti_thre
    miti_cve_map = defaultdict(set)
    for rel, facts in factset.items():
        if rel not in [rel_dict['mitigation:cve-id']]:
            continue
        for h, r, t in facts:
            assert r==rel
            assert h in entset['mitigation']
            assert t in entset['cve-id']
            miti_cve_map[h].add(t)

    keep_miti = set()
    for miti, cve_ids in miti_cve_map.items():
        if len(cve_ids) >= args.mini_miti_thre:
            keep_miti.add(miti)
    entset['mitigation'] = entset['mitigation'] & keep_miti
    
    # step2. update facts after remove some mitigation entities
    all_ent = set()
    for k, v in entset.items():
        all_ent = all_ent | v
    for k, v in factset.items():
        to_pop = set()
        for f in v:
            h, r, t = f
            if (h not in all_ent) or (t not in all_ent):
                to_pop.add(f)
        factset[k] = factset[k] - to_pop

    # step3. keep cve-ids still with mitigation info
    keep_cve = set()
    for rel, facts in factset.items():
        if rel not in [rel_dict['mitigation:cve-id']]:
            continue
        for h, r, t in facts:  # no rev relation so far
            assert r==rel and t in entset['cve-id'] and h not in entset['cve-id']
            keep_cve.add(t)
    entset['cve-id'] = entset['cve-id'] & keep_cve

    # step4. update facts after remove some cve entities
    all_ent = set()
    for k, v in entset.items():
        all_ent = all_ent | v
    for k, v in factset.items():
        to_pop = set()
        for f in v:
            h, r, t = f
            if (h not in all_ent) or (t not in all_ent):
                to_pop.add(f)
        factset[k] = factset[k] - to_pop

    # step5. clean isolated entities in entset
    all_ent = set()
    for rel in factset:
        for fact in factset[rel]:
            all_ent.add(fact[0])
            all_ent.add(fact[2])
    for cate in entset:
        entset[cate] = entset[cate] & all_ent

    return entset, factset


# NOTE: depricated
def augment_cwe_miti(args, entity_set, fact_set, cwe_dict):
    miti_desc = []
    for cwe_id in cwe_dict:
        if cwe_id in entity_set['cwe-id']:
            for _, _, desc in cwe_dict[cwe_id]['mitigation']:
                miti_desc.append(desc)

    miti_corpus, miti_kw2freq = cyber.find_keyword_set(miti_desc, 
                                                        method='yake', 
                                                        setting={'n_gram': 5},
                                                        word_class=None,
                                                        task='mitigation')
    filtered_freq = cyber.filter_list_w_percentile(miti_kw2freq.values(), 0.025, 0.25)
    miti_keywords = set()
    cwe_miti_dict = defaultdict(dict) # {cwe_name: {miti_stage: set(miti_kws)}}, all str
    for kw, fq in miti_kw2freq.items():
        if fq in filtered_freq:
            miti_keywords.add(kw)
    for cwe_id in cwe_dict:
        if cwe_id in entity_set['cwe-id']:
            for stage, _, desc in cwe_dict[cwe_id]['mitigation']:
                for kw in miti_keywords:
                    if kw in desc:
                        add_stage = cyber.extend_mitigation_name(stage, 'miti:stage:')
                        add_kw = cyber.extend_mitigation_name(kw, 'miti:method:')
                        entity_set['miti-stage'].add(add_stage)
                        entity_set['miti-method'].add(add_kw)
                        fact_set['miti:method:can:mitigate:cwe'].add((add_kw, 'miti:method:can:mitigate:cwe', cwe_id))
                        fact_set['miti:method:is:in:miti:stage'].add((add_kw, 'miti:method:is:in:miti:stage', add_stage))

                        if add_stage not in cwe_miti_dict[cwe_id]:
                            cwe_miti_dict[cwe_id][add_stage] = set()
                        cwe_miti_dict[cwe_id][add_stage].add(add_kw)
    # with open(os.path.join(args.kg_path, 'entset.pkl'), 'wb') as pklfile:
    #     pickle.dump(entity_set, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    #     print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
    # with open(os.path.join(args.kg_path, 'factset.pkl'), 'wb') as pklfile: # now without reverse relations
    #     pickle.dump(fact_set, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
    #     print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
    # with open(os.path.join(args.kg_path, 'miti_keywords.txt'), 'w') as txtfile: 
    #     for kw in miti_keywords:
    #         txtfile.write(kw+'\n')
    #     print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), txtfile.name))
    return entity_set, fact_set, miti_keywords, cwe_miti_dict


#--------------------------------------------------------------------------------------------------#
# Functions below are proceeding toward int the KG and saving
#--------------------------------------------------------------------------------------------------#

def gen_id_map(entset, factset):
    ent2id, id2ent = OrderedDict(), OrderedDict()
    rel2id, id2rel = OrderedDict(), OrderedDict()
    entid2cate = OrderedDict()
    for k, v in entset.items():
        for ent in v:
            _id = len(ent2id)
            ent2id[ent] = _id
            id2ent[_id] = ent
            entid2cate[_id] = k  # entity category
    for k, _ in factset.items(): # augmented
        _id = len(rel2id)
        rel2id[k] = _id
        id2rel[_id] = k

    return ent2id, id2ent, rel2id, id2rel, entid2cate


def add_rev_rel(factset):
    rev_factset = defaultdict(set)
    for rel in factset:
        if rel == rel_dict['cve-id:cve-id']:
            continue
        for h, r, t in factset[rel]:
            assert r == rel
            if not rel.startswith(rev_rel_prefix):
                add_rel = rev_rel_prefix + rel
                rev_factset[add_rel].add((t, add_rel, h))
            else:
                add_rel = rel[len(rev_rel_prefix):]
                rev_factset[add_rel].add((t, add_rel, h))
    factset.update(rev_factset)
    return factset


def save_kg(args, entset, factset, ent2id, id2ent, rel2id, id2rel, entid2cate):
    print('\n----- Saved KG Statistics -----')
    cyber.stat(entset, name='Knowledge entities')
    cyber.stat(factset, name='Augmented knowledge facts')

    with open(os.path.join(args.kg_path, 'entset.pkl'), 'wb') as pklfile: 
        pickle.dump(entset, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))

    with open(os.path.join(args.kg_path, 'factset.pkl'), 'wb') as pklfile: # with reverse relations, cover the old files
        pickle.dump(factset, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
    with open(os.path.join(args.kg_path, 'ent2id.pkl'), 'wb') as pklfile:
        pickle.dump(ent2id, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
    with open(os.path.join(args.kg_path, 'id2ent.pkl'), 'wb') as pklfile:
        pickle.dump(id2ent, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
    with open(os.path.join(args.kg_path, 'rel2id.pkl'), 'wb') as pklfile:
        pickle.dump(rel2id, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
    with open(os.path.join(args.kg_path, 'id2rel.pkl'), 'wb') as pklfile:
        pickle.dump(id2rel, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
    with open(os.path.join(args.kg_path, 'entid2cate.pkl'), 'wb') as pklfile:
        pickle.dump(entid2cate, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
    id_entset = defaultdict(set)
    for k, v in entset.items():
        for ent in v:
            id_entset[k].add(ent2id[ent])
            
    id_factset = defaultdict(set)
    for k, v in factset.items():
        for f in v:
            h, r, t = f
            assert r == k
            id_factset[rel2id[k]].add((ent2id[h], rel2id[r], ent2id[t]))
        
    with open(os.path.join(args.kg_path, 'id_entset.pkl'), 'wb') as pklfile:
        pickle.dump(id_entset, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
    with open(os.path.join(args.kg_path, 'id_factset.pkl'), 'wb') as pklfile:
        pickle.dump(id_factset, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))
        
    # save meta-info files
    with open(os.path.join(args.kg_path, 'stats.txt'), 'w') as txtfile:
        txtfile.write('numentity: %d\n' % len(ent2id))
        txtfile.write('numrelations: %d' % len(rel2id))
        print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), txtfile.name))

    with open(os.path.join(args.kg_path, 'detail_stats.txt'), 'w') as txtfile:
        txtfile.write('----- Entity detailed info -----\n')
        ent_num = 0
        for k, v in entset.items():
            txtfile.write('%s: %d\n' % (k, len(v)))
            ent_num += len(v)
        txtfile.write('Total entities: %d\n' % ent_num)
        fact_num = 0
        txtfile.write('\n----- Facts detailed info -----\n')
        for k, v in factset.items():
            txtfile.write('%s: %d\n' % (k, len(v)))
            fact_num += len(v)
        txtfile.write('Total facts: %d\n' % fact_num)
        print('%s %s Saved %s\n' % (date.today(), datetime.now().strftime("%H:%M:%S"), txtfile.name))

def save_file(args, content, name, format='txt'):
    if format == 'txt':
        if not name.endswith('.txt'):
            name += '.txt'
        assert isinstance(content, list)
        with open(os.path.join(args.kg_path, name), 'w') as txtfile:
            for l in content:
                txtfile.write(l)
                txtfile.write('\n')
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), txtfile.name))
    elif format in ['pickle', 'pkl']:
        if not name.endswith('.pkl'):
            name += '.pkl'
        with open(os.path.join(args.kg_path, name), 'wb') as pklfile:
            pickle.dump(content, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
            print('%s %s Saved %s' % (date.today(), datetime.now().strftime("%H:%M:%S"), pklfile.name))