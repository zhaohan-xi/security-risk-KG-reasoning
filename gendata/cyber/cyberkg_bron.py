import os, json
import cyberkg_utils as cyber
from cyberkg_backbone import (ent_prefix, 
                            rel_dict, 
                            ent_prefix_delimiter)

def augment_cve_evidence_by_BRON(args, entset, factset, cleanup=False):
    '''
        entset, factset: string set (org names)
        cleanup: True - filter BRON part by only keeping linked entities/paths to our CVE
                 False - keep original BRON part (besides its CPE)
    '''
    bron_graph = json.load(open(os.path.join(args.bron_path, 'BRON.json'), 'rb'))

    def extend_mapped_id_with_prefix(x2bron_idmap, prefix):
        # org xxx2bron_idmap has no prefix in bron_id
        # but BRON facts has prefix 'cve_', 'cwe_', ...
        # here we simply extend dict values with prefix
        if not prefix.endswith('_'):
            prefix = prefix + '_'
        return {k: prefix+v for k, v in x2bron_idmap.items()}

    # build reverse maps
    cve2bron_idmap = json.load(open(os.path.join(args.bron_path, 
                                                'BRON/original_id_to_bron_id', 
                                                'cve_id_bron_id.json'), 'rb'))
    cwe2bron_idmap = json.load(open(os.path.join(args.bron_path, 
                                                'BRON/original_id_to_bron_id', 
                                                'cwe_id_to_bron_id.json'), 'rb'))
    capec2bron_idmap = json.load(open(os.path.join(args.bron_path, 
                                                'BRON/original_id_to_bron_id', 
                                                'capec_id_to_bron_id.json'), 'rb'))
    tech2bron_idmap = json.load(open(os.path.join(args.bron_path, 
                                                'BRON/original_id_to_bron_id', 
                                                'technique_id_to_bron_id.json'), 'rb'))
    tac2bron_idmap = json.load(open(os.path.join(args.bron_path, 
                                                'BRON/original_id_to_bron_id', 
                                                'tactic_name_to_bron_id.json'), 'rb'))

    cve2bron_idmap = extend_mapped_id_with_prefix(cve2bron_idmap, 'cve_')
    cwe2bron_idmap = extend_mapped_id_with_prefix(cwe2bron_idmap, 'cwe_')
    capec2bron_idmap = extend_mapped_id_with_prefix(capec2bron_idmap, 'capec_')
    tech2bron_idmap = extend_mapped_id_with_prefix(tech2bron_idmap, 'technique_')
    tac2bron_idmap = extend_mapped_id_with_prefix(tac2bron_idmap, 'tactic_')

    bron2cve_idmap = {v: k for k, v in cve2bron_idmap.items()}
    bron2cwe_idmap = {v: k for k, v in cwe2bron_idmap.items()}
    bron2capec_idmap = {v: k for k, v in capec2bron_idmap.items()}
    bron2tech_idmap = {v: k for k, v in tech2bron_idmap.items()}
    bron2tac_idmap = {v: k for k, v in tac2bron_idmap.items()}

    # step1: join CVE between ours and BRON's
    
    # NOTE: during query generation, we only use CVEs
    #       who has all kinds of evidences as cdd_cve
    cve_bronid = set()
    for edge in bron_graph['edges']:
        h, t, _ = edge
        if h.startswith('cve_'):
            cve_bronid.add(h)
        elif t.startswith('cve_'):
            cve_bronid.add(t)
    cve_ids_in_BRON = set([bron2cve_idmap[bid] for bid in cve_bronid])

    if cleanup:
        common_cveid = cve_ids_in_BRON & entset['cve-id']
        common_cve_bronid = set([cve2bron_idmap[_id] for _id in common_cveid])
        
    # step2: from cve to cwe

    kept_cwe_bronid = set()
    for edge in bron_graph['edges']:
        h, t, _ = edge
        # no need to consider reverse cve->cwe, both have same num
        if h.startswith('cwe') and t.startswith('cve'):  
            if not cleanup or t in common_cve_bronid:
                kept_cwe_bronid.add(h) 
                cwe_id = cyber.extend_weakness_name(
                    bron2cwe_idmap[h], ent_prefix['weakness'], ent_prefix_delimiter)
                cve_id = bron2cve_idmap[t]

                entset['cve-id'].add(cve_id)
                entset['weakness'].add(cwe_id)
                factset[rel_dict['weakness:cve-id']].add((cwe_id, rel_dict['weakness:cve-id'], cve_id))
                
    # add more cwe and extend CWE to KG
    for cwe_id in entset['weakness']:
        cwe_id = cwe_id.split(ent_prefix_delimiter)[-1] # remove prefix
        if cwe_id in cwe2bron_idmap:
            kept_cwe_bronid.add(cwe2bron_idmap[cwe_id])

    # step3 from cwe to attack pattern

    kept_capec_bronid = set()
    for edge in bron_graph['edges']:
        h, t, _ = edge
        if h.startswith('capec') and t.startswith('cwe'):
            if not cleanup or t in kept_cwe_bronid:
                kept_capec_bronid.add(h)
                capec_id = cyber.extend_evidence_name(
                        bron2capec_idmap[h], ent_prefix['attack-pattern'], ent_prefix_delimiter)
                cwe_id = cyber.extend_weakness_name(
                        bron2cwe_idmap[t], ent_prefix['weakness'], ent_prefix_delimiter)
            
                entset['weakness'].add(cwe_id)
                entset['attack-pattern'].add(capec_id)
                factset[rel_dict['attack-pattern:weakness']].add((capec_id, rel_dict['attack-pattern:weakness'], cwe_id))

    # step4: from attack pattern to technique

    kept_tech_bronid = set()
    for edge in bron_graph['edges']:
        h, t, _ = edge
        if h.startswith('technique') and t.startswith('capec'):
            if not cleanup or t in kept_capec_bronid:
                kept_tech_bronid.add(h)
                tech_id = cyber.extend_evidence_name(
                        bron2tech_idmap[h], ent_prefix['technique'], ent_prefix_delimiter)
                capec_id = cyber.extend_evidence_name(
                        bron2capec_idmap[t], ent_prefix['attack-pattern'], ent_prefix_delimiter)
                
                entset['attack-pattern'].add(capec_id)
                entset['technique'].add(tech_id)
                factset[rel_dict['technique:attack-pattern']].add((tech_id, rel_dict['technique:attack-pattern'], capec_id))
        
    # step5: from technique to tactic
    for edge in bron_graph['edges']:
        h, t, _ = edge
        if h.startswith('tac') and t.startswith('technique'):
            if not cleanup or t in kept_tech_bronid:
                tac_id = tech_id = cyber.extend_evidence_name(
                        bron2tac_idmap[h], ent_prefix['tactic'], ent_prefix_delimiter)
                tech_id = tech_id = cyber.extend_evidence_name(
                        bron2tech_idmap[t], ent_prefix['technique'], ent_prefix_delimiter)

                entset['technique'].add(tech_id)
                entset['tactic'].add(tac_id)
                factset[rel_dict['tactic:technique']].add((tac_id, rel_dict['tactic:technique'], tech_id))
    
    return entset, factset