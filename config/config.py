import argparse
from email.policy import default
from random import choices

def add_com_group(group):
    group.add_argument('--cuda', action='store_true', help='use GPU')
    group.add_argument('--seed', default=0, type=int, help="random seed")

    group.add_argument('--kg_path', type=str, default=None, help="path to load KG")
    group.add_argument('--q_path', type=str, default=None, help="path to load query-anaswer set")
    group.add_argument('--log_folder', type=str, default='logs', help="folder name for experimental logs")
    group.add_argument('--save_path', default=None, type=str, help="no need to set manually, will configure automatically")

    group.add_argument('--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
    group.add_argument('--hidden_dim', default=400, type=int, help="embedding dimension")
    group.add_argument('--batch_size', default=512, type=int, help="batch size of queries")
    group.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    group.add_argument('--learning_rate', default=0.001, type=float)
    group.add_argument('--cpu_num', default=1, type=int, help="used to speed up torch.dataloader")

    group.add_argument('--print_on_screen', action='store_true')
    group.add_argument('--verbose', action='store_true')

    group.add_argument('--zeroday', action='store_true', help="use zeroday setting")
    
    group.add_argument('--yaml', default=None, type=str, help="configuration saving path at a .yaml file")
    group.add_argument('--debug_yaml', default='/data/zhaohan/adv-reasoning/config/debug.yaml', type=str,  help="configuration for fast running")
    group.add_argument('--debug', action='store_true', help="debug mode, fast run")

def add_krl_group(group):
    group.add_argument('--domain', default=None, type=str, choices=['cyber', 'med'], help='the domain of current experiment')
    
    group.add_argument('--do_train', action='store_true', help="do train")
    group.add_argument('--do_valid', action='store_true', help="do valid")
    group.add_argument('--do_test', action='store_true', help="do test")
    
    group.add_argument('--gamma', default=24, type=float, help="margin in the loss")
    group.add_argument('--krl_train_steps', default=100000, type=int, help="maximum iterations to train the reasoning model")
    group.add_argument('--warm_up_steps', default=None, type=int, help="no need to set manually, will configure automatically")
    group.add_argument('--save_checkpoint_steps', default=10000, type=int, help="save checkpoints every xx steps")
    group.add_argument('--valid_steps', default=10000, type=int, help="evaluate validation queries every xx steps")
    group.add_argument('--log_steps', default=500, type=int, help='train log every xx steps')
    group.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    group.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    group.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    group.add_argument('--model', default='box', type=str, help='the reasoning model, vec for GQE, box for Query2box, beta for BetaE, rgcn for RGCN')
    group.add_argument('-betam', '--beta_mode', default="(300,2)", type=str, help='(hidden_dim,num_layer) for BetaE relational projection')
    group.add_argument('-boxm', '--box_mode', default="(none,0.02)", type=str, help='(offset activation,center_reg) for Query2box, center_reg balances the in_box dist and out_box dist')
    group.add_argument('--cen_layer', type=int, default=4, help='# layers of center net')
    group.add_argument('--off_layer', type=int, default=4, help='# layers of Query2Box offset net')
    group.add_argument('--prj_layer', type=int, default=4, help='# layers of BetaE project net')

    group.add_argument('--krl_ckpt_path', default=None, type=str, help='path for loading the KRL checkpoints')


def add_atk_group(group):
    group.add_argument('--attack', default=None, choices=['kgp', 'eva', 'cop'], help="Type of current ROAR attack")
    group.add_argument('--under_attack', action='store_true', help="DON'T set manually, used to distinguish benign/adversarial KRL process")

    # init
    group.add_argument('--atk_kg_path', type=str, default=None, help="perturbed KG path")
    group.add_argument('--atk_q_path', type=str, default=None, help="initialized QA path (add trigger and split into ben/atk)")
    group.add_argument('--tar_evi', default=None, type=str, help="the evident entity in targeted logic the attacker aims to")
    group.add_argument('--tar_ans', default=None, type=str, help="a targeted answer the attacker aims to perturb the system always reasoning to (in targeted attack)")
    group.add_argument('--tar_evi_cate', default=None, type=str, help="targeted evident entity category in entset.pkl")
    group.add_argument('--tar_A2B_path', default=None, type=str, help="the targeted path from the targeted ans for task A(e.g., CVE or disease) to the targeted ans to taskB (e.g., miti or drug)")
    group.add_argument('--atk_obj', default=None, type=str, choices=['untargeted', 'targeted'], 
    help="attack types: partially targeted (untargeted) or fully targeted; for evasion -- untargeted refers to degrade the must-hit set")
    group.add_argument('--max_atk_iter', default=6, type=int, help="maximum perturbation iterations: KGE -> QA reasoning -> peturb embedding space -> perturb KG/train queries")
    
    # sur model config
    group.add_argument('--sur_model', default='box', type=str, help='the surrogate KRL model, vec for GQE, box for Query2box, beta for BetaE, rgcn for RGCN')
    group.add_argument('--sur_hidden_dim', default=None, type=int, help="embedding dimension")
    group.add_argument('--sur_beta_mode', default="(300,2)", type=str, help='(hidden_dim,num_layer) for surrogate BetaE relational projection')
    group.add_argument('--sur_box_mode', default="(none,0.02)", type=str, help='(offset activation,center_reg) for surrogate Query2box, center_reg balances the in_box dist and out_box dist')
    group.add_argument('--sur_cen_layer', type=int, default=2, help='# layers of surrogate center net')
    group.add_argument('--sur_off_layer', type=int, default=2, help='# layers of surrogate Query2Box offset net')
    group.add_argument('--sur_prj_layer', type=int, default=2, help='# layers of surrogate BetaE project net')

    # kg poisoning
    group.add_argument('--kg_rm_ratio', default=0.3, type=float, help='number of facts removed from original KG to generate a surrogate one')
    group.add_argument('--atk_budget', default=100, type=int, help='max number of facts that malicious entities can connect to')
    group.add_argument('--atk_lambda', default=1.0, type=float, help='the lambda used to balance adversarial loss and benign loss')
    group.add_argument('--atk_steps', default=20000, type=int, help='backward optimization iterations to attack (surrogate) reasoning model')
    
    # evasion
    group.add_argument('--eva_ckpt_path', default=None, type=str, help='path to load the model checkpoints in evasion attack')
    group.add_argument('--model_postfix', default=None, type=str, help='model name postfix for loading different models under same dir')
    group.add_argument('--gen_hit_testset', action='store_true', help="(re)generate testset that a well-trained model can hit")
    group.add_argument('--eva_optim_steps', default=25, type=int, help='evasion optimization steps for finding evasion embedding')
    # group.add_argument('--trigger_ratio', default=0.4, type=float, help='trigger evidence ratio comparing with original evidence num (in evasion attack)')
    group.add_argument('--eva_num', default=2, type=int, help='number of additional evidence (together with logic path) added by evasion attack')

def add_cm_group(group):
    group.add_argument('--eva_num_at', default=2, type=int, help='number of evasion scale used by adv-training')
    group.add_argument('--noisy_fact_ratio', default=0.01, type=float, help='number of noisy facts we aim to remove')
    group.add_argument('--robust_kge', action='store_true', help="filtering noisy facts during kge")
    group.add_argument('--adv_train', action='store_true', help="use adv_train as counter measure")
    
def add_ks_group(group):
    group.add_argument('--tar_kg_path',  default=None, type=str, help="KG path for surrogate KRL (dont specify value)")
    group.add_argument('--sur_kg_path',  default=None, type=str, help="KG path for surrogate KRL (dont specify value)")
    group.add_argument('--sur_q_path',  default=None, type=str, help="queries path for surrogate KG (dont specify value)")
    group.add_argument('--tar_model_path', default=None, type=int, help="loading path for the targeted KRL model")

def parse_args():
    parser = argparse.ArgumentParser()
    com_group = parser.add_argument_group(title="Common Parameters")
    krl_group = parser.add_argument_group(title="KRL Parameters")
    atk_group = parser.add_argument_group(title="Attack Parameters")
    cm_group = parser.add_argument_group(title="Countermeasure")
    ks_group = parser.add_argument_group(title="Knowledge Stealing")
    
    add_com_group(com_group)
    add_krl_group(krl_group)
    add_atk_group(atk_group)
    add_cm_group(cm_group)
    add_ks_group(ks_group)

    return parser.parse_args()

# if __name__=='__main__':
#     print(parse_args())