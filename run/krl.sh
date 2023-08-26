# cyber - Q2B - AB task
CUDA_VISIBLE_DEVICES=0 nohup python -u ../main/krl_AB.py --yaml ../config/cyber_krl.yaml > ../krl_cyber.log 2>&1 &

# med - GQE - AB task
CUDA_VISIBLE_DEVICES=3 nohup python -u ../main/krl_AB.py --yaml ../config/med_krl.yaml > ../krl_med_vec.log 2>&1 &


# --------------------------------------------------------------------------------
# cyber - general
CUDA_VISIBLE_DEVICES=0 nohup python -u ../main/krl_general.py --yaml ../config/wiki_krl.yaml > ../krl_general.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u ../main/krl_general.py --yaml ../config/wn_krl.yaml > ../krl_general.log 2>&1 &


pid        task
1013749    wikikg, train with betae