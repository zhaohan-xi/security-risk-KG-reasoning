# cyber 
CUDA_VISIBLE_DEVICES=0 nohup python -u ../main/roar.py --yaml ../config/cyber_roar.yaml > ../cyber_tar_kgp.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u ../main/roar.py --yaml ../config/cyber_roar.yaml > ../cyber_tar_eva.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u ../main/roar.py --yaml ../config/cyber_roar.yaml > ../cyber_untar_kgp.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u ../main/roar.py --yaml ../config/cyber_roar.yaml > ../cyber_untar_eva.log 2>&1 &


# cyber - debug
CUDA_VISIBLE_DEVICES=0 python ../main/roar.py --yaml ../config/cyber_roar.yaml --debug


# ----------------------------------------------------------------------------

# med 
CUDA_VISIBLE_DEVICES=0 nohup python -u ../main/roar.py --yaml ../config/med_roar.yaml > ../med_tar_kgp.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u ../main/roar.py --yaml ../config/med_roar.yaml > ../med_tar_eva.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u ../main/roar.py --yaml ../config/med_roar.yaml > ../med_untar_kgp.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u ../main/roar.py --yaml ../config/med_roar.yaml > ../med_untar_eva.log 2>&1 &


# med - debug
CUDA_VISIBLE_DEVICES=3 python ../main/roar.py --yaml ../config/med_roar.yaml --debug




pid         time            setting
1279153     tar kgp for revision 
1279320     untar kgp for revision 

expt
                        cyber           med
(1) Basic               02              03
(2) Diff Surrogate      01              03
(3) Diff Trigger        RUN-01          \
(4) Cop                 RUN-02          03