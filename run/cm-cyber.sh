# Q2B - collaborative attack - targeted
CUDA_VISIBLE_DEVICES=3 nohup python -u ../main/cm_cyber.py --cuda --do_train --do_test \
  --data_path ../data/cyberkg/cyberkg_s --use_case 'cyberkg_s' --eval_attack -lr 0.001 \
  --kge_train_steps 80000 --qa_train_steps 9000 --model box --valid_steps 3000 \
  --attack_vector_kg --attack_vector_tst --atk_entnum 0 --atk_budget 25 --sur_atk_num 150 \
  --tar_evi 'Android' --tar_ans 'CVE-2021-0471' --tar_evi_cate 'product' \
  --atk_steps 6000 --atk_obj 'targeted' --max_pturb_it 4 --eva_optim_steps 25 \
  --eva_num 2 --eva_num_at 2 --robust_kge --adv_train > ../tar_cyber.log 2>&1 &


# Q2B - collaborative attack - untargeted
CUDA_VISIBLE_DEVICES=3 nohup python -u ../main/cm_cyber.py --cuda --do_train --do_test \
  --data_path ../data/cyberkg/cyberkg_s --use_case 'cyberkg_s' --eval_attack -lr 0.001 \
  --kge_train_steps 80000 --qa_train_steps 9000 --model box --valid_steps 3000 \
  --attack_vector_kg --attack_vector_tst --atk_entnum 0 --atk_budget 25 --sur_atk_num 150 \
  --tar_evi 'Android' --tar_ans 'CVE-2021-0471' --tar_evi_cate 'product' \
  --atk_steps 8000 --atk_obj 'untargeted' --max_pturb_it 4 --eva_optim_steps 25 \
  --eva_num 3 --eva_num_at 2 --robust_kge --adv_train > ../untar_cyber.log 2>&1 &

#---------------------------------------------------------------------------------------------
# tmp
CUDA_VISIBLE_DEVICES=2 python ../main/cm_cyber.py --cuda --do_train \
  --data_path ../data/cyberkg/cyberkg_s --use_case 'cyberkg_s' --eval_attack -lr 0.001 \
  --kge_train_steps 3 --qa_train_steps 3 --model box --valid_steps 3000 \
  --attack_vector_kg --attack_vector_tst --atk_entnum 0 --atk_budget 25 --sur_atk_num 25 \
  --tar_evi 'Android' --tar_ans 'CVE-2021-0471' --tar_evi_cate 'product' \
  --atk_steps 3 --atk_obj 'targeted' --max_pturb_it 2 --eva_optim_steps 3 --eva_num 2 \
  --robust_kge --adv_train --verbose