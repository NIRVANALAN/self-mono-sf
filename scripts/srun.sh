#!/bin/bash
srun -p NTU --mpi=pmi2 --gres=gpu:$1 -n1 --ntasks-per-node=1 --job-name=$2 --kill-on-bad-exit=1 -w SG-IDC1-10-51-1-45 \
bash ./eval_monosf_finetune_kitti_test.sh
#bash ./eval_monodepth_selfsup_kitti_train.sh \
#bash ./eval_monosf_selfsup_kitti_test.sh \