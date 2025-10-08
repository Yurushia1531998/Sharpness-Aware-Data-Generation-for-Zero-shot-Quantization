#!/bin/bash
#SBATCH --job-name=test_4000
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --account=ft49
#SBATCH --partition=fit
#SBATCH --qos=fitq
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1


python3 main.py --model_name=resnet18 --bit_w=4 --bit_a=4  --enable_act_train=False \
  --samples=1024   --recon_iter 20000 --distill_batch=128 --recon_batch=32 --warmup_iters 4000 --use_wandb True --start_iters=500 --distill_iter=500  \
  --loss_type=mse_first_order_maxselfOptUpdateSingle_unChange_diverseSim_normGrad \
  --neighbor_factor=1 --bn_factor=1  --grad_factor=1 --diverse_factor=1 --threshold_sim 0 \
  --saved_image_folder="/storage2/tu/cuong/save_folderv2/"   \
  --saved_warmup_image_folder=None \
  --warmup_quantize_model_path=None \
  --val_path="/storage1/tu/imagenet/val" 

   