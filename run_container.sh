export WANDB_API_KEY=73851e91ebbedd73c408f1cf2f4948a451238384
export WANDB_DIR=wandb/$SLURM_JOBID
export WANDB_CONFIG_DIR=wandb/$SLURM_JOBID
export WANDB_CACHE_DIR=wandb/$SLURM_JOBID
export WANDB_START_METHOD="thread"
wandb login

torchrun --nnodes=1 --nproc_per_node=1 train.py \
         --data_path "/gpfs/work5/0/jhstue005/JHS_data/CityScapes"
