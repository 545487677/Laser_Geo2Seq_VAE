#!/bin/bash

# GPU configs
MASTER_PORT=10068
data_path='/vepfs/fs_users/guojianz/dp_project/laser/'
save_dir="./save_finetune"

# Model paths
finetune_mol_model='/vepfs/fs_users/guojianz/FuncMG/cz_unimol/Uni-Mol/unimol/weight/mol_pre_all_h_220816.pt'

# Experiment settings
task_num=1
epoch=5000
task_name='0702_plqy'
dropout=0.1
warmup=0.06
update_freq=1
log_path=/vepfs/fs_ckps/guojianz/laser_0729_plqy

if [ "$task_name" == "qm7dft" ] || [ "$task_name" == "qm8dft" ] || [ "$task_name" == "qm9dft" ] || [ "$task_name" == "opv" ] || [ "$task_name" == "opv_rdkit_dft" ] || [ "$task_name" == "0702_plqy" ]; then
    metric="valid_mae"
elif [ "$task_name" == "esol" ] || [ "$task_name" == "freesolv" ] || [ "$task_name" == "lipo" ]; then
    metric="valid_agg_rmse"
else 
    metric="valid_agg_auc"
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

# Predefined parameter values
lr_values=0.00001 # (0.00005 0.00001 0.0001 0.0003 0.0005 0.001 0.005)
batch_size_values=32 # (8 16 32)
layers_values=12 # (4 5 6 7 8 9 10 11 12 13 14 15)

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    n_gpu=$(nvidia-smi -L | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s , 0 $(($n_gpu - 1)))
else
    IFS=',' read -r -a gpu_array <<< "$CUDA_VISIBLE_DEVICES"
    n_gpu=${#gpu_array[@]}
fi

# 打印 GPU 数量和 CUDA_VISIBLE_DEVICES
echo "Number of GPUs to be used: $n_gpu"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Loop over each combination of parameters
for lr in "${lr_values[@]}"; do
    for batch_size in "${batch_size_values[@]}"; do
        for layers in "${layers_values[@]}"; do
            global_batch_size=$(($batch_size * $n_gpu * $update_freq))
            
            echo "epoch_${epoch}_lr_${lr}_bsz_${global_batch_size}_dropout_${dropout}_warmup_${warmup}_gpu_${n_gpu}_layers_${layers}"
            exp=0722_cond_pretrain_epoch_${epoch}_lr_${lr}_bsz_${global_batch_size}_dropout_${dropout}_warmup_${warmup}_gpu_${n_gpu}_layers_${layers}
            mkdir -p ${log_path}/$exp

            torchrun --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ../unimol --train-subset train --valid-subset valid \
                --num-workers 8 --ddp-backend=c10d \
                --task-name $task_name \
                --dict-name dict.txt \
                --task mol_finetune --loss finetune_smooth_mae --arch unimol_ori \
                --num-classes $task_num \
                --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
                --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size \
                --update-freq $update_freq --seed 1 \
                --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
                --log-interval 20 --log-format simple \
                --patience 2000 \
                --finetune-from-model $finetune_mol_model \
                --save-dir "${log_path}/${exp}" \
                --find-unused-parameters \
                --validate-interval 1  \
                --keep-last-epochs 1 \
                --max-atoms 512 \
                --tensorboard-logdir "${log_path}/${exp}/tsb" \
                --encoder-layers ${layers} \
                2>&1 | tee "${log_path}/${exp}/train.log"
        done
    done
done


# --keep-last-epochs 1