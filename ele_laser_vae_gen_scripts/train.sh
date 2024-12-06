export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
MASTER_PORT=10086

data_path='/vepfs/fs_projects/FunMG/laser/laser_geo2seq_gen/'  # replace to your data path
task_name='ele_mol'  # data folder name  # choice: qm9, oled, opv
lr=$1
local_batch_size=$2
update_freq=$3
# step=$4
wd=1e-12
warmup_steps=10000
max_steps=1000000
dropout=0.1
seed=42
clip_norm=5.0
decoder=TFM #GRU,TFM, SE3, EGNN
echo $decoder
encoder_weight_path='/vepfs/fs_users/guojianz/dp_project/laser_vae_gen/pretrain_weight/mol_pre_all_h_220816.pt'


pytorch_version=`python -c 'import torch; print(torch.__version__)'`
if [[ $pytorch_version = 1* ]]; then
       echo "PyTorch 1 found. Using torch.distributed.launch..."
       torchcmd="python -m torch.distributed.launch"
else
       echo "PyTorch 2 found. Using torchrun..."
       torchcmd="torchrun"
fi

# 获取 GPU 数量
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    n_gpu=$(nvidia-smi -L | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s , 0 $(($n_gpu - 1)))
else
    IFS=',' read -r -a gpu_array <<< "$CUDA_VISIBLE_DEVICES"
    n_gpu=${#gpu_array[@]}
fi
echo "Using $n_gpu GPUs"

global_batch_size=$(($local_batch_size * $n_gpu * $update_freq))
log_dir="/vepfs/fs_ckps/guojianz/Laser_Design/vae_v2_${task_name}_${global_batch_size}_${lr}_${dropout}_${seed}_${wd}"
rm -rf $log_dir
mkdir -p $log_dir

$torchcmd --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name --user-dir ../unimol --train-subset valid --valid-subset valid \
        --num-workers 8 --ddp-backend=c10d \
        --task vae --loss finetune_vae_loss --arch vae  \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8  --weight-decay $wd --clip-norm $clip_norm \
        --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps --max-update $max_steps \
        --batch-size $local_batch_size --pooler-dropout $dropout --update-freq $update_freq --seed $seed \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --tensorboard-logdir ${log_dir}/tsb \
        --log-interval 500 --log-format simple \
        --finetune-encoder-model $encoder_weight_path \
        --validate-interval 20 --patience 100 \
        --save-interval 20 \
        --encoder unimol-laser \
        --decoder TFM \
        --max-atoms 350 \
        --find-unused-parameters \
        --all-gather-list-size 16384000 \
        --save-dir $log_dir 2>&1 | tee ${log_dir}/train.log
