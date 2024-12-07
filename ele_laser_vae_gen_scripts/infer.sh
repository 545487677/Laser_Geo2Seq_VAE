data_path='/vepfs/fs_projects/FunMG/laser/laser_geo2seq_gen/'  # replace to your data path

weight_path='/vepfs/fs_ckps/guojianz/Laser_Design/vae_v2_ele_mol_256_1e-4_0.1_42_1e-12/checkpoint_best.pt'
seed=1
scale=1.5
results_path='./infer_res_'${seed}_${scale} # replace to your results path

batch_size=64
task_name='ele_mol' # data folder name 
echo $task_name
mkdir -p $results_path

# pytorch_version=`python -c 'import torch; print(torch.__version__)'`
# if [[ $pytorch_version = 1* ]]; then
#        echo "PyTorch 1 found. Using torch.distributed.launch..."
#        torchcmd="python -m torch.distributed.launch"
# else
#        echo "PyTorch 2 found. Using torchrun..."
#        torchcmd="torchrun"
# fi

# # 获取 GPU 数量
# if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
#     n_gpu=$(nvidia-smi -L | wc -l)
#     export CUDA_VISIBLE_DEVICES=$(seq -s , 0 $(($n_gpu - 1)))
# else
#     IFS=',' read -r -a gpu_array <<< "$CUDA_VISIBLE_DEVICES"
#     n_gpu=${#gpu_array[@]}
# fi
# echo "Using $n_gpu GPUs"

## VAE
python  /vepfs/fs_users/guojianz/dp_project/laser_vae_gen/unimol/infer.py --user-dir ../unimol $data_path --task-name $task_name \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task vae_infer --loss finetune_vae_infer --arch vae \
       --path $weight_path  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 50 --log-format simple \
       --encoder unimol-laser \
       --decoder TFM \
       --sample_size 100000 \
       --max_len 350 \
       --infer_method GREEDY \
       --seed ${seed} \
       --scale ${scale} \
       --infer_mode gen \
       2>&1 | tee ${results_path}/log.txt

# 100000