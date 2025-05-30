n_gpu=1
layers=11 #9
results_path="./"  # replace to your results path
weight_path='./weights/checkpoint_best.pt'  # replace to your ckpt path
batch_size=32

## FOR VALIDATION: 
# data_path=../
# task_name='0702_exclude_kic'  
# R2 for Column 0: 0.8949863910675049
# R2 for Column 1: 0.8492592573165894
# R2 for Column 2: 0.7453403472900391
# R2 for Column 3: 0.8653849959373474
# MAE for column 0: 0.06146015226840973
# MAE for column 1: 0.2645672559738159
# MAE for column 2: 0.8619914054870605
# MAE for column 3: 0.5529901385307312
## 

# ## FOR INFERENCE:
data_path='./'
task_name='infer_results'
# ## 

task_num=4
dict_name='dict.txt'


MASTER_PORT=10086
torchrun --nproc_per_node=$n_gpu --master_port=$MASTER_PORT ../unimol/infer.py --user-dir ../unimol $data_path --task-name $task_name --valid-subset valid \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task mol_finetune_infer --loss finetune_smooth_mae --arch unimol_ori \
       --num-classes $task_num \
       --dict-name $dict_name \
       --finetune-from-model $weight_path  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --seed 1 \
       --log-interval 50 --log-format simple  --encoder-layers ${layers}

