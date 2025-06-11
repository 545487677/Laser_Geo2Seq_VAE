n_gpu=1
layers=12 #9
results_path="./"  # replace to your results path
weight_path='./weights_plqy/checkpoint_best.pt'  # replace to your ckpt path
batch_size=32

## FOR VALIDATION: 
# data_path=../
# task_name='0702_plqy'  
# # R2 for Column 0: 0.8255677223205566
# # MAE for column 0: 0.06858295202255249
## 

# ## FOR INFERENCE:
data_path='./'
task_name='infer_plqy'
# ## 

task_num=1
dict_name='dict.txt'


MASTER_PORT=10086
torchrun --nproc_per_node=$n_gpu --master_port=$MASTER_PORT ../unimol-pre/infer.py --user-dir ../unimol $data_path --task-name $task_name --valid-subset valid \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task mol_finetune_infer --loss finetune_smooth_mae --arch unimol_ori \
       --num-classes $task_num \
       --dict-name $dict_name \
       --finetune-from-model $weight_path  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --seed 1 \
       --log-interval 50 --log-format simple  --encoder-layers ${layers}

