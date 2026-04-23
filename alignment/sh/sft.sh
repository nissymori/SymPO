wandb_key=$1
dataset=$2
model=$3
model_name=$4
batch_size=$5
seed=$6 
'''
WANDB_API_KEY=$wandb_key accelerate launch --config_file accelerate_config/fsdp_8gpu.yaml --main_process_port 29500 launch.py \
    loss=sft \
    model=$model \
    datasets=[$dataset] \
    exp_name=sft/seed_$seed \
    seed=$seed \
    ++cache_dir=.cache/data/$dataset/$model_name/ \
    ++model.name_or_path=$model_name \
    ++lr=5e-6 \
    ++loss.beta=0.1 \
    model.batch_size=$batch_size \
    model.eval_batch_size=$batch_size \
    model.max_length=512 \
    model.max_prompt_length=256 \
'''
# sample
python -m train.sample \
    .cache/data/$dataset/$model_name/sft/seed_$seed/FINAL \
    --gpu_count 2 \
    --output_file .cache/outputs/$dataset/$model_name/sft/seed_$seed/completions.json \
    --datasets $dataset \
    --mode custom_dataset \
    --max_tokens 512 \
    --max_prompt_length 256 \
    --seeds 10 20 30 40 50 60 70 80 90 100