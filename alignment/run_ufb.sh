WANDB_KEY=...
dataset=ultrabin
model=llama
model_name=meta-llama/Llama-3.2-3B-Instruct
batch_size=64
seed=0

# SFT
./sh/sft.sh $WANDB_KEY $dataset $model $model_name $batch_size $seed

# Preference Training
for noise_ratio in 0.0 0.2 0.3 0.4; do
    ./sh/dpo.sh $WANDB_KEY $dataset $model $model_name $noise_ratio $batch_size $seed
    ./sh/cdpo.sh $WANDB_KEY $dataset $model $model_name $noise_ratio $batch_size $seed
    ./sh/rdpo.sh $WANDB_KEY $dataset $model $model_name $noise_ratio $batch_size $seed
    ./sh/ropo.sh $WANDB_KEY $dataset $model $model_name $noise_ratio $batch_size $seed
    ./sh/sympo.sh $WANDB_KEY $dataset $model $model_name $noise_ratio sigmoid $batch_size $seed
    ./sh/sympo.sh $WANDB_KEY $dataset $model $model_name $noise_ratio symmetric_ramp $batch_size $seed
done