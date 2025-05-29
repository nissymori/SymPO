WANDB_KEY=...
dataset=ufb
model=llama
model_name=meta-llama/Llama-3.2-3B-Instruct
examples=20000
seed=0

# SFT
./sh/sft.sh $WANDB_KEY $dataset $model $model_name $examples $seed

# Preference Training
for noise_ratio in 0.0 0.2 0.3 0.4; do
    ./sh/dpo.sh $WANDB_KEY $dataset $model $model_name $noise_ratio $examples $seed
    ./sh/cdpo.sh $WANDB_KEY $dataset $model $model_name $noise_ratio $examples $seed
    ./sh/rdpo.sh $WANDB_KEY $dataset $model $model_name $noise_ratio $examples $seed
    ./sh/ropo.sh $WANDB_KEY $dataset $model $model_name $noise_ratio $examples $seed
    ./sh/sympo.sh $WANDB_KEY $dataset $model $model_name $noise_ratio sigmoid $examples $seed
    ./sh/sympo.sh $WANDB_KEY $dataset $model $model_name $noise_ratio symmetric_ramp $examples $seed
done