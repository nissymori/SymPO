# Code for LLM Experiments

## Usage
### Training
- SFT
```
accelerate launch --config_file accelerate_config/fsdp_4gpu.yaml --main_process_port 29500 launch.py wandb.enabled=false n_epochs=1 loss=sft model=pythia datasets=[hh] exp_name=stf ++cache_dir=.cache/data/models ++model.name_or_path=EleutherAI/pythia-410m ++lr=5e-6 ++loss.beta=0.1 model.batch_size=32 model.eval_batch_size=32 model.max_length=512 model.max_prompt_length=256
```

- DPO
```
accelerate launch --config_file accelerate_config/fsdp_4gpu.yaml --main_process_port 29500 launch.py wandb.enabled=false n_epochs=1 loss=dpo model=pythia datasets=[hh] exp_name=dpo seed=1 ++cache_dir=.cache/data/models ++model.name_or_path=EleutherAI/pythia-410m ++lr=5e-6 ++loss.beta=0.1 model.batch_size=32 model.eval_batch_size=32 model.max_length=512 model.max_prompt_length=256 ++model.load_from=.cache/data/models/stf/FINAL
```

- SymPO
```
accelerate launch --config_file accelerate_config/fsdp_4gpu.yaml --main_process_port 29500 launch.py wandb.enabled=false n_epochs=1 loss=sympo model=pythia datasets=[hh] exp_name=sympo seed=1 ++cache_dir=.cache/data/models ++model.name_or_path=EleutherAI/pythia-410m ++lr=5e-6 ++loss.beta=0.1 model.batch_size=32 model.eval_batch_size=32 model.max_length=512 model.max_prompt_length=256 ++model.load_from=.cache/data/models/stf/FINAL
```

### Sample (generate a completion given a prompt)
```
HF_TOKEN=... python -m train.sample .cache/data/models/dpo/FINAL --gpu_count 2 --output_file .cache/outputs/dpo/completions.json --datasets hh --mode custom_dataset --max_tokens 512 --max_prompt_length 256 --seeds 10 20 30
```

### GPT-4 Evaluation
```
OPENAI_API_KEY=... python -m train.evaluate --input_file .cache/outputs/dpo/completions.json --task default --evaluator gpt-4o-mini
```
