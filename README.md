# On Symmetric Losses for Robust Policy Optimization with Noisy Preferences
This is the official code for the paper: On Symmetric Losses for Robust Policy Optimization with Noisy Preferences.

## Experiment with MNIST Preference Dataset
The code is at [mnistpref/](https://github.com/nissymori/SymPO/tree/main/mnistpref).
For the reproduction of the results in Section 5.1 of the paper, please refer to [mnristpref/run_mnist.sh](https://github.com/nissymori/SymPO/tree/main/mnistpref/run_mnist.sh)

## Language Model Alignment with Noisy Preferences
The code is at [alignment/](https://github.com/nissymori/SymPO/tree/main/alignment).
Below are some examples of the training scripts for SFT, DPO, and SymPO with Anthropic HH dataset and Llama 3.2 3B-Instruct model.
For reproduction of the training results in Section 5.2 of the paper, please refer to
- [alignment/run_hh.sh](https://github.com/nissymori/SymPO/tree/main/alignment/run_hh.sh): for Anthropic HH dataset
- [alignment/run_ufb.sh](https://github.com/nissymori/SymPO/tree/main/alignment/run_ufb.sh): for UFB dataset

### Training
At alignment, we use the following scripts to train the models.
- SFT
```
accelerate launch --config_file accelerate_config/fsdp_4gpu.yaml --main_process_port 29500 launch.py wandb.enabled=false loss=sft model=llama datasets=[hh] exp_name=stf ++cache_dir=.cache/data/models ++model.name_or_path=meta-llama/Llama-3.2-3B-Instruct
```

- DPO
```
accelerate launch --config_file accelerate_config/fsdp_4gpu.yaml --main_process_port 29500 launch.py wandb.enabled=false noise_ratio=0.1 loss=dpo model=llama datasets=[hh] exp_name=dpo ++cache_dir=.cache/data/models ++model.name_or_path=meta-llama/Llama-3.2-3B-Instruct ++model.load_from=.cache/data/models/stf/FINAL
```

- SymPO
```
accelerate launch --config_file accelerate_config/fsdp_4gpu.yaml --main_process_port 29500 launch.py wandb.enabled=false noise_ratio=0.1 loss=sympo model=llama datasets=[hh] exp_name=sympo ++cache_dir=.cache/data/models ++model.name_or_path=meta-llama/Llama-3.2-3B-Instruct ++model.load_from=.cache/data/models/stf/FINAL
```

### Sample (generate a completion given a prompt)
```
HF_TOKEN=... python -m train.sample .cache/data/models/dpo/FINAL --gpu_count 2 --output_file .cache/outputs/dpo/completions.json --datasets hh --mode custom_dataset --max_tokens 512 --max_prompt_length 256 --seeds 10 20 30
```

### GPT-4 Evaluation
```
OPENAI_API_KEY=... python -m train.evaluate --input_file .cache/outputs/dpo/completions.json --task default --evaluator gpt-4o-mini
```
