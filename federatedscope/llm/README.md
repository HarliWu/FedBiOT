# FederatedScope-LLM

FederatedScope-LLM (FS-LLM) is an efficient package for federated large language model. We provide a hands-on tutorial here, while for more detailed tutorial, please refer to [TO-BE-RELEASED]().

## Quick Start

Let’s start with finetuning GPT-2 on [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) to familiarize you with FS-LLM.

### Step 1. Installation

The installation of FS-LLM is similar to minimal FS, except that it requires **Pytorch>=1.13.0** (we recommend version 2.0.X) because of the [PEFT](https://github.com/huggingface/peft) dependency:

```bash
# Create virtual environments with conda
conda create -n fs-llm python=3.9
conda activate fs-llm

# Install Pytorch>=1.13.0 (e.g., Pytorch==2.0.0)
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install FS-LLM with editable mode
pip install -e .[llm]
```

Now, you have successfully installed the FS-LLM.

### Step 2. Run with exmaple config

Now, we can fine-tune a GPT2 on Alpaca with FedAvg.

```bash
python federatedscope/main.py --cfg federatedscope/llm/baseline/testcase.yaml
```

For more details about customized configurations, see **Advanced**.

## Advanced

### Start with built-in functions

You can easily run through a customized `yaml` file. Here we only introduce the configuration related to FS-LLM, other configurations please refer to [Configurations](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/configs/README.md). For more examples, please refer to `federatedscope/llm/baseline`.

```yaml
# For this configuration, you might need a GPU with at least 32GB of video memory to run.

# Whether to use GPU
use_gpu: True

# Deciding which GPU to use
device: 0

# Early stop steps, set `0` to disable
early_stop:
  patience: 0

# Federate learning related options
federate:
  # `standalone` or `distributed`
  mode: standalone
  # Number of communication round
  total_round_num: 500
  # Saving path for ckpt
  save_to: "llama_rosetta_9_fed.ckpt"
  # Number of dataset being split
  client_num: 9
  # Enable for saving memory, all workers share the same model instance
  share_local_model: True

# Dataset related options
data:
  # Root directory where the data stored
  root: data/
  # Dataset name
  type: 'rosetta_alpaca@llm'
  # Train/val/test splits
  splits: [0.89,0.1,0.01]
  # Use meta inforamtion to split `rosetta_alpaca`
  splitter: 'meta'

# LLM related options
llm:
  # Max token length for model input (training)
  tok_len: 650
  # ChatBot related options
  chat:
    # Max token length for model input (inference)
    max_len: 1000
    # Max number of history texts
    max_history_len: 10
  # Path for store model cache, default in `~/.cache/`
  cache:
    model: ''
  # PEFT related options
  adapter:
    # Set ture to enable PEFT finetuning
    use: True
    # Args for PEFT finetuning
    args: [ { 'adapter_package': 'peft', 'adapter_method': 'lora', 'r': 8, 'lora_alpha': 32, 'lora_dropout': 0.1 } ]

# DataLoader related options
dataloader:
  # Batch size for iter loader
  batch_size: 1

# Model related options
model:
  # Model type (format: {MODEL_REPO}@huggingface_llm)
  type: 'decapoda-research/llama-7b-hf@huggingface_llm'

# Train related options
train:
  # Number of local update steps
  local_update_steps: 30
  # `batch` or `epoch` for local_update_steps
  batch_or_epoch: batch
  # Optimizer related options
  optimizer:
    # Learning rate
    lr: 0.003
    # Weight decay
    weight_decay: 0.0
  # Set ture to enable `model.half()`
  is_enable_half: True

# Trainer related options
trainer:
  # Trainer type
  type: llmtrainer

# Evaluation related options
eval:
  # Frequency of evaluation
  freq: 50
  # Evaluation metrics
  metrics: ['loss']
  # Set key to track best model
  best_res_update_round_wise_key: val_loss
```

### Fine-tuning Datasets

In general, we use instruction SFT following [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) team. And in standalone mode, all dataset can be split into several clients with spesific `splitter` (i.e., `lda`, `meta`, `iid`) and `federate.num_client`. 

#### Built-in Data

| data.type             | Source                                                | Note                                                |
| --------------------- | ----------------------------------------------------- | --------------------------------------------------- |
| `alpaca@llm`          | [Link](https://github.com/tatsu-lab/stanford_alpaca)  | `IIDSplitter`                                       |
| `alpaca_cleaned@llm`  | [Link](https://github.com/gururise/AlpacaDataCleaned) | `IIDSplitter`                                       |
| `dolly-15k@llm`       | [Link](https://github.com/databrickslabs/dolly)       | `LDASplitter` or `MetaSplitter` split to 8 clients. |
| `gsm8k@llm`           | [Link](https://github.com/openai/grade-school-math)   | `IIDSplitter`                                       |
| `rosetta_alpaca@llm`  | [Link](https://github.com/sahil280114/codealpaca)     | `LDASplitter` or `MetaSplitter` split to 9 clients. |
| `code_search_net@llm` | [Link](https://github.com/github/CodeSearchNet)       | `LDASplitter` or `MetaSplitter` split to 6 clients. |

#### Self-maintained Data

| data.type                 | Note                                                         |
| ------------------------- | ------------------------------------------------------------ |
| `YOU_DATA_NAME.json@llm`  | Format: `[{'instruction': ..., 'input': ..., 'output':...}]`, default key: `instruction`, `input`, `output`, `category` |
| `YOU_DATA_NAME.jsonl@llm` | Format of each line: `{'instruction': ..., 'input': ..., 'output':...}`, default key: `instruction`, `input`, `output`, `category` |

#### Evaluation tools

We evaluate model domain capability of fine-tuned models with easy-to-use evaluation tools.

```bash
FederatedScope
├── federatedscope
│   ├── llm
│   │   ├── eval
│   │   │   ├── eval_for_code
│   │   │   ├── eval_for_gsm8k
│   │   │   ├── eval_for_helm
│   │   │   ├── eval_for_mmlu
...
```

How to use: 

For example, to evaluate the model fine-tuned with `python federatedscope/main.py --cfg sft_gsm8k.yaml`, you can run `python federatedscope/llm/eval/eval_for_gsm8k/eval.py --cfg sft_gsm8k.yaml` in the `eval_for_gsm8k` directory. For other usages, please refer to the `README.md` file in each subdirectory.

### Agorithms

#### Parameter-Efficient Fine-Tuning

With the help of parameter-efficient fine-tuning methods, federally fine-tuning a large model requires passing only a very small percentage of model parameters (adapters), making it possible for the client enable efficient adaptation of pre-trained language models to various downstream applications. We adopt [PEFT](https://github.com/huggingface/peft) for fine-tuning LLMs, and more methods are coming soon!

| Methods       | Source                                                       | Example for `llm.adapter.args`                               |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| LoRA          | [Link](https://arxiv.org/abs/2106.09685)                     | `[ { 'adapter_package': 'peft', 'adapter_method': 'lora', 'r': 8, 'lora_alpha': 32, 'lora_dropout': 0.1 } ]` |
| Prefix Tuning | [Link](https://aclanthology.org/2021.acl-long.353/), [Link](https://arxiv.org/pdf/2110.07602.pdf) | `[{'adapter_package': 'peft', 'adapter_method': 'prefix', 'prefix_projection': False, 'num_virtual_tokens': 20}]` |
| P-Tuning      | [Link](https://arxiv.org/abs/2103.10385)                     | `[{'adapter_package': 'peft', 'adapter_method': 'p-tuning', 'encoder_reparameterization_type': 'MLP', 'encoder_dropout': 0.1, 'num_virtual_tokens': 20}]` |
| Prompt Tuning | [Link](https://arxiv.org/abs/2104.08691)                     | `[{'adapter_package': 'peft', 'adapter_method': 'prompt', 'prompt_tuning_init': 'RANDOM', 'num_virtual_tokens': 20}]` |

#### Federate fine-tune closed-source LLMs 

We support federated fine-tuning not only for open-source LLMs, but also for closed-source LLMs. In this scenario, clients can fine-tune LLMs without fully accessing the model, where models and data are both considered as privacy.

| Methods        | Source                                   | How to enable                 |
| -------------- | ---------------------------------------- | ----------------------------- |
| Offsite-Tuning | [Link](https://arxiv.org/abs/2302.04870) | `llm.offsute_tuning.use=True` |

#### Federate fine-tune with multi-card

To make the federate fine-tuning efficient, we adopt a series of multi-card acceleration operators.

| Methods               | Source                                                       | How to use                       | Note                                          |
| --------------------- | ------------------------------------------------------------ | -------------------------------- | --------------------------------------------- |
| torch.nn.DataParallel | [Link](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) | `cfg.train.data_para_dids=[0,1]` | -                                             |
| DeepSpeed             | [Link](https://github.com/microsoft/DeepSpeed)               | Coming soon                      | Use `nvcc - V` to make sure `CUDA` installed. |
| HF Accelerate         | [Link](https://huggingface.co/docs/accelerate/index)     | See below | - | 

## FAQ

- `WARNING: Skip the batch due to the loss is NaN, it may be caused by exceeding the precision or invalid labels.`
  - Possible reason 1: This is because `llm.tok_len` limits the input length, causing the label to be empty, which automatically skips that data. Setting a larger `llm.tok_len` can avoid this.
  - Possible reason 2: Due to the enabling of `train.is_enable_half`, numerical overflow may occur. This usually happens when setting the `optimizer.type` to `Adam`, since the default `eps` is `1e-8` but `fp16` requires at least `1e-5`.
- `ValueError: Tokenizer class LLaMATokenizer does not exist or is not currently imported. `
  - This is a problem with `transformers`, you can fix it in your local file. Replace `LLaMATokenizer` with `LlamaTokenizer` in `PATH_TO_DATA_ROOT/MODEL_REPO/snapshots/..../tokenizer_config.json`
- `OutOfMemoryError: CUDA out of memory.`
  - Torch's garbage collection mechanism may not be timely resulting in OOM, please set `cfg.eval.count_flops` to `False`.

## New: Support of HF Accelerate 

**Note: This is the quick instrcution for *standalone* FS config.** The distributed FS using HF accelerate is coming soon. 

To generate the config file, you should run 

```bash
accelerate config --config_file accelerator_config.yaml
```

and follow the intruction prompt. Finally, a file `accelerator_config.yaml` is created with the following format: 

```yaml
# Note: You can prepare this file manually 
# This is an example that supports FederatedScope Standalone setting with multiple GPUs for fine-tuning (with the help of "model sharding") 

compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: 2,3,4
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

To enable HF accelerate in the fine-tuning, run the following command: 

```bash
accelerate launch --config_file accelerator_config.yaml federatedscope/main.py --cfg [fs_config_file].yaml llm.accelerator.use True
```

For the evaluation on GSM8k, HumanEval, and HELM, run the following command 

```bash
CUDA_VISIBLE_DEVICES=1,2 python federatedscope/llm/eval/eval_for_[task]/eval.py --cfg [fs_config_file].yaml
```
