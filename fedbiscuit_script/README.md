# Script Description

In this project, we try to work on these projects:

Reward model: 
- Qwen 0.5B
- Gemma 2B
- LLaMA-2-7B (tldr only)

LLM: (finetuned with Alpaca dataset)
- Gemma 2B (mlabonne/Gemmalpaca-2B)
- LLaMA-2-7B (NEU-HAI/Llama-2-7b-alpaca-cleaned)

Dataset:
- Summarization only (TLDR)
- QA only (SHP, with two kinds of non-i.i.d. setting, i.e., $\alpha=0.3$ & $\alpha=1.0$)
- Mixed both datasets 

Baseline for reward model training: 
- FedBis
- FedBiscuit (U=3: 3 LoRA adapters)
- FedBiscuit (U=5: 5 LoRA adapters)
- FedRM (Use a single output, which gives the rating for each summarization)
- FedLoRA (Train the LLM with the better response) 
- FedDPO (Train the LLM using DPO approach)

