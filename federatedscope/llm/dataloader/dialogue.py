import os
import json
import copy
import pickle

from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonls, load_jsonl
from federatedscope.llm.dataset.llm_dataset import (
    DefaultToken,
    LLMDataset,
    LLMComparisonDataset,
)

# <<SYS>>
# You're are a helpful Assistant, and you only response to the "Assistant"
# Remember, maintain a natural tone. Be precise, concise, and casual.
# Keep it short\n
# <</SYS>>
# {conversation_history}\n\n
# [INST]
# User:{user_message}
# [/INST]\n
# Assistant:

DIALOG_PROMPT_DICT = {
    "dialog": (
        # "You are a helpful Assistant. "
        # "Read the conversation between USER and ASSISTANT below, "
        # "and write a precise, concise, and casual response "
        # "as an ASSISTANT.\n"
        "{conversation_history}"
        "\nAssistant:"),
    "dialog_cmp": ("Below is a query followed by two responses. Pick a "
                   "helpful response that is precise, concise, and casual. "
                   "State your choice with a single capital letter, "
                   'i.e., "A" if RESPONSE A is better, '
                   '"B" if RESPONSE B is better.\n\n'
                   "### QUERY:\n"
                   "{conversation_history}"
                   "### RESPONSE A: {output_A}\n"
                   "### RESPONSE B: {output_B}\n"
                   "### YOUR CHOICE:"),
}
