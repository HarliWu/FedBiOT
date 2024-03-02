import re
import torch
import os
import random
import transformers
from transformers import GenerationConfig
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.dataloader.reddit_tldr import _download_tldr_cmpr
from federatedscope.llm.misc.fschat import FSChatBot


@torch.no_grad()
def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    init_cfg.freeze()

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to
    fschatbot = FSChatBot(init_cfg)

    _, _, list_data_dict = _download_tldr_cmpr(
        os.path.join(init_cfg.data.root, 'reddit-tldr-comparison'))

    prompt = ("Below is a forum post followed by two summaries. "
              "Pick a more precise and concise one that summarizes the most "
              "important points in the given forum post, without including "
              "unimportant or irrelevant details. "
              "State your choice by strictly following this format: "
              "\"A\" if summary A is better, "
              "\"B\" if summary B is better.\n\n"
              "### Subreddit:\n{subreddit}\n\n### Title:\n{title}\n\n"
              "### Post:\n{post}\n\n### Summary A:{summary_A}\n\n"
              "### Summary B:{summary_B}\n\n"
              "### Your choice:\n")

    prompt = ("Below is a forum post followed by two summaries. "
              "Pick a more precise and concise one that summarizes the most "
              "important points in the given forum post, without including "
              "unimportant or irrelevant details. State your choice with a "
              "single capital letter, i.e., \"A\" if summary A is better, "
              "\"B\" if summary B is better.\n\n"
              "### Subreddit:\n{subreddit}\n\n### Title:\n{title}\n\n"
              "### Post:\n{post}\n\n### Summary A:\n{summary_A}\n\n"
              "### Summary B:\n{summary_B}\n\n### Your choice:")

    # list all choices
    choices = []
    for choice in init_cfg.trainer.choices:
        choices.append(fschatbot.tokenizer(f':{choice}')[2:]['input_ids'][0])

    forward_model = fschatbot.model.merge_and_unload()

    correct, total = 0, 0

    # Print result to a text file
    results_display = open(os.path.join(init_cfg.outdir, 'test_results.txt'),
                           'w')
    testset = tqdm(list_data_dict)
    for sample in testset:
        input_text = prompt.format_map(sample)
        input_text = input_text.replace('### Summary A: ', '### Summary A:\n')
        input_text = input_text.replace('### Summary B: ', '### Summary B:\n')
        # results_display.write(input_text)

        generation_config = GenerationConfig(
            temperature=0.25,
            early_stopping=True,
            num_beams=2,
            no_repeat_ngram_size=2,
            do_sample=True,
        )
        generate_kwargs = dict(
            generation_config=generation_config,
            max_new_tokens=10,
        )
        model_completion = fschatbot.generate(input_text, generate_kwargs)
        input_text_tokens = fschatbot.tokenizer(
            input_text,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = input_text_tokens.input_ids.to('cuda:0')
        attention_mask = input_text_tokens.attention_mask.to('cuda:0')
        outputs = forward_model(input_ids=input_ids,
                                attention_mask=attention_mask)
        logits = outputs.logits
        model_choices = []
        for i in range(logits.shape[0]):
            new_logit = logits[i][-1][choices].unsqueeze(0)
            _, predicted = new_logit.max(1)
            model_choices.append(chr(predicted[0] + ord('A')))

        if chr(sample["choice"] + ord("A")) == model_choices[0]:
            correct += 1
        total += 1

        results_display.write(
            f'Post:\n{sample["post"]}\n\n'
            f'Summary A:\n{sample["summary_A"]}\n\n'
            f'Summary B:\n{sample["summary_B"]}\n\n'
            f'Human selected: {chr(sample["choice"]+ord("A"))}\n\n'
            f'Model-generated summary: {model_completion}\n\n'
            f'Model-choice: {model_choices[0]}\n\n')

        results_display.write('==========================\n\n')
        results_display.flush()

    results_display.write(f'Correct rates: {correct/total*100}% '
                          f'({correct}/{total})')
    results_display.close()


if __name__ == "__main__":
    main()
