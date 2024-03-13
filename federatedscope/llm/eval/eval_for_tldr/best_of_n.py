import torch
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import argparse
import numpy as np

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.llm.dataloader.dataloader import load_jsonl, \
    LLMDataCollator, get_tokenizer
from federatedscope.llm.misc.fschat import FSChatBot
from federatedscope.llm.dataset.llm_dataset import DefaultToken, \
    LLMDataset


def _generate_best_of_n_dataset(gen_cfg, n=16):
    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to
    fschatbot = FSChatBot(gen_cfg)

    # create the dataset
    fp = os.path.join(gen_cfg.data.root, 'reddit-tldr_test.jsonl')
    if not os.path.exists(fp):
        download_url(
            'https://openaipublic.blob.core.windows.net/'
            'summarize-from-feedback/datasets/'
            'tldr_3_filtered/test.jsonl', gen_cfg.data.root)
        os.rename(os.path.join(gen_cfg.data.root, 'test.jsonl'), fp)

    list_data_dict = load_jsonl(fp,
                                subreddit='subreddit',
                                title='title',
                                post='post',
                                summary='summary')

    prompt = ("Below is a forum post. Write a precise and concise summary "
              "that includes the most important points of the post.\n\n"
              "### Subreddit:\n{subreddit}\n\n### Title:\n{title}\n\n"
              "### Post:\n{post}\n\n### TL; DR:")

    results_display = os.path.join(gen_cfg.outdir,
                                   f'{fschatbot.curpfx}_summarization.txt')
    results_display = open(results_display, 'w')

    for sample in tqdm(list_data_dict):
        input_text = prompt.format_map(sample)
        generate_kwargs = dict(
            top_p=1.0,
            temperature=1.0,
            do_sample=True,
            max_length=gen_cfg.llm.chat.max_len,
            num_return_sequences=n,
        )
        model_completions = fschatbot.generate(input_text, generate_kwargs)

        results_display.write(f'Post:\n{sample["post"]}\n\n'
                              f'Human summary:\n{sample["summary"]}\n\n')
        summaries = []
        for i, completion in enumerate(model_completions):
            results_display.write(
                f'Model-generated summary {i}:\n{completion}\n\n')
            summaries.append(completion)

        sample['summaries'] = summaries

        results_display.write('==========================\n\n')
        results_display.flush()

    return list_data_dict


def best_of_n_dataset(init_cfg, gen_cfg=None, n=16):
    gen_fp = os.path.join(init_cfg.data.root, 'reddit-tldr-comparison',
                          f'reddit-tldr_test_{n}-gen.json')
    if os.path.exists(gen_fp):
        # load the dataset
        list_data_dict = json.load(gen_fp)
    else:
        # create the dataset
        list_data_dict = _generate_best_of_n_dataset(gen_cfg, n)
        json.dump(list_data_dict, gen_fp)
        exit(-1)
    return list_data_dict


def cal_acc(logits, labels, choices):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value)
    for idx, choice in enumerate(choices):
        new_labels[shift_labels == choice] = idx

    new_labels = new_labels.view(-1)
    new_logits = shift_logits[..., choices].view(-1, len(choices))
    new_logits = new_logits[(new_labels != DefaultToken.IGNORE_INDEX.value), :]
    new_labels = new_labels[(new_labels != DefaultToken.IGNORE_INDEX.value)]
    _, predicted = new_logits.max(1)

    return new_labels, predicted, predicted.eq(new_labels).sum().item()


def best_of_n(model, dataset, tokenizer, n=16):
    prompt = ("Below is a forum post followed by two summaries. "
              "Pick a more precise and concise one that summarizes the most "
              "important points in the given forum post, without including "
              "unimportant or irrelevant details. State your choice with a "
              "single capital letter, i.e., \"A\" if SUMMARY A is better, "
              "\"B\" if SUMMARY B is better.\n\n"
              "### SUBREDDIT: r/{subreddit}\n"
              "### TITLE: {title}\n"
              "### POST: {post}\n"
              "### SUMMARY A:{summary_A}\n"
              "### SUMMARY B:{summary_B}\n"
              "### YOUR CHOICE:")

    choices = [tokenizer(f'{c}')['input_ids'][-1] for c in ['A', 'B']]
    # Correct idx is 0 means no changed, 1 means changed to the new index
    last_better_idx = np.array([0] * len(dataset))
    for i in range(1, n):
        for better_idx, sample in zip(last_better_idx, dataset):
            sample['summary_A'] = sample['summaries'][better_idx]
            sample['summary_B'] = sample['summaries'][i]
            sample['choice'] = " A"

        test_dataset = LLMDataset(dataset,
                                  tokenizer,
                                  prompt_input=prompt,
                                  prompt_no_input=prompt,
                                  output_tag='choice')
        dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=n,
            shuffle=False,
            collate_fn=LLMDataCollator(tokenizer=tokenizer))

        predicted_indices = []
        for data_batch in tqdm(dataloader):
            input_ids = data_batch["input_ids"].to('cuda:0')
            labels = data_batch["labels"].to('cuda:0')
            attention_mask = data_batch["attention_mask"].to('cuda:0')
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted, _ = cal_acc(outputs.logits, labels, choices)
            predicted_indices += predicted.tolist()

        assert len(predicted_indices) == len(last_better_idx)
        predicted_indices = np.array(predicted_indices)
        last_better_idx[predicted_indices == 1] = i

    # print the final results
    return last_better_idx


@torch.no_grad()
def main():
    # Create new parser for generation
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-cfg-file',
                        dest='gen_cfg_file',
                        help='Generation config file path',
                        required=False,
                        default=None,
                        type=str)
    gen_args, extra = parser.parse_known_args()

    # Load the reward choice config
    init_cfg = global_cfg.clone()
    args = parse_args(extra)

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # Load the generation config
    gen_cfg = init_cfg.clone()
    if gen_args.gen_cfg_file:
        gen_cfg.merge_from_file(gen_args.gen_cfg_file)
    gen_cfg.freeze(save=False)

    init_cfg.freeze()

    # best_of_n dataset
    dataset = best_of_n_dataset(init_cfg, gen_cfg, n=16)

    # get model and tokenizer
    model_name, _ = init_cfg.model.type.split('@')
    model = get_llm(init_cfg, device_map='auto')
    tokenizer, _ = get_tokenizer(model_name, init_cfg.data.root,
                                 init_cfg.llm.tok_len)

    # get the best-of-n results and display them
    results = best_of_n(model, dataset, tokenizer, n=16)
    results_display = open(os.path.join(init_cfg.outdir, 'test_results.txt'),
                           'w')
    for best_idx, sample in zip(results, dataset):
        results_display.write(f'Subreddit: r/{sample["subreddit"]}\n\n'
                              f'Title:\n{sample["title"]}\n\n'
                              f'Post:\n{sample["post"]}\n\n'
                              f'Best generated summary [[{best_idx}]]:\n'
                              f'{sample["summaries"][best_idx]}\n\n')
        results_display.write('==========================\n\n')
        results_display.flush()


if __name__ == "__main__":
    main()
