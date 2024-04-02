import torch
import random
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
from federatedscope.llm.dataloader.reddit_tldr import TLDR_PROMPT_DICT
from federatedscope.llm.dataset.llm_dataset import DefaultToken, \
    LLMDataset
from federatedscope.llm.eval.eval_for_tldr.auto_j_vllm import evaluation

logger = None


@torch.no_grad()
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

    prompt = TLDR_PROMPT_DICT["summary"]

    results_display = os.path.join(gen_cfg.outdir,
                                   f'{fschatbot.curpfx}_summarization.txt')
    results_display = open(results_display, 'w')

    for sample in tqdm(list_data_dict):
        input_text = prompt.format_map(sample)
        generate_kwargs = dict(
            top_p=1.0,
            temperature=1.0,
            do_sample=True,
            max_new_tokens=60,
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


@torch.no_grad()
def best_of_n_dataset(init_cfg, gen_cfg=None, n=16):
    gen_fp = os.path.join(init_cfg.data.root, 'reddit-tldr-comparison',
                          f'reddit-tldr_test_{n}-gen.json')
    if os.path.exists(gen_fp):
        # load the dataset
        list_data_dict = json.load(open(gen_fp, "r"))
    else:
        # create the dataset
        list_data_dict = _generate_best_of_n_dataset(gen_cfg, n)
        json.dump(list_data_dict, open(gen_fp, "w"))
    return list_data_dict


@torch.no_grad()
def cal_acc(logits, labels, choices):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    new_labels = torch.full_like(shift_labels, DefaultToken.IGNORE_INDEX.value)
    for idx, choice in enumerate(choices):
        new_labels[shift_labels == choice] = idx

    new_labels = new_labels.view(-1)
    new_logits = shift_logits[..., choices].view(-1, len(choices))
    new_logits = new_logits[(new_labels != DefaultToken.IGNORE_INDEX.value), :]
    # print(new_logits)
    new_labels = new_labels[(new_labels != DefaultToken.IGNORE_INDEX.value)]
    _, predicted = new_logits.max(1)

    return new_labels, new_logits, predicted, predicted.eq(
        new_labels).sum().item()


@torch.no_grad()
def best_of_n(model, dataset, tokenizer, n=16, output_dir=None):
    prompt = TLDR_PROMPT_DICT["summary_cmp"]

    choices = [tokenizer(f'{c}')['input_ids'][-1] for c in ['A', 'B']]
    # Correct idx is 0 means no changed, 1 means changed to the new index
    last_better_idx = np.array([0] * len(dataset))
    for i in range(1, n):
        logger.info(f'===== This is {i}-th evaluation =====')
        eval_dataset = []

        for better_idx, sample in zip(last_better_idx, dataset):
            sample['summary_A'] = sample['summaries'][better_idx]
            if sample['summary_A'].startswith(" ") is False:
                sample['summary_A'] = " " + sample['summary_A']
            sample['summary_B'] = sample['summaries'][i]
            if sample['summary_B'].startswith(" ") is False:
                sample['summary_B'] = " " + sample['summary_B']
            sample['choice'] = random.choice([" A", " B"])
            eval_dataset.append({
                'subreddit': sample['subreddit'],
                'title': sample['title'],
                'post': sample['post'],
                'summary_A': sample['summary_B'],
                'summary_B': sample['summary_A'],
                'choice': sample['choice']
            })

        test_dataset = LLMDataset(eval_dataset,
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
        # results_display = open(os.path.join(output_dir,
        #                                     f'iter_{i}.txt'), 'w')
        for idx, data_batch in enumerate(tqdm(dataloader)):
            input_ids = data_batch["input_ids"].to('cuda:0')
            labels = data_batch["labels"].to('cuda:0')
            attention_mask = data_batch["attention_mask"].to('cuda:0')
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, new_logits, predicted, _ = cal_acc(outputs.logits, labels,
                                                  choices)
            predicted_indices += predicted.tolist()

            # results_display.write(f'{new_logits}\n\n')
            # results_display.write(f'Input ID: {input_ids.shape} '
            #                       f'Labels shape: {labels.shape} '
            #                       f'Outputs shape: {outputs.logits.shape}\n')
            # results_display.flush()

            # sample = eval_dataset[idx]
            # results_display.write(
            # f'Post:\n{sample["post"]}\n\n'
            # f'Summary A:\n{sample["summary_A"]}\n\n'
            # f'Summary B:\n{sample["summary_B"]}\n\n'
            # f'Model-choice: {chr(predicted[0] + ord("A"))}\n\n')
            # results_display.write('==========================\n\n')
            # results_display.flush()

        # results_display.close()

        # logger.info(f'Last better index: {list(last_better_idx)} '
        #             f'({len(last_better_idx)})')
        # logger.info(f'Predicted indices: {predicted_indices} '
        #             f'({len(predicted_indices)})')
        predicted_indices = np.array(predicted_indices)
        last_better_idx[predicted_indices == 0] = i

    # print the final results
    return last_better_idx


def best_of_n_local(model,
                    dataset,
                    tokenizer,
                    n=16,
                    num_clients=1,
                    output_dir=None):
    clients_best_idx = []
    for client_id in range(num_clients):
        logger.info(f'============ Client {client_id+1} ============')
        model.set_active_adapter(f'Client_{client_id+1}')
        clients_best_idx.append(
            best_of_n(model, dataset, tokenizer, n, output_dir))

    # Choose the indices with most votes
    array = np.array(clients_best_idx).T
    majority_votes_idx = [
        np.bincount(array[i]).argmax() for i in range(len(array))
    ]

    return clients_best_idx, majority_votes_idx


def print_results(results_display, dataset, bsn_results):
    for best_idx, sample in zip(bsn_results, dataset):
        results_display.write(f'Subreddit: r/{sample["subreddit"]}\n\n'
                              f'Title:\n{sample["title"]}\n\n'
                              f'Post:\n{sample["post"]}\n\n'
                              f'Best generated summary [[{best_idx}]]:\n'
                              f'{sample["summaries"][best_idx]}\n\n')
        results_display.write('==========================\n\n')
        results_display.flush()


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

    import logging
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Load the generation config
    gen_cfg = init_cfg.clone()
    if gen_args.gen_cfg_file:
        gen_cfg.merge_from_file(gen_args.gen_cfg_file)
    gen_cfg.freeze(save=False)

    init_cfg.freeze()

    # best_of_n dataset
    dataset = best_of_n_dataset(init_cfg, gen_cfg, n=16)
    # dataset = dataset[:1000]

    # get model and tokenizer
    model_name, _ = init_cfg.model.type.split('@')
    model = get_llm(init_cfg, device_map='auto')
    tokenizer, _ = get_tokenizer(model_name, init_cfg.data.root,
                                 init_cfg.llm.tok_len)

    # load model from checkpoint
    num_ckpt = \
        init_cfg.federate.total_round_num // init_cfg.federate.save_freq
    prefix = ['final_'] + \
             [str(i*init_cfg.federate.save_freq) + '_'
              for i in range(num_ckpt, -1, -1)] + ['']
    dirname, filename = os.path.split(init_cfg.federate.save_to)
    for pre in prefix:
        print(os.path.join(dirname, pre + filename))
        if os.path.exists(os.path.join(dirname, pre + filename)):
            ckpt_path = os.path.join(dirname, pre + filename)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model'])
            print(f'Model of Round {ckpt["cur_round"]} loads '
                  f'from the checkpoint {ckpt_path}')
            break
    # model = model.merge_and_unload()

    # get the best-of-n results and display them
    if init_cfg.llm.adapter.local_only:
        clients_results, results = \
            best_of_n_local(model, dataset, tokenizer, n=16,
                            num_clients=init_cfg.federate.client_num,
                            output_dir=init_cfg.outdir)
        for i in range(init_cfg.federate.client_num):
            path = os.path.join(init_cfg.outdir,
                                f'test_results_client_{i+1}.txt')
            print_results(open(path, 'w'), dataset, clients_results[i])
            # evaluate best-of-n selection using auto_j
            # evaluation(path)
    else:
        results = best_of_n(model,
                            dataset,
                            tokenizer,
                            n=16,
                            output_dir=init_cfg.outdir)

    path = os.path.join(init_cfg.outdir, 'test_results.txt')
    print_results(open(path, 'w'), dataset, results)
    # evaluate best-of-n selection using auto_j
    # evaluation(path)


if __name__ == "__main__":
    main()
