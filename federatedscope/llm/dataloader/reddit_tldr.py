import os
import json
import copy
import pickle

from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonls, load_jsonl
from federatedscope.llm.dataset.llm_dataset import DefaultToken, \
    LLMDataset, LLMComparisonDataset

TLDR_PROMPT_DICT = {
    # "summary": ("Below is a forum post. Write a precise and concise summary "
    #             "that includes the most important points of the post.\n\n"
    #             "### SUBREDDIT: r/{subreddit}\n"
    #             "### TITLE: {title}\n"
    #             "### POST: {post}\n"
    #             "### TL;DR:"),
    "summary": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nSummarize the following Reddit post in "
        "a paragraph of 50 words or less.\n\n"
        "### Input:\n"
        "SUBREDDIT: r/{subreddit}\n"
        "TITLE: {title}\n"
        "POST: {post}\n\n"
        "### Response:"),
    "summary_cmp": (
        "Below is a forum post followed by two summaries. "
        "Pick a more precise and concise one that summarizes the most "
        "important points in the given forum post, without including "
        "unimportant or irrelevant details. State your choice with a "
        "single capital letter, i.e., \"A\" if SUMMARY A is better, "
        "\"B\" if SUMMARY B is better.\n\n"
        "### SUBREDDIT: r/{subreddit}\n"
        "### TITLE: {title}\n"
        "### POST: {post}\n"
        "### SUMMARY A:{output_A}\n"
        "### SUMMARY B:{output_B}\n"
        "### YOUR CHOICE:"),
    "mix_cmp": ("Below is an instruction that describes a task, "
                "paired with an input that provides further context. "
                "There are two responses that complete the request. "
                "Pick an appropriate response and state your choice with "
                "a single capital letter, i.e., "
                "\"A\" if RESPONSE A is better and more appropriate, "
                "\"B\" if RESPONSE B is better and more appropriate.\n\n"
                "### Instruction:\nSummarize the following Reddit post.\n\n"
                "### Input:\n"
                "SUBREDDIT: r/{subreddit}\n"
                "TITLE: {title}\n"
                "POST: {post}\n\n"
                "### RESPONSE A: {output_A}\n"
                "### RESPONSE B: {output_B}\n"
                "### YOUR CHOICE:")
}


def _download_tldr_cmpr(data_root):
    all_files = [f'batch{i}' for i in range(3, 21)] + ['batch22']
    # all_files = ['batch8']

    for cmp_file in all_files:
        download_url(
            'https://openaipublic.blob.core.windows.net/'
            'summarize-from-feedback/dataset/comparisons/'
            f'{cmp_file}.json', data_root)

    # Preprocess the above data
    file_paths = [
        os.path.join(data_root, f'{cmp_file}.json') for cmp_file in all_files
    ]
    list_data_dict = load_jsonls(file_paths,
                                 subreddit='info.subreddit',
                                 title='info.title',
                                 post='info.post',
                                 output_A='summaries.0.text',
                                 output_B='summaries.1.text',
                                 category='worker',
                                 split='split',
                                 choice='choice')
    # Split the dataset into ['train', 'val', 'test']
    list_train_dict, list_val_dict, list_test_dict = [], [], []
    for data_dict in list_data_dict:
        if data_dict['split'] == 'train':
            list_train_dict.append(data_dict)
        elif data_dict['split'] == 'valid1':
            list_val_dict.append(data_dict)
        elif data_dict['split'] == 'valid2':
            list_test_dict.append(data_dict)

    # # merge the worker with less than 10 samples
    # dict_cat = {}
    # for idx, sample in enumerate(list_train_dict):
    #     if sample['category'] not in dict_cat:
    #         dict_cat[sample['category']] = [idx]
    #     else:
    #         dict_cat[sample['category']].append(idx)
    # for values in dict_cat.values():
    #     if len(values) < 50:
    #         for idx in values:
    #             list_train_dict[idx]['category'] = 'merged_client'

    return list_train_dict, list_val_dict, list_test_dict


def _download_tldr_human(data_root):
    train_fp, val_fp, test_fp = [
        os.path.join(data_root, 'reddit-tldr_train.jsonl'),
        os.path.join(data_root, 'reddit-tldr_val.jsonl'),
        os.path.join(data_root, 'reddit-tldr_test.jsonl')
    ]

    for name, fp in [('train', train_fp), ('valid', val_fp),
                     ('test', test_fp)]:
        if not os.path.exists(fp):
            download_url(
                'https://openaipublic.blob.core.windows.net/'
                'summarize-from-feedback/datasets/'
                f'tldr_3_filtered/{name}.jsonl', data_root)
            os.rename(os.path.join(data_root, f'{name}.jsonl'), fp)

    dataloader_kwargs = {
        'subreddit': 'subreddit',
        'title': 'title',
        'post': 'post',
        'summary': 'summary'
    }
    list_train_dict = load_jsonl(train_fp, **dataloader_kwargs)
    list_val_dict = load_jsonl(val_fp, **dataloader_kwargs)
    list_test_dict = load_jsonl(test_fp, **dataloader_kwargs)

    return list_train_dict, list_val_dict, list_test_dict


def _tldr_human_for_prtraining(data_root):
    train_fp, val_fp, test_fp = [
        os.path.join(data_root, 'reddit-tldr_train_finetune.jsonl'),
        os.path.join(data_root, 'reddit-tldr_val_finetune.jsonl'),
        os.path.join(data_root, 'reddit-tldr_test_finetune.jsonl')
    ]

    dataloader_kwargs = {
        'subreddit': 'subreddit',
        'title': 'title',
        'post': 'post',
        'summary': 'summary'
    }
    if os.path.exists(train_fp) and os.path.exists(val_fp) and \
            os.path.exists(test_fp):
        list_train_dict = load_jsonl(train_fp, **dataloader_kwargs)
        list_val_dict = load_jsonl(val_fp, **dataloader_kwargs)
        list_test_dict = load_jsonl(test_fp, **dataloader_kwargs)

    else:
        h_train, h_val, h_test = _download_tldr_human(data_root)
        c_train, c_val, c_test = _download_tldr_cmpr(data_root)

        # get a full list of comparison data
        c_posts = []
        for list_dict in [c_train, c_val, c_test]:
            for sample in list_dict:
                if sample['post'] not in c_posts:
                    c_posts.append(sample['post'])

        # remove the comparison data in human dataset
        list_train_dict = [s for s in h_train if s['post'] not in c_posts]
        list_val_dict = [s for s in h_val if s['post'] not in c_posts]
        list_test_dict = [s for s in h_test if s['post'] not in c_posts]

        # Add a space to the start of a summary, and save to file
        for fp, list_dict in [(train_fp, list_train_dict),
                              (val_fp, list_val_dict),
                              (test_fp, list_test_dict)]:
            with open(fp, "w") as file:
                for sample in list_dict:
                    sample["summary"] = " " + sample["summary"]
                    file.write(json.dumps(sample) + "\n")

    return list_train_dict, list_val_dict, list_test_dict


def get_tldr_dataset(list_data_dict,
                     tokenizer,
                     prompt=TLDR_PROMPT_DICT['summary']):
    return LLMDataset(list_data_dict,
                      tokenizer,
                      prompt_input=prompt,
                      prompt_no_input=prompt,
                      output_tag='summary')


def load_human_annotated_dataset(data_root, tokenizer):
    list_train_dict, list_val_dict, list_test_dict = \
        _download_tldr_human(data_root)

    train_dataset = LLMDataset(list_train_dict,
                               tokenizer,
                               prompt_input=TLDR_PROMPT_DICT['summary'],
                               prompt_no_input=TLDR_PROMPT_DICT['summary'],
                               output_tag='summary')
    val_dataset = LLMDataset(list_val_dict,
                             tokenizer,
                             prompt_input=TLDR_PROMPT_DICT['summary'],
                             prompt_no_input=TLDR_PROMPT_DICT['summary'],
                             output_tag='summary')
    test_dataset = LLMDataset(list_test_dict,
                              tokenizer,
                              prompt_input=TLDR_PROMPT_DICT['summary'],
                              prompt_no_input=TLDR_PROMPT_DICT['summary'],
                              output_tag='summary')

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def load_human_finetuning_dataset(data_root,
                                  tokenizer,
                                  rlhf=False,
                                  max_num_test=-1,
                                  raw_no_prompt=False):
    list_train_dict, list_val_dict, list_test_dict = \
        _tldr_human_for_prtraining(data_root)

    # First 60% for fine-tuning, last 40% for rlhf
    idx = int(len(list_train_dict) * 0.6)
    list_train_dict = list_train_dict[:idx] if not rlhf else \
        list_train_dict[idx:]
    if raw_no_prompt:
        if max_num_test > 0:
            return (list_train_dict, list_val_dict[:max_num_test],
                    list_test_dict[:max_num_test])
        else:
            return list_train_dict, list_val_dict, list_test_dict

    train_dataset = LLMDataset(list_train_dict,
                               tokenizer,
                               prompt_input=TLDR_PROMPT_DICT['summary'],
                               prompt_no_input=TLDR_PROMPT_DICT['summary'],
                               output_tag='summary')
    val_dataset = LLMDataset(list_val_dict,
                             tokenizer,
                             prompt_input=TLDR_PROMPT_DICT['summary'],
                             prompt_no_input=TLDR_PROMPT_DICT['summary'],
                             output_tag='summary')
    test_dataset = LLMDataset(list_test_dict,
                              tokenizer,
                              prompt_input=TLDR_PROMPT_DICT['summary'],
                              prompt_no_input=TLDR_PROMPT_DICT['summary'],
                              output_tag='summary')

    # shrink val and test dataset
    if max_num_test > 0:
        val_dataset.input_ids = val_dataset.input_ids[:max_num_test]
        test_dataset.input_ids = test_dataset.input_ids[:max_num_test]

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def load_comparison_dataset(data_root, tokenizer, max_num_test=-1):
    token_name = os.path.basename(tokenizer.name_or_path)
    train_set_path = os.path.join(data_root, f'{token_name}_train.pickle')
    val_set_path = os.path.join(data_root, f'{token_name}_val.pickle')
    test_set_path = os.path.join(data_root, f'{token_name}_test.pickle')
    if os.path.exists(train_set_path) and os.path.exists(val_set_path) \
            and os.path.exists(test_set_path):
        with open(train_set_path, 'rb') as f_train, \
                open(val_set_path, 'rb') as f_val, \
                open(test_set_path, 'rb') as f_test:
            train_dataset = pickle.load(f_train)
            val_dataset = pickle.load(f_val)
            test_dataset = pickle.load(f_test)

    else:
        list_train_dict, list_val_dict, list_test_dict = \
            _download_tldr_cmpr(data_root)

        # load dataset, which should be tuple
        train_dataset = LLMComparisonDataset(
            list_train_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary'],
            prompt_no_input=TLDR_PROMPT_DICT['summary'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')
        val_dataset = LLMComparisonDataset(
            list_val_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary'],
            prompt_no_input=TLDR_PROMPT_DICT['summary'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')
        test_dataset = LLMComparisonDataset(
            list_test_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary'],
            prompt_no_input=TLDR_PROMPT_DICT['summary'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')

        # Store these three lists to a pickle file
        with open(train_set_path, 'wb') as f_train, \
                open(val_set_path, 'wb') as f_val, \
                open(test_set_path, 'wb') as f_test:
            pickle.dump(train_dataset, f_train)
            pickle.dump(val_dataset, f_val)
            pickle.dump(test_dataset, f_test)

    # shrink val and test dataset
    if max_num_test > 0:
        val_dataset.win_dataset.input_ids = \
            val_dataset.win_dataset.input_ids[:max_num_test]
        val_dataset.lose_dataset.input_ids = \
            val_dataset.lose_dataset.input_ids[:max_num_test]
        test_dataset.win_dataset.input_ids = \
            test_dataset.win_dataset.input_ids[:max_num_test]
        test_dataset.lose_dataset.input_ids = \
            test_dataset.lose_dataset.input_ids[:max_num_test]

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def load_best_dataset(data_root, tokenizer, max_num_test=-1):
    train_dataset, val_dataset, test_dataset = \
        load_comparison_dataset(data_root, tokenizer, max_num_test)
    # Use the win_dataset only
    dataset = (train_dataset.win_dataset, val_dataset.win_dataset,
               test_dataset.win_dataset)
    return dataset


def load_comparison_dataset_by_choice(data_root, tokenizer, max_num_test=-1):
    token_name = os.path.basename(tokenizer.name_or_path)
    train_set_path = os.path.join(data_root,
                                  f'{token_name}_train_choice.pickle')
    val_set_path = os.path.join(data_root, f'{token_name}_val_choice.pickle')
    test_set_path = os.path.join(data_root, f'{token_name}_test_choice.pickle')
    if os.path.exists(train_set_path) and os.path.exists(val_set_path) and \
            os.path.exists(test_set_path):
        with open(train_set_path, 'rb') as f_train, \
                open(val_set_path, 'rb') as f_val, \
                open(test_set_path, 'rb') as f_test:
            train_dataset = pickle.load(f_train)
            val_dataset = pickle.load(f_val)
            test_dataset = pickle.load(f_test)

    else:
        list_train_dict, list_val_dict, list_test_dict = \
            _download_tldr_cmpr(data_root)

        # For training dataset, we should exchange the order
        # and append the new training dataset to the list_train_dict
        exchange_list_train_dict = copy.deepcopy(list_train_dict)
        for sample in exchange_list_train_dict:
            sample['output_A'], sample['output_B'] = \
                sample['output_B'], sample['output_A']
            sample['choice'] = 1 - sample['choice']
        list_train_dict = list_train_dict + exchange_list_train_dict

        # map the choice to "A" and "B" instead of 0 and 1
        for list_dict in [list_train_dict, list_test_dict, list_val_dict]:
            for sample in list_dict:
                sample['choice'] = " " + chr(sample['choice'] + ord("A"))

        train_dataset = LLMDataset(
            list_train_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary_cmp'],
            prompt_no_input=TLDR_PROMPT_DICT['summary_cmp'],
            output_tag='choice')
        val_dataset = LLMDataset(
            list_val_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary_cmp'],
            prompt_no_input=TLDR_PROMPT_DICT['summary_cmp'],
            output_tag='choice')
        test_dataset = LLMDataset(
            list_test_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary_cmp'],
            prompt_no_input=TLDR_PROMPT_DICT['summary_cmp'],
            output_tag='choice')

        # Store these three lists to a pickle file
        with open(train_set_path, 'wb') as f_train, \
                open(val_set_path, 'wb') as f_val, \
                open(test_set_path, 'wb') as f_test:
            pickle.dump(train_dataset, f_train)
            pickle.dump(val_dataset, f_val)
            pickle.dump(test_dataset, f_test)

    # shrink val and test dataset
    if max_num_test > 0:
        val_dataset.input_ids = val_dataset.input_ids[:max_num_test]
        test_dataset.input_ids = test_dataset.input_ids[:max_num_test]

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def check_sim(data_root):
    cmpr_list_train_dict, cmpr_list_val_dict, cmpr_list_test_dict = \
        _download_tldr_cmpr(data_root)

    human_list_train_dict, human_list_val_dict, human_list_test_dict = \
        _download_tldr_human(data_root)

    # show if human-annotated overlaps cmpr in terms of train_dict
    cmpr_train = [sample['post'] for sample in cmpr_list_val_dict]
    human_train = [sample['post'] for sample in human_list_train_dict]

    print(len(cmpr_train))  # 92858
    print(len(human_train))  # 116722

    total_overlapping = 0

    for data in cmpr_train:
        if data in human_train:
            total_overlapping += 1
            human_train.pop(human_train.index(data))

    print(len(human_train))
    print(total_overlapping)  # 59685/282/475


if __name__ == "__main__":
    data_root = os.path.join('/local/scratch/d/wu1977/dataset/',
                             'reddit-tldr-comparison')
    check_sim(data_root)
