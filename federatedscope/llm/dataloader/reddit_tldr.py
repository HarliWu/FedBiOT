import os
import pickle

from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonls, load_jsonl
from federatedscope.llm.dataset.llm_dataset import DefaultToken, \
    LLMDataset, LLMComparisonDataset

TLDR_PROMPT_DICT = {
    "summary": ("Below is a forum post. Write a precise and concise summary "
                "that includes the most important points of the post.\n\n"
                "### Subreddit:\n{subreddit}\n\n### Title:\n{title}\n\n"
                "### Post:\n{post}\n\n### TL; DR:"),
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
        "### SUMMARY A:{summary_A}\n"
        "### SUMMARY B:{summary_B}\n"
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
                                 summary_A='summaries.0.text',
                                 summary_B='summaries.1.text',
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

    return list_train_dict, list_val_dict, list_test_dict


def _download_tldr_human(data_root):
    train_fp, valid_fp, test_fp = [
        os.path.join(data_root, 'reddit-tldr_train.jsonl'),
        os.path.join(data_root, 'reddit-tldr_valid.jsonl'),
        os.path.join(data_root, 'reddit-tldr_test.jsonl')
    ]

    for name, fp in [('train', train_fp), ('valid', valid_fp),
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
    list_val_dict = load_jsonl(valid_fp, **dataloader_kwargs)
    list_test_dict = load_jsonl(test_fp, **dataloader_kwargs)

    return list_train_dict, list_val_dict, list_test_dict


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


def load_comparison_dataset(data_root, tokenizer):
    if os.path.exists(os.path.join(data_root, 'train.pickle')) and \
        os.path.exists(os.path.join(data_root, 'val.pickle')) and \
            os.path.exists(os.path.join(data_root, 'test.pickle')):
        with open(os.path.join(data_root, 'train.pickle'), 'rb') as f_train, \
            open(os.path.join(data_root, 'val.pickle'), 'rb') as f_val, \
                open(os.path.join(data_root, 'test.pickle'), 'rb') as f_test:
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
            output_A='summary_A',
            output_B='summary_B',
            choice='choice')
        val_dataset = LLMComparisonDataset(
            list_val_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary'],
            prompt_no_input=TLDR_PROMPT_DICT['summary'],
            output_A='summary_A',
            output_B='summary_B',
            choice='choice')
        test_dataset = LLMComparisonDataset(
            list_test_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary'],
            prompt_no_input=TLDR_PROMPT_DICT['summary'],
            output_A='summary_A',
            output_B='summary_B',
            choice='choice')

        # Store these three lists to a pickle file
        with open(os.path.join(data_root, 'train.pickle'), 'wb') as f_train, \
            open(os.path.join(data_root, 'val.pickle'), 'wb') as f_val, \
                open(os.path.join(data_root, 'test.pickle'), 'wb') as f_test:
            pickle.dump(train_dataset, f_train)
            pickle.dump(val_dataset, f_val)
            pickle.dump(test_dataset, f_test)

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def load_comparison_dataset_by_choice(data_root, tokenizer, max_num_test=-1):
    if os.path.exists(os.path.join(data_root, 'train_choice.pickle')) and \
        os.path.exists(os.path.join(data_root, 'val_choice.pickle')) and \
            os.path.exists(os.path.join(data_root, 'test_choice.pickle')):
        with open(os.path.join(data_root,
                               'train_choice.pickle'), 'rb') as f_train, \
            open(os.path.join(data_root,
                              'val_choice.pickle'), 'rb') as f_val, \
                open(os.path.join(data_root,
                                  'test_choice.pickle'), 'rb') as f_test:
            train_dataset = pickle.load(f_train)
            val_dataset = pickle.load(f_val)
            test_dataset = pickle.load(f_test)

    else:
        list_train_dict, list_val_dict, list_test_dict = \
            _download_tldr_cmpr(data_root)

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
        with open(os.path.join(data_root,
                               'train_choice.pickle'), 'wb') as f_train, \
            open(os.path.join(data_root,
                              'val_choice.pickle'), 'wb') as f_val, \
                open(os.path.join(data_root,
                                  'test_choice.pickle'), 'wb') as f_test:
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
    cmpr_train = [sample['post'] for sample in cmpr_list_test_dict]
    human_train = [sample['post'] for sample in human_list_train_dict]

    print(len(cmpr_train))  # 92858
    print(len(human_train))  # 116722

    total_overlapping = 0

    for data in cmpr_train:
        if data in human_train:
            total_overlapping += 1

    print(total_overlapping)  # 59685/282/475


if __name__ == "__main__":
    data_root = os.path.join('/local/scratch/d/wu1977/dataset/',
                             'reddit-tldr-comparison')
    check_sim(data_root)
