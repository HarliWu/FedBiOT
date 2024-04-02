import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from federatedscope.llm.eval.eval_for_tldr.auto_j_constants_prompt \
    import build_autoj_input


def extract_single_rating(score_output):
    pred_score = 0.0
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        pred_score = float(score_output[pos + len("Rating: [["):pos2].strip())
    return pred_score


@torch.no_grad()
def auto_j_eval_rating(dataset):
    model = AutoModelForCausalLM.from_pretrained("GAIR/autoj-13b",
                                                 device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained("GAIR/autoj-13b")

    generate_kwargs = dict(temperature=0.0,
                           top_p=1.0,
                           do_sample=False,
                           max_new_tokens=1024)

    auto_j_comments, auto_j_ratings = [], []
    all_samples = tqdm(dataset)
    for sample in all_samples:
        input_text = build_autoj_input(prompt=sample['query'],
                                       resp1=sample['response'],
                                       resp2=None,
                                       protocol="single")
        input_tokens = tokenizer(input_text, return_tensors="pt")
        input_ids = input_tokens.input_ids.to('cuda:0')
        attention_mask = input_tokens.attention_mask.to('cuda:0')
        output_ids = model.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    **generate_kwargs)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:],
                                    skip_special_tokens=True,
                                    ignore_tokenization_space=True)
        auto_j_comments.append(response)
        rate = extract_single_rating(response)
        auto_j_ratings.append(rate)

        all_samples.set_postfix({
            'zeros': np.count_nonzero(np.array(auto_j_ratings) == 0),
            'avg_ratings': np.mean(auto_j_ratings)
        })

    return auto_j_comments, auto_j_ratings


def read_file(path='test_results.txt'):
    f = open(path, 'r', encoding="utf8")
    colletion = []

    tag = ''
    record = {'subreddit': '', 'title': '', 'post': '', 'response': ''}
    for line in f.readlines():
        if 'Subreddit:' in line:
            record['subreddit'] = line.replace('Subreddit: ', '')
            tag = 'subreddit'

        elif 'Title:' in line:
            tag = 'title'

        elif 'Post:' in line:
            tag = 'post'

        elif 'generated summary' in line:
            tag = 'response'

        elif '=============' in line:
            # The end of the record, which should be
            # appended to the collection
            for key in record.keys():
                if record[key].endswith('\n\n'):
                    record[key] = record[key][:-2]
            query = ("Summarize the following post\n\n"
                     "Title: {title}\n\n"
                     "Post: {post}").format_map(record)
            record['query'] = query
            colletion.append(record)
            record = {'subreddit': '', 'title': '', 'post': '', 'response': ''}

        else:
            # This is a normal line and should be
            # extended to current tag of record
            record[tag] = record[tag] + line

    return colletion


def evaluation(file_path):
    dataset = read_file(file_path)
    auto_j_comments, auto_j_ratings = auto_j_eval_rating(dataset)

    # print the evaluation results
    with open(file_path + '_autoj_eval.txt', 'w') as f:
        for sample, comment, rating in zip(dataset, auto_j_comments,
                                           auto_j_ratings):
            f.write(f'Subreddit: {sample["subreddit"]}\n\n'
                    f'Title:\n{sample["title"]}\n\n'
                    f'Post:\n{sample["post"]}\n\n'
                    f'Best generated summary:\n{sample["response"]}\n\n'
                    f'Auto-J Comment:\n{comment}\n\n'
                    f'Auto-J Rating: {rating}\n\n')
            f.write('==========================\n\n')
            f.flush()
        f.write(f'{auto_j_ratings}\n\n')
        f.write(f'Average Auto-J Rating: {np.mean(auto_j_ratings)}\n\n')

    return auto_j_ratings, np.mean(auto_j_ratings)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-outputs-file',
                        dest='file',
                        help='Path to model-generated outputs',
                        type=str)
    args = parser.parse_args()

    evaluation(args.file)
