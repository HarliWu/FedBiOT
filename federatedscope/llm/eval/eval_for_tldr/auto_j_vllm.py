import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from vllm import LLM, SamplingParams
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
    num_gpus = torch.cuda.device_count()
    model_name_or_dir = "GAIR/autoj-13b"
    llm = LLM(model=model_name_or_dir, tensor_parallel_size=num_gpus)

    sampling_params = SamplingParams(temperature=0.0,
                                     top_p=1.0,
                                     max_tokens=1024)

    auto_j_comments, auto_j_ratings = [], []

    inputs = []

    for sample in tqdm(dataset):
        input_text = build_autoj_input(prompt=sample['query'],
                                       resp1=sample['response'],
                                       resp2=None,
                                       protocol="single")
        inputs.append(input_text)

    outputs = llm.generate(inputs, sampling_params)
    auto_j_comments = [item.outputs[0].text for item in outputs]
    auto_j_ratings = [
        extract_single_rating(judgement) for judgement in auto_j_comments
    ]

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
