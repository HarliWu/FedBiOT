import re
import torch
import os
import random
import evaluate
import transformers
from transformers import GenerationConfig
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonl, load_jsonls
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
    # raw_fschatbot = FSChatBot(init_cfg, use_raw=True)

    # Get test file
    fp = os.path.join(init_cfg.data.root, 'reddit-tldr_test.jsonl')
    if not os.path.exists(fp):
        download_url(
            'https://openaipublic.blob.core.windows.net/'
            'summarize-from-feedback/datasets/'
            'tldr_3_filtered/test.jsonl', init_cfg.data.root)
        os.rename(os.path.join(init_cfg.data.root, 'test.jsonl'), fp)

    list_data_dict = load_jsonl(fp,
                                subreddit='subreddit',
                                title='title',
                                post='post',
                                summary='summary')

    prompt = ("Below is a forum post. Write a precise and concise summary "
              "that includes the most important points of the post.\n\n"
              "### Subreddit:\n{subreddit}\n\n### Title:\n{title}\n\n"
              "### Post:\n{post}\n\n### TL; DR:")

    while True:
        try:
            results_display = os.path.join(
                init_cfg.outdir, f'{fschatbot.curpfx}_summarization.txt')
            results_display = open(results_display, 'w')
            predictions, references = [], []

            for sample in tqdm(list_data_dict):
                input_text = prompt.format_map(sample)
                generate_kwargs = dict(top_k=0,
                                       top_p=1.0,
                                       do_sample=True,
                                       max_new_tokens=56,
                                       repetition_penalty=1.15,
                                       temperature=0.6,
                                       max_length=init_cfg.llm.chat.max_len)
                model_completion = fschatbot.generate(input_text,
                                                      generate_kwargs)

                results_display.write(
                    f'Post:\n{sample["post"]}\n\n'
                    f'Human summary:\n{sample["summary"]}\n\n'
                    f'Model-generated summary:\n{model_completion}\n\n')

                results_display.write('==========================\n\n')
                results_display.flush()

                predictions.append(model_completion)
                references.append(sample["summary"])

            bleu = evaluate.load("bleu")
            bleu_results = bleu.compute(predictions=predictions,
                                        references=[[ref]
                                                    for ref in references])
            print(bleu_results)
            results_display.write(f'bleu: {bleu_results}\n\n')

            rouge = evaluate.load("rouge")
            rouge_results = rouge.compute(predictions=predictions,
                                          references=references)
            print(rouge_results)
            results_display.write(f'rouge: {rouge_results}\n\n')

            results_display.close()

            print('load the next model...')
            fschatbot.next_model()
        except Exception as err:
            print(f'{err}, so finished all evaluations....')
            break


if __name__ == "__main__":
    main()
