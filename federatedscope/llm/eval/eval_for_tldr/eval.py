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
    fp = os.path.join(init_cfg.data.root, 'reddit-tldr-comparison',
                      'test.jsonl')
    if not os.path.exists(fp):
        download_url(
            'https://openaipublic.blob.core.windows.net/'
            'summarize-from-feedback/datasets/tldr_3_filtered/test.jsonl',
            os.path.join(init_cfg.data.root, 'reddit-tldr-comparison'))
    list_data_dict = load_jsonl(fp,
                                title='title',
                                post='post',
                                category='subreddit',
                                summary='summary')

    prompt = ("Below is a forum post. Write a precise and concise summary "
              "that includes the most important points of the post.\n\n"
              "### Subreddit:\n{category}\n\n### Title:\n{title}\n\n"
              "### Post:\n{post}\n\n### TL; DR:")

    # Print result to a text file
    results_display = open(os.path.join(init_cfg.outdir, 'test_results.txt'),
                           'w')
    testset = tqdm(list_data_dict)
    for sample in testset:
        input_text = prompt.format_map(sample)
        generation_config = GenerationConfig(
            temperature=0.6,
            early_stopping=True,
            num_beams=2,
            no_repeat_ngram_size=2,
            do_sample=True,
        )
        generate_kwargs = dict(
            generation_config=generation_config,
            max_new_tokens=128,
        )
        model_completion = fschatbot.generate(input_text, generate_kwargs)
        # raw_model_completion = raw_fschatbot.generate(input_text,
        #                                               generate_kwargs)

        results_display.write(
            f'Post:\n{sample["post"]}\n\n'
            f'Human summary:\n{sample["summary"]}\n\n'
            # f'Raw model-generated summary:\n{raw_model_completion}\n\n'
            f'Model-generated summary:\n{model_completion}\n\n')

        results_display.write('==========================\n\n')
        results_display.flush()

    results_display.close()


if __name__ == "__main__":
    main()
