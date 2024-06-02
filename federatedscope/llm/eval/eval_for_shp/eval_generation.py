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
from federatedscope.llm.dataloader.shp import _download_shp, SHP_PROMPT_DICT
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

    # get SHP test prompt
    _, _, list_data_dict = \
        _download_shp(os.path.join(init_cfg.data.root, 'shp'))

    prompt = SHP_PROMPT_DICT["shp"]

    try:
        results_display = os.path.join(
            init_cfg.outdir, f'{fschatbot.curpfx}_summarization.txt')
        results_display = open(results_display, 'w')

        for sample in tqdm(list_data_dict):
            input_text = prompt.format_map(sample)
            generate_kwargs = dict(
                top_p=1.0,
                temperature=0.0,
                do_sample=False,
                max_new_tokens=init_cfg.llm.max_new_token,
            )
            model_completions = fschatbot.generate(input_text, generate_kwargs)

            results_display.write(
                f'Instruction:\n{sample["instruction"]}\n\n'
                f'Model-generated response [[0]]:\n{model_completions}\n\n')

            results_display.write('==========================\n\n')
            results_display.flush()

    except Exception as err:
        print(f'{err}, so finished all evaluations....')


if __name__ == "__main__":
    main()
