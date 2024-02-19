# ref: https://github.com/mit-han-lab/offsite-tuning/blob/
# main/offsite_tuning/eval_harness.py

import torch

from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot


class LMEvaluator(HFLM):
    def __init__(self, fschatbot:FSChatBot):
        super().__init__(pretrained=fschatbot.model, 
                         backend="causal",
                         tokenizer=fschatbot.tokenizer)


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

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to
    fschatbot = FSChatBot(init_cfg)

    # Load to the evaluator 
    results = evaluator.simple_evaluate(
        model=LMEvaluator(fschatbot), 
        task=['hellaswag'], 
        write_out=True
    )

    
if __name__ == "__main__":
    main()
